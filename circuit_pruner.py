"""
ZITEP Circuit Pruner — Phase 5: Spectral Pruning and Deployment.

Takes the verified topological bridges and creates a binary mask over the full
model weights. Parameters inside verified circuits get a mask value of 1;
everything else gets 0. The mask is applied via element-wise multiplication,
physically deleting all parameters outside the verified reasoning pathway.

The resulting pruned model is exported as a compact safetensors artifact.
"""

import os
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import numpy as np
from safetensors.torch import save_file
from rich.console import Console
from rich.table import Table

from config import GemmaArchitecture, PruningConfig
from model_loader import LayerWeights
from tda_analyzer import TopologicalBridge, TopologyResult, WeightGraphNode
from spectral_verifier import VerificationReport, CircuitVerification
from utils import (
    logger,
    format_param_count,
    create_progress,
    Timer,
)

console = Console()


@dataclass
class LayerMaskInfo:
    """Mask statistics for a single layer."""
    layer_idx: int
    attention_type: str
    total_params: int
    kept_params: int
    pruned_params: int
    sparsity: float  # Fraction pruned


@dataclass
class CompressionReport:
    """Statistics about the pruning operation."""
    original_total_params: int
    kept_total_params: int
    pruned_total_params: int
    compression_ratio: float        # kept / original
    overall_sparsity: float         # pruned / original
    per_layer: List[LayerMaskInfo]
    original_size_mb: float
    pruned_size_mb: float


@dataclass
class PrunedArtifact:
    """The final exported pruned model artifact."""
    output_path: str
    format: str                     # "safetensors" or "onnx"
    file_size_mb: float
    compression_report: CompressionReport
    verified_bridge_ids: List[int]
    num_verified_circuits: int


def _compute_layer_importance_mask(
    layer_weights: LayerWeights,
    bridges: List[TopologicalBridge],
    node_metadata: List[WeightGraphNode],
    verified_bridge_ids: set,
    arch: GemmaArchitecture,
    config: PruningConfig,
    use_magnitude_fallback: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    Generate binary masks for all weight matrices in a single layer.
    A parameter is kept (mask=1) if it lies on a verified topological bridge.
    """
    layer_idx = layer_weights.layer_idx
    node_by_id = {n.node_id: n for n in node_metadata}

    # Check if any verified bridge passes through this layer
    layer_in_verified_bridge = False
    for bridge in bridges:
        if bridge.bridge_id not in verified_bridge_ids:
            continue
        for nid in bridge.path_nodes:
            node = node_by_id.get(nid)
            if node and node.layer_idx == layer_idx:
                layer_in_verified_bridge = True
                break
        if layer_in_verified_bridge:
            break

    masks = {}
    matrix_map = {
        "q_proj": layer_weights.q_proj,
        "k_proj": layer_weights.k_proj,
        "v_proj": layer_weights.v_proj,
        "o_proj": layer_weights.o_proj,
        "gate_proj": layer_weights.gate_proj,
        "up_proj": layer_weights.up_proj,
        "down_proj": layer_weights.down_proj,
        "input_layernorm": layer_weights.input_layernorm,
        "post_attention_layernorm": layer_weights.post_attention_layernorm,
        "pre_feedforward_layernorm": layer_weights.pre_feedforward_layernorm,
        "post_feedforward_layernorm": layer_weights.post_feedforward_layernorm,
    }

    # Helper: create a magnitude-based floor mask (keeps top fraction of weights)
    def _magnitude_floor_mask(tensor: torch.Tensor, keep_fraction: float) -> torch.Tensor:
        """Keep at least the top `keep_fraction` of weights by magnitude."""
        abs_w = tensor.abs().float().flatten()
        threshold = torch.quantile(abs_w, 1.0 - keep_fraction)
        return (tensor.abs() >= threshold).float()

    # Minimum fraction of weights to keep in any layer to prevent signal collapse.
    # Without this, RMS norm inflates near-zero outputs and destroys the residual stream.
    # For a 1B model, anything below 50% per-layer causes coherence collapse.
    MIN_KEEP_FRACTION = 0.70

    if layer_in_verified_bridge:
        if config.structured_pruning:
            # Structured pruning: identify which heads/neurons are active in the bridge
            active_heads = set()
            active_neurons = set()

            for bridge in bridges:
                if bridge.bridge_id not in verified_bridge_ids:
                    continue
                for nid in bridge.path_nodes:
                    node = node_by_id.get(nid)
                    if node and node.layer_idx == layer_idx:
                        if node.component_type == "attention_head":
                            active_heads.add(node.component_idx)
                        elif node.component_type == "mlp_neuron":
                            active_neurons.add(node.component_idx)

            for mat_name, tensor in matrix_map.items():
                if "layernorm" in mat_name:
                    masks[mat_name] = torch.ones_like(tensor)
                    continue

                # Start with the structured bridge mask
                struct_mask = torch.zeros_like(tensor)

                if mat_name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                    head_dim = arch.head_dim
                    if not active_heads:
                        struct_mask = torch.ones_like(tensor)
                    else:
                        for head_idx in active_heads:
                            if mat_name == "o_proj":
                                start = head_idx * head_dim
                                end = min(start + head_dim, tensor.shape[1])
                                struct_mask[:, start:end] = 1.0
                            else:
                                num_kv = arch.num_key_value_heads
                                if mat_name in ("k_proj", "v_proj"):
                                    kv_idx = head_idx % num_kv
                                    start = kv_idx * head_dim
                                    end = min(start + head_dim, tensor.shape[0])
                                else:
                                    start = head_idx * head_dim
                                    end = min(start + head_dim, tensor.shape[0])
                                struct_mask[start:end, :] = 1.0

                elif mat_name in ("gate_proj", "up_proj"):
                    if not active_neurons:
                        struct_mask = torch.ones_like(tensor)
                    else:
                        for neuron_idx in active_neurons:
                            window = max(64, arch.intermediate_size // 4)
                            start = max(0, neuron_idx * (arch.intermediate_size // len(active_neurons)) - window // 2)
                            end = min(tensor.shape[0], start + window)
                            struct_mask[start:end, :] = 1.0

                elif mat_name == "down_proj":
                    if not active_neurons:
                        struct_mask = torch.ones_like(tensor)
                    else:
                        for neuron_idx in active_neurons:
                            window = max(64, arch.intermediate_size // 4)
                            start = max(0, neuron_idx * (arch.intermediate_size // len(active_neurons)) - window // 2)
                            end = min(tensor.shape[1], start + window)
                            struct_mask[:, start:end] = 1.0

                # Union with magnitude floor so no matrix is ever fully zeroed
                mag_floor = _magnitude_floor_mask(tensor, MIN_KEEP_FRACTION)
                masks[mat_name] = torch.clamp(struct_mask + mag_floor, max=1.0)
        else:
            # Unstructured pruning: magnitude-based within the layer
            for mat_name, tensor in matrix_map.items():
                if "layernorm" in mat_name:
                    masks[mat_name] = torch.ones_like(tensor)
                else:
                    keep_frac = max(MIN_KEEP_FRACTION, config.importance_percentile / 100.0)
                    masks[mat_name] = _magnitude_floor_mask(tensor, keep_frac)
    else:
        # Layer not in any verified bridge — keep top weights by magnitude
        # Use gentler pruning than bridge layers to avoid destroying signal propagation
        NON_BRIDGE_KEEP = 0.50
        for mat_name, tensor in matrix_map.items():
            if "layernorm" in mat_name:
                masks[mat_name] = torch.ones_like(tensor)
            else:
                keep_frac = max(NON_BRIDGE_KEEP, config.importance_percentile / 100.0) if use_magnitude_fallback else NON_BRIDGE_KEEP
                masks[mat_name] = _magnitude_floor_mask(tensor, keep_frac)

    return masks


def generate_binary_mask(
    all_layer_weights: List[LayerWeights],
    topology_result: TopologyResult,
    verification_report: VerificationReport,
    arch: GemmaArchitecture,
    config: PruningConfig,
    embedding_matrix: torch.Tensor,
) -> Tuple[Dict[str, torch.Tensor], CompressionReport]:
    """
    Generate binary masks for the entire model based on verified topological bridges.

    Returns:
        Tuple of (mask_dict mapping weight names → binary tensors, compression stats)
    """
    # Identify which bridges to use for masking
    verified_ids = set()
    for cv in verification_report.circuit_results:
        if cv.is_stable:
            verified_ids.add(cv.bridge_id)

    if not verified_ids and verification_report.circuit_results:
        logger.warning("No circuits passed strict verification. Using top bridges by score.")
        sorted_results = sorted(
            verification_report.circuit_results,
            key=lambda x: x.verification_score,
            reverse=True,
        )
        for cv in sorted_results[:5]:
            verified_ids.add(cv.bridge_id)

    if not verified_ids and topology_result.bridges:
        logger.warning("No verification results available. Using ALL topological bridges.")
        for bridge in topology_result.bridges:
            verified_ids.add(bridge.bridge_id)

    use_magnitude_fallback = not verified_ids
    if use_magnitude_fallback:
        logger.warning("No topological bridges found. Falling back to magnitude-based pruning.")

    logger.info(f"Generating masks for {len(verified_ids)} bridge(s) "
                f"(magnitude fallback: {use_magnitude_fallback})")

    full_mask = {}
    per_layer_info = []

    # Always keep the embedding matrix
    embed_mask = torch.ones_like(embedding_matrix)
    full_mask["model.embed_tokens.weight"] = embed_mask

    with create_progress() as progress:
        task = progress.add_task("Generating pruning masks", total=len(all_layer_weights))

        for layer_weights in all_layer_weights:
            layer_masks = _compute_layer_importance_mask(
                layer_weights=layer_weights,
                bridges=topology_result.bridges,
                node_metadata=topology_result.node_metadata,
                verified_bridge_ids=verified_ids,
                arch=arch,
                config=config,
                use_magnitude_fallback=use_magnitude_fallback,
            )

            layer_idx = layer_weights.layer_idx
            prefix = f"model.layers.{layer_idx}"
            attn_prefix = f"{prefix}.self_attn"
            mlp_prefix = f"{prefix}.mlp"

            name_mapping = {
                "q_proj": f"{attn_prefix}.q_proj.weight",
                "k_proj": f"{attn_prefix}.k_proj.weight",
                "v_proj": f"{attn_prefix}.v_proj.weight",
                "o_proj": f"{attn_prefix}.o_proj.weight",
                "gate_proj": f"{mlp_prefix}.gate_proj.weight",
                "up_proj": f"{mlp_prefix}.up_proj.weight",
                "down_proj": f"{mlp_prefix}.down_proj.weight",
                "input_layernorm": f"{prefix}.input_layernorm.weight",
                "post_attention_layernorm": f"{prefix}.post_attention_layernorm.weight",
                "pre_feedforward_layernorm": f"{prefix}.pre_feedforward_layernorm.weight",
                "post_feedforward_layernorm": f"{prefix}.post_feedforward_layernorm.weight",
            }

            total_params = 0
            kept_params = 0
            for short_name, full_name in name_mapping.items():
                if short_name in layer_masks:
                    mask = layer_masks[short_name]
                    full_mask[full_name] = mask
                    total_params += mask.numel()
                    kept_params += mask.sum().int().item()

            pruned = total_params - kept_params
            sparsity = pruned / total_params if total_params > 0 else 0.0

            per_layer_info.append(LayerMaskInfo(
                layer_idx=layer_idx,
                attention_type=layer_weights.attention_type,
                total_params=total_params,
                kept_params=kept_params,
                pruned_params=pruned,
                sparsity=sparsity,
            ))

            progress.advance(task)

    # Final norm
    full_mask["model.norm.weight"] = torch.ones(arch.hidden_size, device=embedding_matrix.device)

    # Compute overall stats
    original_total = sum(li.total_params for li in per_layer_info) + embedding_matrix.numel()
    kept_total = sum(li.kept_params for li in per_layer_info) + embedding_matrix.numel()
    pruned_total = original_total - kept_total

    bytes_per_param = 2  # bfloat16
    original_size = original_total * bytes_per_param / (1024 ** 2)
    pruned_size = kept_total * bytes_per_param / (1024 ** 2)

    report = CompressionReport(
        original_total_params=original_total,
        kept_total_params=kept_total,
        pruned_total_params=pruned_total,
        compression_ratio=kept_total / original_total if original_total > 0 else 0.0,
        overall_sparsity=pruned_total / original_total if original_total > 0 else 0.0,
        per_layer=per_layer_info,
        original_size_mb=original_size,
        pruned_size_mb=pruned_size,
    )

    return full_mask, report


def apply_mask(
    raw_weights: Dict[str, torch.Tensor],
    mask: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """
    Apply the binary mask to the model weights via element-wise multiplication.
    Weights not in the mask are kept as-is (e.g., final norm).
    """
    pruned_weights = {}

    for name, tensor in raw_weights.items():
        if name in mask:
            pruned_weights[name] = tensor * mask[name].to(dtype=tensor.dtype)
        else:
            pruned_weights[name] = tensor

    return pruned_weights


def export_pruned_model(
    pruned_weights: Dict[str, torch.Tensor],
    output_dir: str,
    compression_report: CompressionReport,
    verified_bridge_ids: List[int],
    config: PruningConfig,
) -> PrunedArtifact:
    """
    Export the pruned model as a safetensors file.
    """
    os.makedirs(output_dir, exist_ok=True)

    artifacts = []

    if config.export_safetensors:
        output_path = os.path.join(output_dir, "zitep_pruned_model.safetensors")

        # Move to CPU for saving
        cpu_weights = {}
        for name, tensor in pruned_weights.items():
            cpu_weights[name] = tensor.cpu().contiguous()

        save_file(cpu_weights, output_path)

        file_size = os.path.getsize(output_path) / (1024 ** 2)

        artifact = PrunedArtifact(
            output_path=output_path,
            format="safetensors",
            file_size_mb=file_size,
            compression_report=compression_report,
            verified_bridge_ids=verified_bridge_ids,
            num_verified_circuits=len(verified_bridge_ids),
        )
        artifacts.append(artifact)

        logger.info(f"Pruned model saved: {output_path} ({file_size:.1f} MB)")

    # Save compression report
    report_path = os.path.join(output_dir, "compression_report.json")
    report_data = {
        "original_total_params": compression_report.original_total_params,
        "kept_total_params": compression_report.kept_total_params,
        "pruned_total_params": compression_report.pruned_total_params,
        "compression_ratio": round(compression_report.compression_ratio, 6),
        "overall_sparsity": round(compression_report.overall_sparsity, 6),
        "original_size_mb": round(compression_report.original_size_mb, 2),
        "pruned_size_mb": round(compression_report.pruned_size_mb, 2),
        "verified_bridge_ids": verified_bridge_ids,
        "per_layer": [
            {
                "layer_idx": li.layer_idx,
                "attention_type": li.attention_type,
                "total_params": li.total_params,
                "kept_params": li.kept_params,
                "sparsity": round(li.sparsity, 4),
            }
            for li in compression_report.per_layer
        ],
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    logger.info(f"Compression report saved: {report_path}")

    return artifacts[0] if artifacts else None


def print_compression_summary(report: CompressionReport) -> None:
    """Print a rich summary of the compression results."""
    console.print("\n")
    table = Table(title="Compression Report — Per-Layer Breakdown", show_lines=True)
    table.add_column("Layer", style="cyan", justify="center", width=6)
    table.add_column("Type", style="green", width=8)
    table.add_column("Total", style="white", justify="right", width=12)
    table.add_column("Kept", style="yellow", justify="right", width=12)
    table.add_column("Sparsity", style="red", justify="center", width=10)

    for li in report.per_layer:
        table.add_row(
            str(li.layer_idx),
            li.attention_type,
            format_param_count(li.total_params),
            format_param_count(li.kept_params),
            f"{li.sparsity * 100:.1f}%",
        )

    console.print(table)

    # Overall summary
    summary_table = Table(title="Overall Compression", show_lines=True)
    summary_table.add_column("Metric", style="cyan", width=25)
    summary_table.add_column("Value", style="green", width=25)

    summary_table.add_row("Original Parameters", format_param_count(report.original_total_params))
    summary_table.add_row("Kept Parameters", format_param_count(report.kept_total_params))
    summary_table.add_row("Pruned Parameters", format_param_count(report.pruned_total_params))
    summary_table.add_row("Compression Ratio", f"{report.compression_ratio:.4f}")
    summary_table.add_row("Overall Sparsity", f"{report.overall_sparsity * 100:.1f}%")
    summary_table.add_row("Original Size", f"{report.original_size_mb:.1f} MB")
    summary_table.add_row("Effective Size", f"{report.pruned_size_mb:.1f} MB")

    console.print(summary_table)
