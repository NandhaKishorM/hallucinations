"""
ZITEP Model Loader — Load Gemma 3 1B safetensors as raw static weight tensors.

This module loads the model's .safetensors weight files into memory as plain
torch.Tensor dictionaries. NO AutoModelForCausalLM is instantiated, NO forward
pass capability is created. The weights are treated as pure mathematical objects.
"""

import os
import glob
from typing import Dict, List, NamedTuple, Optional

import torch
from safetensors import safe_open
from rich.console import Console

from config import GemmaArchitecture, PipelineConfig
from utils import (
    logger,
    log_gpu_memory,
    clear_gpu_cache,
    format_param_count,
    create_progress,
    Timer,
)

console = Console()


class LayerWeights(NamedTuple):
    """All weight matrices for a single transformer layer."""
    layer_idx: int
    attention_type: str  # "local" or "global"

    # Self-attention projections — shape: (out_features, in_features)
    q_proj: torch.Tensor       # (num_heads * head_dim, hidden_size)
    k_proj: torch.Tensor       # (num_kv_heads * head_dim, hidden_size)
    v_proj: torch.Tensor       # (num_kv_heads * head_dim, hidden_size)
    o_proj: torch.Tensor       # (hidden_size, num_heads * head_dim)

    # GeGLU MLP — the gate and up projections define the gating mechanism
    gate_proj: torch.Tensor    # (intermediate_size, hidden_size)
    up_proj: torch.Tensor      # (intermediate_size, hidden_size)
    down_proj: torch.Tensor    # (hidden_size, intermediate_size)

    # Layer norms
    input_layernorm: torch.Tensor        # (hidden_size,)
    post_attention_layernorm: torch.Tensor  # (hidden_size,)
    pre_feedforward_layernorm: torch.Tensor  # (hidden_size,)
    post_feedforward_layernorm: torch.Tensor  # (hidden_size,)


def _find_safetensor_files(model_path: str) -> List[str]:
    """Find all .safetensors files in the model directory."""
    if os.path.isfile(model_path) and model_path.endswith(".safetensors"):
        return [model_path]

    pattern = os.path.join(model_path, "*.safetensors")
    files = sorted(glob.glob(pattern))

    if not files:
        raise FileNotFoundError(
            f"No .safetensors files found in {model_path}. "
            f"Download the model first: huggingface-cli download google/gemma-3-1b-it"
        )

    return files


def load_raw_weights(
    model_path: str,
    device: torch.device,
    dtype: Optional[torch.dtype] = None,
) -> Dict[str, torch.Tensor]:
    """
    Load all safetensors weight files into a flat dictionary of tensors.
    Memory-maps the files and transfers to the target device.

    Args:
        model_path: Path to directory containing .safetensors files, or a single file
        device:     Target device (cuda/cpu)
        dtype:      Optional dtype override (default: use file's native dtype)

    Returns:
        Dictionary mapping weight names → tensors on the target device
    """
    files = _find_safetensor_files(model_path)
    logger.info(f"Found {len(files)} safetensor file(s) in {model_path}")

    all_weights: Dict[str, torch.Tensor] = {}
    total_params = 0

    with create_progress() as progress:
        task = progress.add_task("Loading safetensors", total=len(files))

        for filepath in files:
            filename = os.path.basename(filepath)
            logger.info(f"Loading: {filename}")

            with safe_open(filepath, framework="pt", device="cpu") as f:
                for key in f.keys():
                    tensor = f.get_tensor(key)

                    if dtype is not None:
                        tensor = tensor.to(dtype=dtype)

                    # Transfer to GPU
                    tensor = tensor.to(device=device)
                    all_weights[key] = tensor
                    total_params += tensor.numel()

            progress.advance(task)

    logger.info(
        f"Loaded {len(all_weights)} tensors | "
        f"Total parameters: {format_param_count(total_params)} ({total_params:,})"
    )
    log_gpu_memory(device, "after weight loading")

    return all_weights


def _get_weight(
    raw_weights: Dict[str, torch.Tensor],
    key: str,
) -> torch.Tensor:
    """Safely retrieve a weight tensor by key, raising clear errors."""
    if key not in raw_weights:
        available = [k for k in raw_weights.keys() if "layers" not in k or "layers.0." in k]
        raise KeyError(
            f"Weight key '{key}' not found. "
            f"Sample available keys: {available[:10]}"
        )
    return raw_weights[key]


def get_embedding_matrix(raw_weights: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Extract the token embedding matrix.

    In Gemma 3, the embedding is shared between input and output (tied weights).
    Shape: (vocab_size, hidden_size) = (262144, 1152)
    """
    # Gemma 3 uses "model.embed_tokens.weight" for the shared embedding
    embed_key = "model.embed_tokens.weight"
    embedding = _get_weight(raw_weights, embed_key)
    logger.info(f"Embedding matrix shape: {embedding.shape}")
    return embedding


def get_final_norm(raw_weights: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Extract the final RMS normalization weight."""
    return _get_weight(raw_weights, "model.norm.weight")


def extract_layer_weights(
    raw_weights: Dict[str, torch.Tensor],
    layer_idx: int,
    arch: GemmaArchitecture,
) -> LayerWeights:
    """
    Extract all weight matrices for a specific transformer layer.

    Args:
        raw_weights: Full weight dictionary from load_raw_weights
        layer_idx:   Layer index (0 to 25 for Gemma 3 1B)
        arch:        Architecture constants

    Returns:
        LayerWeights named tuple with all projection and MLP matrices
    """
    prefix = f"model.layers.{layer_idx}"
    attn_prefix = f"{prefix}.self_attn"
    mlp_prefix = f"{prefix}.mlp"

    attention_type = "global" if arch.is_global_layer(layer_idx) else "local"

    # Handle optional layernorm keys — Gemma 3 may have pre/post feedforward norms
    def safe_get(key: str, fallback_shape: int = arch.hidden_size) -> torch.Tensor:
        if key in raw_weights:
            return raw_weights[key]
        return torch.ones(fallback_shape, device=next(iter(raw_weights.values())).device)

    return LayerWeights(
        layer_idx=layer_idx,
        attention_type=attention_type,
        q_proj=_get_weight(raw_weights, f"{attn_prefix}.q_proj.weight"),
        k_proj=_get_weight(raw_weights, f"{attn_prefix}.k_proj.weight"),
        v_proj=_get_weight(raw_weights, f"{attn_prefix}.v_proj.weight"),
        o_proj=_get_weight(raw_weights, f"{attn_prefix}.o_proj.weight"),
        gate_proj=_get_weight(raw_weights, f"{mlp_prefix}.gate_proj.weight"),
        up_proj=_get_weight(raw_weights, f"{mlp_prefix}.up_proj.weight"),
        down_proj=_get_weight(raw_weights, f"{mlp_prefix}.down_proj.weight"),
        input_layernorm=_get_weight(raw_weights, f"{prefix}.input_layernorm.weight"),
        post_attention_layernorm=_get_weight(raw_weights, f"{prefix}.post_attention_layernorm.weight"),
        pre_feedforward_layernorm=safe_get(f"{prefix}.pre_feedforward_layernorm.weight"),
        post_feedforward_layernorm=safe_get(f"{prefix}.post_feedforward_layernorm.weight"),
    )


def extract_all_layers(
    raw_weights: Dict[str, torch.Tensor],
    arch: GemmaArchitecture,
) -> List[LayerWeights]:
    """
    Extract weight matrices for all 26 transformer layers.

    Returns:
        List of LayerWeights, one per layer, ordered by layer index.
    """
    layers = []
    with create_progress() as progress:
        task = progress.add_task("Extracting layer weights", total=arch.num_layers)

        for i in range(arch.num_layers):
            layer = extract_layer_weights(raw_weights, i, arch)
            layers.append(layer)
            progress.advance(task)

    global_count = sum(1 for lw in layers if lw.attention_type == "global")
    local_count = sum(1 for lw in layers if lw.attention_type == "local")
    logger.info(f"Extracted {len(layers)} layers: {local_count} local, {global_count} global")

    return layers


def compute_weight_statistics(raw_weights: Dict[str, torch.Tensor]) -> Dict[str, dict]:
    """
    Compute basic statistics for all weight tensors (mean, std, min, max, norm).
    Useful for initial exploration of the weight landscape.
    """
    stats = {}
    for name, tensor in raw_weights.items():
        t = tensor.float()
        stats[name] = {
            "shape": list(tensor.shape),
            "numel": tensor.numel(),
            "mean": t.mean().item(),
            "std": t.std().item(),
            "min": t.min().item(),
            "max": t.max().item(),
            "frobenius_norm": t.norm().item(),
        }
    return stats


def print_weight_summary(raw_weights: Dict[str, torch.Tensor]) -> None:
    """Print a categorized summary of all loaded weights."""
    from rich.table import Table

    categories = {
        "Embedding": [],
        "Attention": [],
        "MLP": [],
        "LayerNorm": [],
        "Other": [],
    }

    total_params = 0
    for name, tensor in raw_weights.items():
        numel = tensor.numel()
        total_params += numel
        entry = (name, tensor.shape, numel)

        if "embed" in name:
            categories["Embedding"].append(entry)
        elif "self_attn" in name:
            categories["Attention"].append(entry)
        elif "mlp" in name:
            categories["MLP"].append(entry)
        elif "norm" in name or "layernorm" in name:
            categories["LayerNorm"].append(entry)
        else:
            categories["Other"].append(entry)

    table = Table(title="Weight Distribution by Category", show_lines=True)
    table.add_column("Category", style="cyan")
    table.add_column("Count", style="green", justify="right")
    table.add_column("Parameters", style="yellow", justify="right")
    table.add_column("% of Total", style="magenta", justify="right")

    for cat_name, entries in categories.items():
        cat_params = sum(e[2] for e in entries)
        pct = (cat_params / total_params * 100) if total_params > 0 else 0
        table.add_row(
            cat_name,
            str(len(entries)),
            format_param_count(cat_params),
            f"{pct:.1f}%",
        )

    table.add_row(
        "[bold]TOTAL[/bold]",
        str(len(raw_weights)),
        format_param_count(total_params),
        "100.0%",
        style="bold",
    )

    console.print(table)
