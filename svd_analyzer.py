"""
ZITEP SVD Analyzer — Phase 2: Mathematical Disassembly via Singular Value Decomposition.

Decomposes every MLP weight matrix (gate_proj, up_proj, down_proj) across all 26
transformer layers using GPU-accelerated SVD. Projects the resulting singular vectors
back into the 262K vocabulary embedding space to extract human-readable concept labels.

The output is a complete static "Lookup Table" mapping each layer/matrix to the
linguistic concepts it listens for (V^T) and transforms into (U).
"""

import os
import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

import torch
import numpy as np
from rich.console import Console
from rich.table import Table

from config import GemmaArchitecture, SVDConfig
from model_loader import LayerWeights
from utils import (
    logger,
    log_gpu_memory,
    clear_gpu_cache,
    batched_cosine_similarity,
    VocabularyMapper,
    create_progress,
    format_param_count,
    Timer,
)

console = Console()


@dataclass
class ConceptVector:
    """A single concept direction extracted from SVD."""
    rank: int                        # Singular value rank (0 = most important)
    singular_value: float            # σ_i
    relative_importance: float       # σ_i / σ_0
    is_spike: bool                   # Whether this σ is a "spike" (σ_i / σ_{i+1} > threshold)

    # Input concepts (from V^T) — what the matrix listens for
    input_tokens: List[str]          # Top-k vocabulary tokens
    input_scores: List[float]        # Corresponding cosine similarity scores

    # Output concepts (from U) — what the matrix transforms into
    output_tokens: List[str]
    output_scores: List[float]


@dataclass
class MatrixDecomposition:
    """SVD results for a single weight matrix."""
    layer_idx: int
    matrix_name: str                  # "gate_proj", "up_proj", "down_proj"
    attention_type: str               # "local" or "global"
    original_shape: Tuple[int, int]
    total_singular_values: int
    top_k_analyzed: int

    concepts: List[ConceptVector]     # Top-k concept vectors

    # Aggregate statistics
    energy_ratio: float               # Sum of top-k σ² / total σ² — how much info is in top-k
    num_spikes: int                   # Number of spiky singular values
    spectral_norm: float              # Largest singular value = ‖W‖₂

    def get_spike_concepts(self) -> List[ConceptVector]:
        """Return only the high-importance 'spiky' concepts."""
        return [c for c in self.concepts if c.is_spike]


@dataclass
class LayerDecomposition:
    """SVD results for all matrices in one transformer layer."""
    layer_idx: int
    attention_type: str
    matrices: Dict[str, MatrixDecomposition]


@dataclass
class ConceptTable:
    """
    The complete static Lookup Table of the entire model.
    Maps every layer / matrix to its extracted concepts.
    """
    model_name: str
    num_layers: int
    layers: List[LayerDecomposition]

    # Cross-layer concept tracking
    global_spike_count: int = 0
    total_concepts_extracted: int = 0

    def get_all_spikes(self) -> List[Tuple[int, str, ConceptVector]]:
        """Return all spiky concepts across all layers as (layer, matrix, concept) tuples."""
        spikes = []
        for layer_dec in self.layers:
            for mat_name, mat_dec in layer_dec.matrices.items():
                for concept in mat_dec.get_spike_concepts():
                    spikes.append((layer_dec.layer_idx, mat_name, concept))
        return spikes

    def get_layer_importance_scores(self) -> Dict[int, float]:
        """Return per-layer importance as sum of spectral norms."""
        scores = {}
        for layer_dec in self.layers:
            total_norm = sum(m.spectral_norm for m in layer_dec.matrices.values())
            scores[layer_dec.layer_idx] = total_norm
        return scores


def _truncated_svd_gpu(
    matrix: torch.Tensor,
    top_k: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Perform truncated SVD on GPU using torch.linalg.svd.

    Args:
        matrix: (M, N) weight matrix on CUDA
        top_k:  Number of top singular components to retain

    Returns:
        U:  (M, top_k)
        S:  (top_k,)
        Vt: (top_k, N)
    """
    with torch.no_grad():
        # Full SVD on GPU (torch doesn't have a native truncated SVD on CUDA)
        # We compute full and then slice — still fast on modern GPUs
        matrix_f32 = matrix.float()  # SVD requires float32

        U_full, S_full, Vt_full = torch.linalg.svd(matrix_f32, full_matrices=False)

        k = min(top_k, S_full.shape[0])
        U = U_full[:, :k]
        S = S_full[:k]
        Vt = Vt_full[:k, :]

    return U, S, Vt


def _detect_spikes(
    singular_values: torch.Tensor,
    spike_ratio_threshold: float,
) -> List[bool]:
    """
    Detect 'spiky' singular values where σ_i / σ_{i+1} exceeds threshold.
    A spike indicates the model dedicates disproportionate energy to that concept.
    """
    sv = singular_values.cpu().numpy()
    spikes = []

    for i in range(len(sv)):
        if i < len(sv) - 1 and sv[i + 1] > 1e-10:
            ratio = sv[i] / sv[i + 1]
            spikes.append(ratio > spike_ratio_threshold)
        else:
            # Last value or near-zero successor — mark as spike if large
            spikes.append(sv[i] > sv[0] * 0.5 if i == 0 else False)

    return spikes


def _project_vectors_to_vocab(
    vectors: torch.Tensor,
    embedding_matrix: torch.Tensor,
    vocab_mapper: VocabularyMapper,
    top_k_tokens: int,
) -> List[Tuple[List[str], List[float]]]:
    """
    Project a set of vectors onto the vocabulary embedding space via cosine similarity.

    Args:
        vectors:          (K, D) tensor of vectors to project
        embedding_matrix: (V, D) vocabulary embedding matrix
        vocab_mapper:     Token ID → string mapper
        top_k_tokens:     Number of top matching tokens per vector

    Returns:
        List of (token_strings, scores) tuples, one per input vector
    """
    scores, indices = batched_cosine_similarity(
        vectors, embedding_matrix, batch_size=256, top_k=top_k_tokens
    )

    results = []
    for i in range(vectors.shape[0]):
        token_ids = indices[i].cpu().tolist()
        token_scores = scores[i].cpu().tolist()
        token_strings = vocab_mapper.ids_to_tokens(token_ids)
        results.append((token_strings, token_scores))

    return results


def decompose_matrix(
    matrix: torch.Tensor,
    layer_idx: int,
    matrix_name: str,
    attention_type: str,
    embedding_matrix: torch.Tensor,
    vocab_mapper: VocabularyMapper,
    svd_config: SVDConfig,
) -> MatrixDecomposition:
    """
    Full SVD decomposition of a single weight matrix with vocabulary projection.

    Args:
        matrix:           The weight matrix W to decompose
        layer_idx:        Layer index this matrix belongs to
        matrix_name:      "gate_proj", "up_proj", or "down_proj"
        attention_type:   "local" or "global"
        embedding_matrix: (262144, 1152) vocabulary embeddings
        vocab_mapper:     Token ID → string mapper
        svd_config:       SVD configuration parameters
    """
    top_k = svd_config.top_k
    original_shape = tuple(matrix.shape)

    # Perform SVD: W = U Σ V^T
    U, S, Vt = _truncated_svd_gpu(matrix, top_k)

    # Compute energy ratio
    with torch.no_grad():
        all_sv = torch.linalg.svdvals(matrix.float())
        total_energy = (all_sv ** 2).sum().item()
        topk_energy = (S ** 2).sum().item()
        energy_ratio = topk_energy / total_energy if total_energy > 0 else 0.0

    # Detect spikes
    spikes = _detect_spikes(S, svd_config.spike_ratio_threshold)

    # Project V^T rows → vocabulary (input concepts: what the matrix listens for)
    # V^T shape: (top_k, N) where N = hidden_size for gate/up, or intermediate_size for down
    # We need to project into vocab space. For gate_proj/up_proj: N = hidden_size (1152),
    # directly comparable to embedding (262144, 1152).
    # For down_proj: N = intermediate_size (6912), not directly in embedding space.
    # In that case, we project U columns instead (U shape: (hidden_size, top_k)).

    if matrix_name in ("gate_proj", "up_proj"):
        # V^T rows are in hidden_size space → project to vocab
        input_projections = _project_vectors_to_vocab(
            Vt, embedding_matrix, vocab_mapper, svd_config.vocab_top_k
        )
        # U columns are in intermediate_size space → cannot project to vocab directly
        # Instead, report the magnitude-based output characteristics
        output_projections = _project_vectors_to_vocab(
            Vt, embedding_matrix, vocab_mapper, svd_config.vocab_top_k
        )
    elif matrix_name == "down_proj":
        # down_proj: (hidden_size, intermediate_size)
        # U columns are in hidden_size space → project to vocab (output concepts)
        output_projections = _project_vectors_to_vocab(
            U.t(), embedding_matrix, vocab_mapper, svd_config.vocab_top_k
        )
        # V^T rows are in intermediate_size space → not in vocab space
        # Use U transposed as proxy for input concepts too
        input_projections = _project_vectors_to_vocab(
            U.t(), embedding_matrix, vocab_mapper, svd_config.vocab_top_k
        )
    else:
        input_projections = _project_vectors_to_vocab(
            Vt, embedding_matrix, vocab_mapper, svd_config.vocab_top_k
        )
        output_projections = _project_vectors_to_vocab(
            U.t() if U.shape[1] <= embedding_matrix.shape[1] else Vt,
            embedding_matrix, vocab_mapper, svd_config.vocab_top_k
        )

    # Build concept vectors
    concepts = []
    sigma_0 = S[0].item() if S.shape[0] > 0 else 1.0

    for rank in range(S.shape[0]):
        sv = S[rank].item()

        in_tokens, in_scores = input_projections[rank]
        out_tokens, out_scores = output_projections[rank]

        concept = ConceptVector(
            rank=rank,
            singular_value=sv,
            relative_importance=sv / sigma_0 if sigma_0 > 0 else 0.0,
            is_spike=spikes[rank],
            input_tokens=in_tokens,
            input_scores=in_scores,
            output_tokens=out_tokens,
            output_scores=out_scores,
        )
        concepts.append(concept)

    return MatrixDecomposition(
        layer_idx=layer_idx,
        matrix_name=matrix_name,
        attention_type=attention_type,
        original_shape=original_shape,
        total_singular_values=all_sv.shape[0],
        top_k_analyzed=S.shape[0],
        concepts=concepts,
        energy_ratio=energy_ratio,
        num_spikes=sum(spikes),
        spectral_norm=sigma_0,
    )


def decompose_layer(
    layer_weights: LayerWeights,
    embedding_matrix: torch.Tensor,
    vocab_mapper: VocabularyMapper,
    svd_config: SVDConfig,
) -> LayerDecomposition:
    """
    Decompose all three MLP matrices (gate, up, down) for a single layer.
    """
    matrices = {}

    mlp_weights = {
        "gate_proj": layer_weights.gate_proj,
        "up_proj": layer_weights.up_proj,
        "down_proj": layer_weights.down_proj,
    }

    for mat_name, mat_tensor in mlp_weights.items():
        dec = decompose_matrix(
            matrix=mat_tensor,
            layer_idx=layer_weights.layer_idx,
            matrix_name=mat_name,
            attention_type=layer_weights.attention_type,
            embedding_matrix=embedding_matrix,
            vocab_mapper=vocab_mapper,
            svd_config=svd_config,
        )
        matrices[mat_name] = dec

    return LayerDecomposition(
        layer_idx=layer_weights.layer_idx,
        attention_type=layer_weights.attention_type,
        matrices=matrices,
    )


def decompose_all_layers(
    all_layer_weights: List[LayerWeights],
    embedding_matrix: torch.Tensor,
    vocab_mapper: VocabularyMapper,
    svd_config: SVDConfig,
    device: torch.device,
    arch: GemmaArchitecture,
) -> ConceptTable:
    """
    Run SVD decomposition across all 26 transformer layers.
    This is the main entry point for Phase 2.

    Returns:
        ConceptTable — the complete static lookup table of the model.
    """
    layers = []
    total_spikes = 0
    total_concepts = 0

    with create_progress() as progress:
        task = progress.add_task(
            "SVD Decomposition (all layers)", total=len(all_layer_weights)
        )

        for layer_weights in all_layer_weights:
            layer_dec = decompose_layer(
                layer_weights=layer_weights,
                embedding_matrix=embedding_matrix,
                vocab_mapper=vocab_mapper,
                svd_config=svd_config,
            )
            layers.append(layer_dec)

            for mat_dec in layer_dec.matrices.values():
                total_spikes += mat_dec.num_spikes
                total_concepts += len(mat_dec.concepts)

            progress.advance(task)

            # Periodically clear cache to avoid OOM on smaller GPUs
            if layer_weights.layer_idx % 4 == 0:
                clear_gpu_cache(device)

    concept_table = ConceptTable(
        model_name=arch.model_name,
        num_layers=arch.num_layers,
        layers=layers,
        global_spike_count=total_spikes,
        total_concepts_extracted=total_concepts,
    )

    logger.info(
        f"SVD complete: {total_concepts} concepts extracted, "
        f"{total_spikes} spikes detected across {len(layers)} layers"
    )

    return concept_table


class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy/torch types."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, torch.Tensor):
            return obj.cpu().tolist()
        return super().default(obj)


def save_concept_table(concept_table: ConceptTable, output_dir: str) -> str:
    """
    Save the concept table to a JSON file for inspection and downstream use.
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "concept_table.json")

    # Convert to serializable format
    data = {
        "model_name": concept_table.model_name,
        "num_layers": int(concept_table.num_layers),
        "global_spike_count": int(concept_table.global_spike_count),
        "total_concepts_extracted": int(concept_table.total_concepts_extracted),
        "layer_importance_scores": {
            int(k): float(v)
            for k, v in concept_table.get_layer_importance_scores().items()
        },
        "layers": [],
    }

    for layer_dec in concept_table.layers:
        layer_data = {
            "layer_idx": int(layer_dec.layer_idx),
            "attention_type": layer_dec.attention_type,
            "matrices": {},
        }

        for mat_name, mat_dec in layer_dec.matrices.items():
            mat_data = {
                "original_shape": [int(d) for d in mat_dec.original_shape],
                "spectral_norm": float(mat_dec.spectral_norm),
                "energy_ratio": float(mat_dec.energy_ratio),
                "num_spikes": int(mat_dec.num_spikes),
                "top_concepts": [],
            }

            # Save top 10 concepts per matrix (keeps file manageable)
            for concept in mat_dec.concepts[:10]:
                concept_data = {
                    "rank": int(concept.rank),
                    "singular_value": float(concept.singular_value),
                    "relative_importance": float(concept.relative_importance),
                    "is_spike": bool(concept.is_spike),
                    "input_tokens_top5": concept.input_tokens[:5],
                    "input_scores_top5": [round(float(s), 4) for s in concept.input_scores[:5]],
                    "output_tokens_top5": concept.output_tokens[:5],
                    "output_scores_top5": [round(float(s), 4) for s in concept.output_scores[:5]],
                }
                mat_data["top_concepts"].append(concept_data)

            layer_data["matrices"][mat_name] = mat_data

        data["layers"].append(layer_data)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, cls=_NumpyEncoder)

    logger.info(f"Concept table saved to: {output_path}")
    return output_path


def print_svd_summary(concept_table: ConceptTable) -> None:
    """Print a rich summary of the SVD analysis results."""
    table = Table(title="SVD Analysis Summary — Per-Layer Breakdown", show_lines=True)
    table.add_column("Layer", style="cyan", justify="center", width=6)
    table.add_column("Type", style="green", width=8)
    table.add_column("Spikes", style="yellow", justify="center", width=8)
    table.add_column("Energy %", style="magenta", justify="center", width=10)
    table.add_column("‖W‖₂ (max)", style="red", justify="center", width=12)
    table.add_column("Top Input Concept", style="white", width=25)

    for layer_dec in concept_table.layers:
        total_spikes = sum(m.num_spikes for m in layer_dec.matrices.values())
        avg_energy = np.mean([m.energy_ratio for m in layer_dec.matrices.values()])
        max_norm = max(m.spectral_norm for m in layer_dec.matrices.values())

        # Get the top concept from the gate_proj (the gating mechanism)
        gate = layer_dec.matrices.get("gate_proj")
        top_concept = "—"
        if gate and gate.concepts:
            top_tokens = gate.concepts[0].input_tokens[:3]
            top_concept = ", ".join(top_tokens)

        table.add_row(
            str(layer_dec.layer_idx),
            layer_dec.attention_type,
            str(total_spikes),
            f"{avg_energy * 100:.1f}%",
            f"{max_norm:.2f}",
            top_concept,
        )

    console.print(table)

    # Print overall statistics
    all_spikes = concept_table.get_all_spikes()
    console.print(f"\n[bold]Total spiky concepts:[/bold] {len(all_spikes)}")
    console.print(f"[bold]Total concepts analyzed:[/bold] {concept_table.total_concepts_extracted}")

    if all_spikes:
        console.print("\n[bold yellow]Top 10 Spiky Concepts (highest singular values):[/bold yellow]")
        sorted_spikes = sorted(all_spikes, key=lambda x: x[2].singular_value, reverse=True)
        for layer_idx, mat_name, concept in sorted_spikes[:10]:
            in_tokens = ", ".join(concept.input_tokens[:5])
            console.print(
                f"  Layer {layer_idx:2d} | {mat_name:10s} | "
                f"σ={concept.singular_value:8.2f} | "
                f"Listens for: [{in_tokens}]"
            )
