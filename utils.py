"""
ZITEP Utilities — Shared helpers for GPU memory management, vocabulary mapping,
similarity computation, and logging used across all pipeline modules.
"""

import os
import sys
import time
import logging
from typing import Dict, List, Optional, Tuple

import torch
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn

console = Console()


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Configure structured logging for the pipeline."""
    logger = logging.getLogger("zitep")
    logger.setLevel(getattr(logging, log_level.upper()))

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(getattr(logging, log_level.upper()))
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] %(name)s — %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


logger = setup_logging()


def get_gpu_memory_info(device: torch.device) -> Dict[str, float]:
    """Return current GPU memory usage in GB."""
    if device.type != "cuda":
        return {"allocated_gb": 0.0, "reserved_gb": 0.0, "total_gb": 0.0, "free_gb": 0.0}

    allocated = torch.cuda.memory_allocated(device) / (1024 ** 3)
    reserved = torch.cuda.memory_reserved(device) / (1024 ** 3)
    total = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
    free = total - allocated

    return {
        "allocated_gb": round(allocated, 3),
        "reserved_gb": round(reserved, 3),
        "total_gb": round(total, 3),
        "free_gb": round(free, 3),
    }


def log_gpu_memory(device: torch.device, label: str = "") -> None:
    """Log current GPU memory to console."""
    info = get_gpu_memory_info(device)
    if device.type == "cuda":
        prefix = f"[GPU {label}]" if label else "[GPU]"
        logger.info(
            f"{prefix} Allocated: {info['allocated_gb']:.2f} GB | "
            f"Free: {info['free_gb']:.2f} GB | "
            f"Total: {info['total_gb']:.2f} GB"
        )


def clear_gpu_cache(device: torch.device) -> None:
    """Aggressively free GPU memory."""
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)


class VocabularyMapper:
    """
    Maps between vocabulary token indices and human-readable token strings.
    Loads the tokenizer once and caches the full vocabulary for fast lookups.
    """

    def __init__(self, model_name_or_path: str):
        from transformers import AutoTokenizer

        logger.info(f"Loading tokenizer from: {model_name_or_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.vocab_size = self.tokenizer.vocab_size

        # Build index → token mapping
        self._id_to_token: Dict[int, str] = {}
        for token, idx in self.tokenizer.get_vocab().items():
            self._id_to_token[idx] = token

        logger.info(f"Tokenizer loaded: {len(self._id_to_token)} tokens in vocabulary")

    def ids_to_tokens(self, token_ids: List[int]) -> List[str]:
        """Convert a list of token IDs to their string representations."""
        return [self._id_to_token.get(tid, f"<UNK_{tid}>") for tid in token_ids]

    def token_to_id(self, token: str) -> Optional[int]:
        """Convert a token string to its ID."""
        vocab = self.tokenizer.get_vocab()
        return vocab.get(token, None)

    def decode_ids(self, token_ids: List[int]) -> str:
        """Decode a sequence of token IDs back to text."""
        return self.tokenizer.decode(token_ids, skip_special_tokens=False)


def batched_cosine_similarity(
    query_vectors: torch.Tensor,
    reference_matrix: torch.Tensor,
    batch_size: int = 512,
    top_k: int = 10,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute cosine similarity between query vectors and a reference matrix,
    returning top-k scores and indices.

    Args:
        query_vectors:    (N, D) tensor of query vectors
        reference_matrix: (V, D) tensor (e.g., vocabulary embeddings, V=262144)
        batch_size:       number of queries to process at once
        top_k:            number of top matches to return per query

    Returns:
        scores:  (N, top_k) tensor of cosine similarity scores
        indices: (N, top_k) tensor of indices into reference_matrix
    """
    device = query_vectors.device

    # Normalize both sets of vectors
    query_norm = torch.nn.functional.normalize(query_vectors.float(), dim=-1)
    ref_norm = torch.nn.functional.normalize(reference_matrix.float(), dim=-1)

    all_scores = []
    all_indices = []

    for start in range(0, query_vectors.shape[0], batch_size):
        end = min(start + batch_size, query_vectors.shape[0])
        batch = query_norm[start:end]  # (B, D)

        # Cosine similarity: (B, D) @ (D, V) → (B, V)
        sim = torch.mm(batch, ref_norm.t())

        # Top-k per query
        topk_scores, topk_indices = torch.topk(sim, k=min(top_k, sim.shape[1]), dim=-1)
        all_scores.append(topk_scores)
        all_indices.append(topk_indices)

    scores = torch.cat(all_scores, dim=0)
    indices = torch.cat(all_indices, dim=0)

    return scores, indices


def compute_spectral_norm(matrix: torch.Tensor) -> float:
    """
    Compute the spectral norm (largest singular value) of a matrix.
    Uses torch.linalg.svdvals for efficiency (no full decomposition).
    """
    with torch.no_grad():
        singular_values = torch.linalg.svdvals(matrix.float())
        return singular_values[0].item()


def ensure_directory(path: str) -> str:
    """Create directory if it doesn't exist, return the path."""
    os.makedirs(path, exist_ok=True)
    return path


def create_progress() -> Progress:
    """Create a rich progress bar for pipeline stages."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=40),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    )


def format_param_count(count: int) -> str:
    """Format parameter count in human-readable form (e.g., 1.07B, 45.3M)."""
    if count >= 1e9:
        return f"{count / 1e9:.2f}B"
    elif count >= 1e6:
        return f"{count / 1e6:.2f}M"
    elif count >= 1e3:
        return f"{count / 1e3:.1f}K"
    return str(count)


def print_architecture_summary(arch) -> None:
    """Print a rich table summarizing the Gemma 3 architecture."""
    table = Table(title="Gemma 3 1B Instruct — Architecture Map", show_lines=True)
    table.add_column("Property", style="cyan", width=30)
    table.add_column("Value", style="green", width=40)

    table.add_row("Model", arch.model_name)
    table.add_row("Vocabulary Size", f"{arch.vocab_size:,}")
    table.add_row("Layers", str(arch.num_layers))
    table.add_row("Hidden Dimension", str(arch.hidden_size))
    table.add_row("MLP Intermediate", str(arch.intermediate_size))
    table.add_row("Attention Heads (Q)", str(arch.num_attention_heads))
    table.add_row("KV Heads (GQA)", str(arch.num_key_value_heads))
    table.add_row("Head Dimension", str(arch.head_dim))
    table.add_row("Context Window", f"{arch.max_position_embeddings:,} tokens")
    table.add_row("Sliding Window", f"{arch.sliding_window_size:,} tokens")
    table.add_row("RoPE Base (Local)", f"{arch.rope_base_local:,.0f}")
    table.add_row("RoPE Base (Global)", f"{arch.rope_base_global:,.0f}")
    table.add_row("Activation", arch.hidden_activation)
    table.add_row("Precision", arch.dtype)
    table.add_row("Global Layers", str(arch.get_global_layer_indices()))
    table.add_row("Local Layers", str(arch.get_local_layer_indices()))

    console.print(table)


def print_phase_header(phase_num: int, title: str) -> None:
    """Print a styled phase header."""
    console.print(f"\n{'═' * 70}", style="bold yellow")
    console.print(f"  PHASE {phase_num}: {title}", style="bold yellow")
    console.print(f"{'═' * 70}\n", style="bold yellow")


class Timer:
    """Simple context-manager timer for benchmarking pipeline phases."""

    def __init__(self, label: str):
        self.label = label
        self.start_time = 0.0
        self.elapsed = 0.0

    def __enter__(self):
        self.start_time = time.perf_counter()
        logger.info(f"⏱ Starting: {self.label}")
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self.start_time
        logger.info(f"✓ Completed: {self.label} in {self.elapsed:.2f}s")
