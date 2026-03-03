"""
ZITEP Configuration — Gemma 3 1B Instruct Architecture Constants & Pipeline Parameters.

All architectural constants are derived from the official Gemma 3 technical report
and the model's config.json. This module defines the frozen model topology and
all tunable pipeline hyperparameters.
"""

from dataclasses import dataclass, field
from typing import List, Tuple
import torch


@dataclass(frozen=True)
class GemmaArchitecture:
    """Frozen architectural constants for Gemma 3 1B Instruct."""

    model_name: str = "google/gemma-3-1b-it"

    # Token vocabulary
    vocab_size: int = 262144

    # Transformer geometry
    num_layers: int = 26
    hidden_size: int = 1152
    intermediate_size: int = 6912  # GeGLU MLP intermediate dimension

    # Grouped-Query Attention
    num_attention_heads: int = 4       # Query heads
    num_key_value_heads: int = 1       # KV heads (GQA 4:1 ratio)
    head_dim: int = 256                # Per-head dimension

    # Positional encoding
    rope_base_local: float = 10_000.0       # RoPE base for local sliding window layers
    rope_base_global: float = 1_000_000.0   # RoPE base for global attention layers
    max_position_embeddings: int = 32768     # 32K context window
    sliding_window_size: int = 1120          # Local attention window

    # Activation
    hidden_activation: str = "gelu_pytorch_tanh"  # GeGLU

    # Normalization
    rms_norm_eps: float = 1e-6

    # Precision
    dtype: str = "bfloat16"

    # 5:1 interleaved attention pattern
    local_global_pattern: Tuple[str, ...] = (
        "local", "local", "local", "local", "local", "global",  # 0-5
        "local", "local", "local", "local", "local", "global",  # 6-11
        "local", "local", "local", "local", "local", "global",  # 12-17
        "local", "local", "local", "local", "local", "global",  # 18-23
        "local", "local",                                        # 24-25
    )

    def is_global_layer(self, layer_idx: int) -> bool:
        """Check if a layer uses global attention (can see full 32K context)."""
        if layer_idx < 0 or layer_idx >= self.num_layers:
            raise ValueError(f"Layer index {layer_idx} out of range [0, {self.num_layers})")
        return self.local_global_pattern[layer_idx] == "global"

    def get_global_layer_indices(self) -> List[int]:
        """Return indices of all global attention layers."""
        return [i for i in range(self.num_layers) if self.is_global_layer(i)]

    def get_local_layer_indices(self) -> List[int]:
        """Return indices of all local sliding window layers."""
        return [i for i in range(self.num_layers) if not self.is_global_layer(i)]

    def get_torch_dtype(self) -> torch.dtype:
        """Return the torch dtype for weight loading."""
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        return dtype_map[self.dtype]


@dataclass(frozen=True)
class SVDConfig:
    """Parameters for the SVD decomposition phase."""

    # Number of top singular values/vectors to retain per matrix
    top_k: int = 64

    # Minimum singular value ratio (σ_i / σ_0) to consider a concept "significant"
    significance_threshold: float = 0.01

    # Spike detection: a singular value is "spiky" if σ_i / σ_{i+1} exceeds this
    spike_ratio_threshold: float = 3.0

    # Number of top vocabulary tokens to retrieve per concept vector
    vocab_top_k: int = 20

    # Batch size for processing layers on GPU (reduce if OOM)
    batch_size: int = 4

    # Whether to use truncated SVD (faster) or full SVD
    use_truncated: bool = True


@dataclass(frozen=True)
class TDAConfig:
    """Parameters for the Topological Data Analysis phase."""

    # Maximum homology dimension to compute (0 = components, 1 = loops)
    max_homology_dim: int = 1

    # Ripser max edge length (epsilon) — filters noise in persistence diagrams
    max_edge_length: float = 2.0

    # Minimum persistence (death - birth) to consider a topological feature significant
    min_persistence: float = 0.05

    # Number of nearest neighbors for building the weight graph
    n_neighbors: int = 15

    # Downsampling factor for weight vectors (TDA on full 1B params is expensive)
    # We sample 1/downsample_factor of the neurons per layer
    downsample_factor: int = 8

    # Edge weight threshold: only keep edges with weight above this percentile
    edge_weight_percentile: float = 90.0


@dataclass(frozen=True)
class SpectralConfig:
    """Parameters for the spectral verification phase."""

    # Maximum acceptable Lipschitz constant for a verified circuit
    max_lipschitz_bound: float = 100.0

    # Whether to run the Z3 SMT formal verification (expensive, optional)
    enable_smt_verification: bool = True

    # SMT solver timeout in seconds
    smt_timeout_seconds: int = 300

    # Number of random perturbation vectors to test circuit stability empirically
    num_perturbation_tests: int = 1000

    # Maximum relative output deviation under perturbation (empirical check)
    max_relative_deviation: float = 0.1


@dataclass(frozen=True)
class PruningConfig:
    """Parameters for the circuit pruning and export phase."""

    # Minimum pheromone/importance score to keep a parameter (percentile-based)
    importance_percentile: float = 85.0

    # Whether to apply structured pruning (entire heads/neurons) vs unstructured
    structured_pruning: bool = True

    # Output formats
    export_safetensors: bool = True
    export_onnx: bool = False

    # Target compression ratio (informational — actual ratio depends on circuit)
    target_compression_ratio: float = 0.05  # Aim for ~5% of original params


@dataclass
class PipelineConfig:
    """Master configuration for the full ZITEP pipeline."""

    # Model source
    model_path: str = "google/gemma-3-1b-it"  # Path to local safetensors or HF model ID
    output_dir: str = "./zitep_output"

    # GPU device
    device: str = "cuda"
    gpu_memory_fraction: float = 0.9  # Max fraction of GPU memory to use

    # Sub-configs
    arch: GemmaArchitecture = field(default_factory=GemmaArchitecture)
    svd: SVDConfig = field(default_factory=SVDConfig)
    tda: TDAConfig = field(default_factory=TDAConfig)
    spectral: SpectralConfig = field(default_factory=SpectralConfig)
    pruning: PruningConfig = field(default_factory=PruningConfig)

    # Logging
    log_level: str = "INFO"
    save_intermediate: bool = True  # Save intermediate results (concept tables, diagrams)

    def get_device(self) -> torch.device:
        """Return the torch device, falling back to CPU if CUDA unavailable."""
        if self.device == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        elif self.device == "cuda":
            print("[WARNING] CUDA requested but not available. Falling back to CPU.")
            return torch.device("cpu")
        return torch.device(self.device)

    def validate(self) -> None:
        """Validate configuration consistency."""
        if not self.model_path:
            raise ValueError("model_path must be specified (local path or HF model ID)")
        if self.arch.num_layers != len(self.arch.local_global_pattern):
            raise ValueError(
                f"Layer count ({self.arch.num_layers}) does not match "
                f"attention pattern length ({len(self.arch.local_global_pattern)})"
            )
        if self.svd.top_k > min(self.arch.hidden_size, self.arch.intermediate_size):
            raise ValueError(
                f"SVD top_k ({self.svd.top_k}) exceeds matrix dimensions"
            )
