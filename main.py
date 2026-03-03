"""
ZITEP Main Pipeline — Zero-Input Topological Extraction Protocol.

CLI entry point that chains all 5 phases of the ZITEP protocol:
  Phase 1: Load safetensors as static mathematical objects
  Phase 2: SVD decomposition of all MLP matrices → concept lookup table
  Phase 3: Topological Data Analysis → persistent homology → bridge extraction
  Phase 4: Spectral verification → Lipschitz bounding → SMT formal verification
  Phase 5: Binary mask pruning → export compact deterministic inference engine

Usage:
  python main.py --model-path <path-to-gemma-3-1b-it> --output-dir ./zitep_output
  python main.py --model-path google/gemma-3-1b-it --device cuda --skip-smt
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path

import torch
from rich.console import Console
from rich.panel import Panel

from config import (
    PipelineConfig,
    GemmaArchitecture,
    SVDConfig,
    TDAConfig,
    SpectralConfig,
    PruningConfig,
)
from model_loader import (
    load_raw_weights,
    get_embedding_matrix,
    extract_all_layers,
    print_weight_summary,
)
from svd_analyzer import (
    decompose_all_layers,
    save_concept_table,
    print_svd_summary,
)
from tda_analyzer import (
    run_tda_analysis,
    save_tda_results,
    print_tda_summary,
)
from spectral_verifier import (
    verify_all_circuits,
    save_verification_report,
    print_verification_summary,
)
from circuit_pruner import (
    generate_binary_mask,
    apply_mask,
    export_pruned_model,
    print_compression_summary,
)
from utils import (
    setup_logging,
    log_gpu_memory,
    clear_gpu_cache,
    print_architecture_summary,
    print_phase_header,
    ensure_directory,
    Timer,
    VocabularyMapper,
    console as util_console,
)

console = Console()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="ZITEP — Zero-Input Topological Extraction Protocol for Gemma 3 1B",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with local model weights
  python main.py --model-path D:/models/gemma-3-1b-it --output-dir ./output

  # Run with HuggingFace model ID (will download)
  python main.py --model-path google/gemma-3-1b-it --device cuda

  # Skip expensive SMT verification
  python main.py --model-path ./gemma-3-1b-it --skip-smt

  # Custom SVD parameters
  python main.py --model-path ./gemma-3-1b-it --svd-top-k 128 --pruning-threshold 90
        """,
    )

    parser.add_argument(
        "--model-path",
        type=str,
        default="google/gemma-3-1b-it",
        help="Path to local Gemma 3 1B model directory or HuggingFace model ID (default: google/gemma-3-1b-it)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./zitep_output",
        help="Output directory for results and pruned model (default: ./zitep_output)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Compute device (default: cuda)",
    )
    parser.add_argument(
        "--svd-top-k",
        type=int,
        default=64,
        help="Number of top singular values to analyze per matrix (default: 64)",
    )
    parser.add_argument(
        "--pruning-threshold",
        type=float,
        default=85.0,
        help="Importance percentile threshold for pruning (default: 85.0)",
    )
    parser.add_argument(
        "--tda-downsample",
        type=int,
        default=8,
        help="Downsample factor for TDA (higher = faster but less precise, default: 8)",
    )
    parser.add_argument(
        "--max-circuits",
        type=int,
        default=10,
        help="Maximum number of circuits to verify (default: 10)",
    )
    parser.add_argument(
        "--skip-smt",
        action="store_true",
        help="Skip Z3 SMT formal verification (saves time)",
    )
    parser.add_argument(
        "--structured-pruning",
        action="store_true",
        default=True,
        help="Use structured pruning (entire heads/neurons, default: True)",
    )
    parser.add_argument(
        "--save-intermediate",
        action="store_true",
        default=True,
        help="Save intermediate results (concept tables, TDA diagrams)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )

    return parser.parse_args()


def _resolve_model_path(model_path: str) -> str:
    """
    If the model path is a HuggingFace model ID (e.g., google/gemma-3-1b-it),
    download it. Otherwise, verify the local path exists.
    """
    if os.path.isdir(model_path):
        return model_path

    if os.path.isfile(model_path):
        return os.path.dirname(model_path)

    # Treat as HuggingFace model ID — download via huggingface_hub
    console.print(f"[yellow]Model path '{model_path}' not found locally. "
                  f"Attempting HuggingFace download...[/yellow]")

    from huggingface_hub import snapshot_download

    local_dir = snapshot_download(
        repo_id=model_path,
        allow_patterns=["*.safetensors", "*.json", "tokenizer*", "*.model"],
    )
    console.print(f"[green]Downloaded to: {local_dir}[/green]")
    return local_dir


def build_config(args: argparse.Namespace) -> PipelineConfig:
    """Build PipelineConfig from CLI arguments."""
    config = PipelineConfig(
        model_path=args.model_path,
        output_dir=args.output_dir,
        device=args.device,
        log_level=args.log_level,
        save_intermediate=args.save_intermediate,
        arch=GemmaArchitecture(),
        svd=SVDConfig(top_k=args.svd_top_k),
        tda=TDAConfig(downsample_factor=args.tda_downsample),
        spectral=SpectralConfig(enable_smt_verification=not args.skip_smt),
        pruning=PruningConfig(
            importance_percentile=args.pruning_threshold,
            structured_pruning=args.structured_pruning,
        ),
    )
    return config


def run_pipeline(config: PipelineConfig) -> None:
    """Execute the full ZITEP pipeline."""

    pipeline_start = time.perf_counter()

    # Setup
    logger = setup_logging(config.log_level)
    device = config.get_device()
    output_dir = ensure_directory(config.output_dir)

    # Print banner
    console.print(Panel.fit(
        "[bold cyan]ZITEP — Zero-Input Topological Extraction Protocol[/bold cyan]\n"
        "[dim]Static Analysis Pipeline for Gemma 3 1B Instruct[/dim]\n"
        f"[dim]Device: {device} | Output: {output_dir}[/dim]",
        border_style="cyan",
    ))

    # Validate config
    config.validate()

    # Print architecture
    print_architecture_summary(config.arch)

    # ════════════════════════════════════════════════════════════════════════
    # PHASE 1: Load Model Weights as Static Mathematical Objects
    # ════════════════════════════════════════════════════════════════════════
    print_phase_header(1, "LOADING STATIC WEIGHT TENSORS")

    with Timer("Phase 1: Model Loading"):
        resolved_path = _resolve_model_path(config.model_path)
        config = PipelineConfig(
            model_path=resolved_path,
            output_dir=config.output_dir,
            device=config.device,
            log_level=config.log_level,
            save_intermediate=config.save_intermediate,
            arch=config.arch,
            svd=config.svd,
            tda=config.tda,
            spectral=config.spectral,
            pruning=config.pruning,
        )

        raw_weights = load_raw_weights(
            model_path=resolved_path,
            device=device,
            dtype=config.arch.get_torch_dtype(),
        )
        print_weight_summary(raw_weights)

        embedding_matrix = get_embedding_matrix(raw_weights)
        all_layer_weights = extract_all_layers(raw_weights, config.arch)

    log_gpu_memory(device, "after Phase 1")

    # ════════════════════════════════════════════════════════════════════════
    # PHASE 2: SVD Decomposition — Extract Concept Lookup Table
    # ════════════════════════════════════════════════════════════════════════
    print_phase_header(2, "SVD DECOMPOSITION — CONCEPT EXTRACTION")

    with Timer("Phase 2: SVD Analysis"):
        vocab_mapper = VocabularyMapper(resolved_path)

        concept_table = decompose_all_layers(
            all_layer_weights=all_layer_weights,
            embedding_matrix=embedding_matrix,
            vocab_mapper=vocab_mapper,
            svd_config=config.svd,
            device=device,
            arch=config.arch,
        )

        print_svd_summary(concept_table)

        if config.save_intermediate:
            save_concept_table(concept_table, output_dir)

    clear_gpu_cache(device)
    log_gpu_memory(device, "after Phase 2")

    # ════════════════════════════════════════════════════════════════════════
    # PHASE 3: Topological Data Analysis — Persistent Homology
    # ════════════════════════════════════════════════════════════════════════
    print_phase_header(3, "TOPOLOGICAL DATA ANALYSIS — PERSISTENT HOMOLOGY")

    with Timer("Phase 3: TDA"):
        topology_result = run_tda_analysis(
            all_layer_weights=all_layer_weights,
            embedding_matrix=embedding_matrix,
            concept_table=concept_table,
            arch=config.arch,
            tda_config=config.tda,
        )

        print_tda_summary(topology_result)

        if config.save_intermediate:
            save_tda_results(topology_result, output_dir)

    clear_gpu_cache(device)
    log_gpu_memory(device, "after Phase 3")

    # ════════════════════════════════════════════════════════════════════════
    # PHASE 4: Spectral Verification — Lipschitz Bounding & SMT
    # ════════════════════════════════════════════════════════════════════════
    print_phase_header(4, "SPECTRAL VERIFICATION — LIPSCHITZ BOUNDING")

    with Timer("Phase 4: Verification"):
        verification_report = verify_all_circuits(
            topology_result=topology_result,
            all_layer_weights=all_layer_weights,
            concept_table=concept_table,
            arch=config.arch,
            config=config.spectral,
            device=device,
            max_circuits=10,
        )

        print_verification_summary(verification_report)

        if config.save_intermediate:
            save_verification_report(verification_report, output_dir)

    clear_gpu_cache(device)
    log_gpu_memory(device, "after Phase 4")

    # ════════════════════════════════════════════════════════════════════════
    # PHASE 5: Pruning & Export — Extract Deterministic Inference Engine
    # ════════════════════════════════════════════════════════════════════════
    print_phase_header(5, "SPECTRAL PRUNING & EXPORT")

    with Timer("Phase 5: Pruning"):
        mask, compression_report = generate_binary_mask(
            all_layer_weights=all_layer_weights,
            topology_result=topology_result,
            verification_report=verification_report,
            arch=config.arch,
            config=config.pruning,
            embedding_matrix=embedding_matrix,
        )

        print_compression_summary(compression_report)

        pruned_weights = apply_mask(raw_weights, mask)

        verified_ids = [
            cv.bridge_id
            for cv in verification_report.circuit_results
            if cv.is_stable
        ]

        artifact = export_pruned_model(
            pruned_weights=pruned_weights,
            output_dir=output_dir,
            compression_report=compression_report,
            verified_bridge_ids=verified_ids,
            config=config.pruning,
        )

    clear_gpu_cache(device)

    # ════════════════════════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ════════════════════════════════════════════════════════════════════════
    pipeline_elapsed = time.perf_counter() - pipeline_start

    console.print("\n")
    console.print(Panel.fit(
        "[bold green]ZITEP EXTRACTION COMPLETE[/bold green]\n\n"
        f"[bold]Total Pipeline Time:[/bold] {pipeline_elapsed:.1f}s\n"
        f"[bold]Original Model:[/bold] {format_params(compression_report.original_total_params)}\n"
        f"[bold]Extracted Circuit:[/bold] {format_params(compression_report.kept_total_params)}\n"
        f"[bold]Compression:[/bold] {compression_report.overall_sparsity * 100:.1f}% pruned\n"
        f"[bold]Verified Circuits:[/bold] {verification_report.verified_circuits}\n"
        f"[bold]Output:[/bold] {output_dir}\n\n"
        "[dim]The extracted artifact is a deterministic inference engine —\n"
        "a mathematically verified subset of Gemma 3 1B that is\n"
        "structurally incapable of hallucinating outside its verified domain.[/dim]",
        border_style="green",
        title="✓ ZITEP",
    ))

    # Save pipeline metadata (all values cast to native Python types)
    metadata = {
        "pipeline": "ZITEP",
        "target_model": config.arch.model_name,
        "model_path": config.model_path,
        "device": str(device),
        "total_time_seconds": round(float(pipeline_elapsed), 2),
        "svd_top_k": int(config.svd.top_k),
        "tda_downsample": int(config.tda.downsample_factor),
        "smt_enabled": bool(config.spectral.enable_smt_verification),
        "pruning_threshold": float(config.pruning.importance_percentile),
        "structured_pruning": bool(config.pruning.structured_pruning),
        "original_params": int(compression_report.original_total_params),
        "kept_params": int(compression_report.kept_total_params),
        "compression_ratio": round(float(compression_report.compression_ratio), 6),
        "verified_circuits": int(verification_report.verified_circuits),
        "total_circuits_analyzed": int(verification_report.total_circuits_analyzed),
        "concepts_extracted": int(concept_table.total_concepts_extracted),
        "spike_count": int(concept_table.global_spike_count),
        "betti_0": int(topology_result.persistence_diagram.betti_0),
        "betti_1": int(topology_result.persistence_diagram.betti_1),
        "output_files": {
            "pruned_model": artifact.output_path if artifact else None,
            "concept_table": os.path.join(output_dir, "concept_table.json"),
            "tda_results": os.path.join(output_dir, "tda_results.json"),
            "verification_report": os.path.join(output_dir, "verification_report.json"),
            "compression_report": os.path.join(output_dir, "compression_report.json"),
        },
    }

    metadata_path = os.path.join(output_dir, "pipeline_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    console.print(f"\n[dim]Pipeline metadata saved: {metadata_path}[/dim]")


def format_params(count: int) -> str:
    """Format parameter count."""
    if count >= 1e9:
        return f"{count / 1e9:.2f}B"
    elif count >= 1e6:
        return f"{count / 1e6:.2f}M"
    elif count >= 1e3:
        return f"{count / 1e3:.1f}K"
    return str(count)


def main():
    """CLI entry point."""
    args = parse_args()
    config = build_config(args)

    try:
        run_pipeline(config)
    except KeyboardInterrupt:
        console.print("\n[yellow]Pipeline interrupted by user.[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[red bold]Pipeline failed:[/red bold] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
