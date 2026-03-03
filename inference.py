"""
ZITEP Inference Engine — Run inference on the pruned deterministic circuit.

Loads the ZITEP-pruned safetensors artifact and performs text generation using
only the verified reasoning pathway. Since the pruned model retains the full
Gemma 3 1B architecture (with zeroed-out weights outside the circuit), it can
be loaded directly into the HuggingFace Gemma model class for standard inference.

Additionally provides a raw manual forward pass mode that bypasses HuggingFace
entirely for maximum transparency — every matrix multiply is explicit.

Usage:
  # Standard HuggingFace inference with pruned weights
  python inference.py --model-path ./zitep_output/zitep_pruned_model.safetensors --prompt "Diagnose: elevated troponin, ST elevation"

  # Manual raw forward pass (full transparency, no HF dependency)
  python inference.py --model-path ./zitep_output/zitep_pruned_model.safetensors --prompt "Symptoms: chest pain" --mode raw

  # Batch inference from file
  python inference.py --model-path ./zitep_output/zitep_pruned_model.safetensors --input-file prompts.txt --output-file results.json
"""

import os
import sys
import json
import math
import argparse
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import numpy as np
from safetensors import safe_open
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.columns import Columns
from rich.text import Text

console = Console()


# ─────────────────────────────────────────────────────────────────────────────
# ZITEP Interpretation Pipeline
# ─────────────────────────────────────────────────────────────────────────────

class ZITEPInterpreter:
    """Loads ZITEP pipeline metadata and provides circuit-level interpretability."""

    def __init__(self, zitep_dir: str):
        self.zitep_dir = zitep_dir
        self.tda = None
        self.verification = None
        self.compression = None
        self.concepts = None
        self.metadata = None
        self._load_all()

    def _load_json(self, filename: str):
        path = os.path.join(self.zitep_dir, filename)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return None

    def _load_all(self):
        self.tda = self._load_json("tda_results.json")
        self.verification = self._load_json("verification_report.json")
        self.compression = self._load_json("compression_report.json")
        self.concepts = self._load_json("concept_table.json")
        self.metadata = self._load_json("pipeline_metadata.json")

        loaded = sum(1 for x in [self.tda, self.verification, self.compression, self.concepts, self.metadata] if x)
        console.print(f"[dim]Loaded {loaded}/5 ZITEP analysis files from {self.zitep_dir}[/dim]")

    def display_circuit_summary(self):
        """Display pre-inference circuit topology summary."""
        # ── Header panel ──
        betti_0 = self.tda["persistence_diagram"]["betti_0"] if self.tda else "?"
        betti_1 = self.tda["persistence_diagram"]["betti_1"] if self.tda else "?"
        num_bridges = len(self.tda.get("bridges", [])) if self.tda else 0
        num_nodes = self.tda.get("num_nodes", 0) if self.tda else 0

        verified = self.verification.get("verified_circuits", 0) if self.verification else 0
        total_analyzed = self.verification.get("total_circuits_analyzed", 0) if self.verification else 0
        lip_range = self.verification.get("overall_lipschitz_range", [0, 0]) if self.verification else [0, 0]

        sparsity = self.compression.get("overall_sparsity", 0) if self.compression else 0
        original_mb = self.compression.get("original_size_mb", 0) if self.compression else 0
        pruned_mb = self.compression.get("pruned_size_mb", 0) if self.compression else 0

        concepts_extracted = self.metadata.get("concepts_extracted", 0) if self.metadata else 0

        summary_text = (
            f"[bold cyan]Topological Invariants[/bold cyan]\n"
            f"  β₀ (connected components): [yellow]{betti_0}[/yellow]\n"
            f"  β₁ (loops/cycles):         [yellow]{betti_1}[/yellow]\n"
            f"  Weight graph nodes:         {num_nodes}\n"
            f"  Topological bridges:        {num_bridges}\n\n"
            f"[bold cyan]Spectral Verification[/bold cyan]\n"
            f"  Circuits analyzed:  {total_analyzed}\n"
            f"  Verified stable:    [green]{verified}[/green]\n"
            f"  Lipschitz range:    [{lip_range[0]:.2f}, {lip_range[1]:.2f}]\n\n"
            f"[bold cyan]Model Compression[/bold cyan]\n"
            f"  Sparsity:   {sparsity * 100:.1f}%\n"
            f"  Original:   {original_mb:.1f} MB\n"
            f"  Pruned:     {pruned_mb:.1f} MB\n"
            f"  SVD concepts extracted: {concepts_extracted}"
        )
        console.print(Panel(summary_text, title="ZITEP Circuit Analysis", border_style="cyan"))

        # ── Bridge details ──
        if self.tda and self.tda.get("bridges"):
            bridge_table = Table(
                title="Topological Bridges — Verified Reasoning Pathways",
                show_lines=True,
                border_style="cyan",
            )
            bridge_table.add_column("ID", style="cyan", width=4)
            bridge_table.add_column("Importance", style="yellow", width=12)
            bridge_table.add_column("Layers", style="dim", width=10)
            bridge_table.add_column("Global?", style="green", width=7)
            bridge_table.add_column("Associated Concepts", style="white", width=50)
            bridge_table.add_column("Stability", style="magenta", width=12)

            # Build Lipschitz lookup from verification
            lip_lookup = {}
            if self.verification and "circuits" in self.verification:
                for circ in self.verification["circuits"]:
                    lip_lookup[circ["bridge_id"]] = circ

            for bridge in self.tda["bridges"][:10]:  # Show top 10
                bid = bridge["bridge_id"]
                importance = f"{bridge['importance_score']:.1f}"
                layers = f"{len(bridge['path_layers'])} layers"
                is_global = "✓" if bridge.get("passes_through_global") else "—"
                concepts = ", ".join(bridge.get("associated_concepts", [])[:8])

                circ = lip_lookup.get(bid, {})
                lip_k = circ.get("lipschitz_constant", None)
                stable = circ.get("is_stable", None)
                if lip_k is not None and lip_k < 1e15:
                    stability = f"K={lip_k:.1f} {'✓' if stable else '✗'}"
                elif lip_k is not None:
                    stability = "K=∞ ✗"
                else:
                    stability = "—"

                bridge_table.add_row(str(bid), importance, layers, is_global, concepts, stability)

            console.print(bridge_table)

    def _compute_lipschitz_stability(self) -> float:
        """
        Compute global circuit stability factor S ∈ (0, 1] from Lipschitz bounds.

        S = 1 / (1 + ln(K_geo))

        where K_geo is the geometric mean of all verified circuit Lipschitz constants.
        S → 1 when K → 1 (perfectly stable), S → 0 when K → ∞ (unstable).
        """
        if not self.verification or not self.verification.get("circuits"):
            return 0.5  # Uninformative prior when no verification data

        lip_values = [
            c["lipschitz_constant"]
            for c in self.verification["circuits"]
            if c.get("lipschitz_constant") is not None
            and c["lipschitz_constant"] > 0
            and c["lipschitz_constant"] < 1e15
        ]

        if not lip_values:
            return 0.5

        # Geometric mean of Lipschitz constants
        k_geo = np.exp(np.mean(np.log(lip_values)))
        stability = 1.0 / (1.0 + np.log(max(k_geo, 1.0)))
        return float(stability)

    def compute_token_risk(self, entropy: float, top1_prob: float,
                           top2_prob: float, stability: float) -> float:
        """
        Compute per-token hallucination risk score R(t) ∈ [0, 1].

        R(t) = 1 - C(t) × M(t) × S

        where:
          C(t) = p₁       — confidence (top-1 probability)
          M(t) = 1 - p₂/p₁  — decision margin (0 = ambiguous, 1 = decisive)
          S    = 1/(1+ln(K))  — circuit stability from Lipschitz bound

        Properties:
          - R = 0 ↔ confident (p₁=1), decisive (p₂=0), stable (K=1)
          - R → 1 when any factor degrades
          - Multiplicative: all three must be strong for low risk
        """
        confidence = max(top1_prob, 1e-12)
        margin = 1.0 - (top2_prob / confidence) if confidence > 1e-12 else 0.0
        margin = max(margin, 0.0)

        risk = 1.0 - (confidence * margin * stability)
        return max(0.0, min(1.0, risk))

    def _auto_calibrate_thresholds(self, risk_scores: list) -> tuple:
        """
        Auto-calibrate risk thresholds from the generation's own distribution.

        Returns (low_threshold, high_threshold):
          low  = μ(R)
          high = μ(R) + σ(R)

        Tokens below μ → LOW, between μ and μ+σ → MEDIUM, above μ+σ → HIGH.
        No hardcoded cutoffs — adapts to each inference run.
        """
        if not risk_scores:
            return (0.33, 0.66)

        mu = float(np.mean(risk_scores))
        sigma = float(np.std(risk_scores))
        low_thresh = mu
        high_thresh = mu + max(sigma, 0.05)
        return (low_thresh, high_thresh)

    def classify_risk(self, entropy: float, top_prob: float,
                      token: str = "", context_tokens: list = None) -> tuple:
        """Backward-compatible single-token risk. Returns (level, score)."""
        stability = self._compute_lipschitz_stability()
        risk = self.compute_token_risk(entropy, top_prob, 0.0, stability)
        if risk < 0.33:
            return ("LOW", round(risk, 4))
        elif risk < 0.66:
            return ("MEDIUM", round(risk, 4))
        else:
            return ("HIGH", round(risk, 4))

    def display_token_interpretation(self, per_token_metadata: list):
        """
        Per-token hallucination risk with mathematically derived scores.

        R(t) = 1 - C(t) × M(t) × S
        Thresholds auto-calibrated: μ(R) and μ(R) + σ(R)
        """
        stability = self._compute_lipschitz_stability()

        # First pass: compute all risk scores
        risk_scores = []
        for tm in per_token_metadata:
            top5 = tm.get("top5", [])
            p1 = top5[0]["prob"] if len(top5) >= 1 else 0.0
            p2 = top5[1]["prob"] if len(top5) >= 2 else 0.0
            risk = self.compute_token_risk(tm["entropy"], p1, p2, stability)
            tm["_risk_score"] = risk
            risk_scores.append(risk)

        # Auto-calibrate thresholds
        low_thresh, high_thresh = self._auto_calibrate_thresholds(risk_scores)

        for tm in per_token_metadata:
            r = tm["_risk_score"]
            if r < low_thresh:
                tm["_risk"] = "LOW"
            elif r < high_thresh:
                tm["_risk"] = "MEDIUM"
            else:
                tm["_risk"] = "HIGH"

        # Build table
        table = Table(
            title="Per-Token Circuit Interpretation",
            show_lines=False,
            border_style="cyan",
        )
        table.add_column("Step", style="cyan", width=5)
        table.add_column("Token", style="green", width=18)
        table.add_column("H(t)", style="yellow", width=7)
        table.add_column("C(t)", style="magenta", width=7)
        table.add_column("M(t)", style="dim", width=7)
        table.add_column("R(t)", style="white", width=7)
        table.add_column("Risk", width=8)
        table.add_column("Top Alt", style="white", width=18)

        for tm in per_token_metadata[:50]:
            top5 = tm.get("top5", [])
            p1 = top5[0]["prob"] if len(top5) >= 1 else 0.0
            p2 = top5[1]["prob"] if len(top5) >= 2 else 0.0
            margin = 1.0 - (p2 / p1) if p1 > 1e-12 else 0.0
            risk_val = tm["_risk_score"]
            risk_level = tm["_risk"]
            risk_style = {"LOW": "green", "MEDIUM": "yellow", "HIGH": "bold red"}[risk_level]
            top_tok = top5[0]["token"][:18] if top5 else "—"

            table.add_row(
                str(tm["step"]),
                tm.get("token", "")[:18],
                f"{tm['entropy']:.3f}",
                f"{p1:.3f}",
                f"{margin:.3f}",
                f"{risk_val:.3f}",
                f"[{risk_style}]{risk_level}[/{risk_style}]",
                top_tok,
            )

        console.print(table)

        # Summary panel
        total = len(per_token_metadata)
        risk_counts = {"LOW": 0, "MEDIUM": 0, "HIGH": 0}
        for tm in per_token_metadata:
            risk_counts[tm["_risk"]] += 1

        entropies = [tm["entropy"] for tm in per_token_metadata]
        max_risk_idx = int(np.argmax(risk_scores)) if risk_scores else 0

        summary = (
            f"[bold cyan]Risk Model: R(t) = 1 - C(t) × M(t) × S[/bold cyan]\n"
            f"  C(t) = top-1 probability  |  M(t) = (p₁-p₂)/p₁  |  S = 1/(1+ln K)\n"
            f"  Lipschitz stability S = {stability:.4f}\n"
            f"  Auto-calibrated: LOW < {low_thresh:.3f} < MEDIUM < {high_thresh:.3f} < HIGH\n\n"
            f"[green]Low risk:    {risk_counts['LOW']:3d}/{total} ({risk_counts['LOW']/total*100:.0f}%)[/green]\n"
            f"[yellow]Medium risk: {risk_counts['MEDIUM']:3d}/{total} ({risk_counts['MEDIUM']/total*100:.0f}%)[/yellow]\n"
            f"[red]High risk:   {risk_counts['HIGH']:3d}/{total} ({risk_counts['HIGH']/total*100:.0f}%)[/red]\n\n"
            f"Avg R(t): {np.mean(risk_scores):.4f} | "
            f"Max R(t): {max(risk_scores):.4f} (step {max_risk_idx}: {per_token_metadata[max_risk_idx]['token']!r})\n"
            f"Avg H(t): {np.mean(entropies):.4f} | Max H(t): {max(entropies):.4f}"
        )
        console.print(Panel(summary, title="Hallucination Risk Analysis", border_style="yellow"))

    def get_concept_summary(self) -> str:
        """Get a brief summary of the concept space."""
        if not self.concepts:
            return "No concept data available"

        total = self.concepts.get("total_concepts_extracted", 0)
        spikes = self.concepts.get("global_spike_count", 0)
        layers = len(self.concepts.get("layers", []))
        return f"{total} concepts across {layers} layers ({spikes} spikes detected)"





# ─────────────────────────────────────────────────────────────────────────────
# Architecture constants (must match the model that was pruned)
# ─────────────────────────────────────────────────────────────────────────────

NUM_LAYERS = 26
HIDDEN_SIZE = 1152
INTERMEDIATE_SIZE = 6912
NUM_ATTENTION_HEADS = 4
NUM_KV_HEADS = 1
HEAD_DIM = 256
VOCAB_SIZE = 262144
RMS_NORM_EPS = 1e-6
ROPE_BASE_LOCAL = 10_000.0
ROPE_BASE_GLOBAL = 1_000_000.0

# 5:1 interleaved attention pattern
LAYER_TYPES = [
    "local", "local", "local", "local", "local", "global",
    "local", "local", "local", "local", "local", "global",
    "local", "local", "local", "local", "local", "global",
    "local", "local", "local", "local", "local", "global",
    "local", "local",
]


# ─────────────────────────────────────────────────────────────────────────────
# Weight Loading
# ─────────────────────────────────────────────────────────────────────────────

def load_pruned_weights(
    safetensors_path: str,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Load the ZITEP-pruned safetensors into a flat dict on the target device."""
    console.print(f"[cyan]Loading pruned model from:[/cyan] {safetensors_path}")

    weights = {}
    with safe_open(safetensors_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            weights[key] = f.get_tensor(key).to(device)

    total_params = sum(t.numel() for t in weights.values())
    nonzero_params = sum((t != 0).sum().item() for t in weights.values())
    sparsity = 1.0 - (nonzero_params / total_params) if total_params > 0 else 0.0

    console.print(f"[green]Loaded {len(weights)} tensors[/green]")
    console.print(f"[yellow]Total params: {total_params:,} | "
                  f"Non-zero: {nonzero_params:,} | "
                  f"Sparsity: {sparsity:.1%}[/yellow]")

    return weights


# ─────────────────────────────────────────────────────────────────────────────
# Tokenizer
# ─────────────────────────────────────────────────────────────────────────────

def load_tokenizer(tokenizer_source: str = "google/gemma-3-1b-it"):
    """Load the Gemma tokenizer from HuggingFace."""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
    return tokenizer


# ─────────────────────────────────────────────────────────────────────────────
# Raw Manual Forward Pass (No HuggingFace model class)
# ─────────────────────────────────────────────────────────────────────────────

def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = RMS_NORM_EPS) -> torch.Tensor:
    """Apply RMS normalization."""
    variance = x.float().pow(2).mean(-1, keepdim=True)
    x_normed = x * torch.rsqrt(variance + eps)
    return (x_normed * weight).to(x.dtype)


def apply_rotary_emb(
    x: torch.Tensor,
    positions: torch.Tensor,
    rope_base: float,
) -> torch.Tensor:
    """Apply Rotary Position Embeddings (RoPE) to query or key tensors."""
    # x shape: (batch, num_heads, seq_len, head_dim)
    head_dim = x.shape[-1]
    half_dim = head_dim // 2

    # Compute frequencies
    freqs = 1.0 / (rope_base ** (torch.arange(0, half_dim, dtype=torch.float32, device=x.device) / half_dim))
    # positions shape: (seq_len,)
    angles = positions.unsqueeze(-1).float() * freqs.unsqueeze(0)  # (seq_len, half_dim)

    cos_vals = torch.cos(angles).unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, half_dim)
    sin_vals = torch.sin(angles).unsqueeze(0).unsqueeze(0)

    x1 = x[..., :half_dim]
    x2 = x[..., half_dim:]

    rotated = torch.cat([
        x1 * cos_vals - x2 * sin_vals,
        x2 * cos_vals + x1 * sin_vals,
    ], dim=-1)

    return rotated.to(x.dtype)


def gelu_approx(x: torch.Tensor) -> torch.Tensor:
    """GELU activation with tanh approximation (matches Gemma's gelu_pytorch_tanh)."""
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))))


def manual_attention(
    hidden_states: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    v_weight: torch.Tensor,
    o_weight: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    positions: torch.Tensor,
    layer_type: str,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute grouped-query attention manually.
    4 query heads, 1 KV head → the KV head is shared across all 4 query heads.
    Gemma 3 applies RMS norm to Q and K after projection (q_norm, k_norm).
    """
    batch_size, seq_len, _ = hidden_states.shape

    # Project Q, K, V
    q = F.linear(hidden_states, q_weight)  # (B, S, num_heads * head_dim)
    k = F.linear(hidden_states, k_weight)  # (B, S, num_kv_heads * head_dim)
    v = F.linear(hidden_states, v_weight)  # (B, S, num_kv_heads * head_dim)

    # Reshape to (B, num_heads, S, head_dim)
    q = q.view(batch_size, seq_len, NUM_ATTENTION_HEADS, HEAD_DIM).transpose(1, 2)
    k = k.view(batch_size, seq_len, NUM_KV_HEADS, HEAD_DIM).transpose(1, 2)
    v = v.view(batch_size, seq_len, NUM_KV_HEADS, HEAD_DIM).transpose(1, 2)

    # Gemma 3 QK normalization — per-head RMS norm on Q and K before RoPE
    q = rms_norm(q, q_norm_weight)
    k = rms_norm(k, k_norm_weight)

    # Apply RoPE
    rope_base = ROPE_BASE_GLOBAL if layer_type == "global" else ROPE_BASE_LOCAL
    q = apply_rotary_emb(q, positions, rope_base)
    k = apply_rotary_emb(k, positions, rope_base)

    # Expand KV heads for GQA: 1 KV head → 4 query heads (repeat KV head 4 times)
    kv_repeat = NUM_ATTENTION_HEADS // NUM_KV_HEADS  # = 4
    k = k.repeat_interleave(kv_repeat, dim=1)  # (B, 4, S, head_dim)
    v = v.repeat_interleave(kv_repeat, dim=1)

    # Scaled dot-product attention
    scale = 1.0 / math.sqrt(HEAD_DIM)
    attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B, 4, S, S)

    # Apply causal mask
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)

    # Attention output
    attn_output = torch.matmul(attn_weights, v)  # (B, 4, S, head_dim)
    attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

    # Output projection
    output = F.linear(attn_output, o_weight)
    return output


def manual_mlp(
    hidden_states: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
) -> torch.Tensor:
    """GeGLU MLP: gate(x) * GELU(gate) ⊙ up(x) → down projection."""
    gate = F.linear(hidden_states, gate_weight)
    up = F.linear(hidden_states, up_weight)
    activated = gelu_approx(gate) * up
    output = F.linear(activated, down_weight)
    return output


def manual_forward_pass(
    input_ids: torch.Tensor,
    weights: Dict[str, torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    """
    Full manual forward pass through the pruned Gemma 3 1B model.
    Returns logits for the last token position.
    """
    batch_size, seq_len = input_ids.shape

    # 1. Token embedding (with Gemma's √hidden_size scaling)
    embed_weight = weights["model.embed_tokens.weight"]
    hidden_states = F.embedding(input_ids, embed_weight)
    hidden_states = hidden_states * math.sqrt(HIDDEN_SIZE)  # Gemma scaling

    # 2. Create causal attention mask
    causal_mask = torch.triu(
        torch.full((seq_len, seq_len), float("-inf"), device=device),
        diagonal=1,
    ).unsqueeze(0).unsqueeze(0)  # (1, 1, S, S)

    # Position indices
    positions = torch.arange(seq_len, device=device)

    # 3. Pass through all 26 transformer layers
    for layer_idx in range(NUM_LAYERS):
        prefix = f"model.layers.{layer_idx}"
        layer_type = LAYER_TYPES[layer_idx]

        # --- Self Attention ---
        residual = hidden_states
        hidden_states = rms_norm(
            hidden_states,
            weights[f"{prefix}.input_layernorm.weight"],
        )

        hidden_states = manual_attention(
            hidden_states=hidden_states,
            q_weight=weights[f"{prefix}.self_attn.q_proj.weight"],
            k_weight=weights[f"{prefix}.self_attn.k_proj.weight"],
            v_weight=weights[f"{prefix}.self_attn.v_proj.weight"],
            o_weight=weights[f"{prefix}.self_attn.o_proj.weight"],
            q_norm_weight=weights[f"{prefix}.self_attn.q_norm.weight"],
            k_norm_weight=weights[f"{prefix}.self_attn.k_norm.weight"],
            positions=positions,
            layer_type=layer_type,
            attention_mask=causal_mask,
        )

        # Post-attention norm (Gemma 3 uses post-norms)
        post_attn_norm_key = f"{prefix}.post_attention_layernorm.weight"
        if post_attn_norm_key in weights:
            hidden_states = rms_norm(hidden_states, weights[post_attn_norm_key])

        hidden_states = residual + hidden_states

        # --- MLP ---
        residual = hidden_states

        pre_ff_norm_key = f"{prefix}.pre_feedforward_layernorm.weight"
        if pre_ff_norm_key in weights:
            hidden_states = rms_norm(hidden_states, weights[pre_ff_norm_key])

        hidden_states = manual_mlp(
            hidden_states=hidden_states,
            gate_weight=weights[f"{prefix}.mlp.gate_proj.weight"],
            up_weight=weights[f"{prefix}.mlp.up_proj.weight"],
            down_weight=weights[f"{prefix}.mlp.down_proj.weight"],
        )

        post_ff_norm_key = f"{prefix}.post_feedforward_layernorm.weight"
        if post_ff_norm_key in weights:
            hidden_states = rms_norm(hidden_states, weights[post_ff_norm_key])

        hidden_states = residual + hidden_states

    # 4. Final RMS norm
    hidden_states = rms_norm(hidden_states, weights["model.norm.weight"])

    # 5. Logits via tied embedding (no separate lm_head — Gemma ties weights)
    logits = F.linear(hidden_states, embed_weight)  # (B, S, vocab_size)

    return logits


# ─────────────────────────────────────────────────────────────────────────────
# HuggingFace-based Inference (loads pruned weights into standard model)
# ─────────────────────────────────────────────────────────────────────────────

def hf_inference(
    prompt: str,
    pruned_safetensors_path: str,
    device: torch.device,
    max_new_tokens: int = 256,
    temperature: float = 0.1,
    top_k: int = 50,
    tokenizer_source: str = "google/gemma-3-1b-it",
) -> Tuple[str, dict]:
    """
    Load the pruned weights into a HuggingFace Gemma model and generate text.
    Returns (generated_text, per_token_metadata) for the interpretation pipeline.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    console.print("[cyan]Loading HuggingFace model with pruned weights...[/cyan]")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)

    # Load the base model architecture, then override with pruned weights
    model = AutoModelForCausalLM.from_pretrained(
        tokenizer_source,
        torch_dtype=torch.bfloat16,
        device_map=str(device),
    )

    # Override with pruned weights if a specific safetensors file is provided
    if pruned_safetensors_path and os.path.isfile(pruned_safetensors_path) and pruned_safetensors_path.endswith(".safetensors"):
        console.print(f"[cyan]Loading pruned weights from {pruned_safetensors_path}[/cyan]")
        pruned_weights = load_pruned_weights(pruned_safetensors_path, device)
        model_state = model.state_dict()

        replaced = 0
        for name, tensor in pruned_weights.items():
            if name in model_state:
                model_state[name].copy_(tensor)
                replaced += 1

        model.load_state_dict(model_state)
        console.print(f"[green]Replaced {replaced} weight tensors with pruned versions[/green]")
    else:
        console.print(f"[yellow]Skipping weight override. Using base model from: {pruned_safetensors_path or tokenizer_source}[/yellow]")


    # Apply Gemma 3 instruct chat template
    chat = [{"role": "user", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(
        chat, tokenize=False, add_generation_prompt=True,
    )
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[1]

    # Generate with scores for per-token analysis
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=max(temperature, 0.01),  # Avoid div-by-zero
            top_k=top_k,
            do_sample=temperature > 0,
            output_scores=True,
            return_dict_in_generate=True,
        )

    # Decode response
    generated_ids = outputs.sequences[0][input_len:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)

    # Build per-token metadata from scores
    per_token = []
    for step_idx, score in enumerate(outputs.scores):
        logits = score[0]  # (vocab_size,) for batch_size=1
        probs = F.softmax(logits.float(), dim=-1)

        # Entropy
        log_probs = torch.log2(probs + 1e-12)
        entropy = -(probs * log_probs).sum().item()

        # Top-5
        top5_probs, top5_ids = torch.topk(probs, min(5, probs.shape[0]))
        top5_tokens = tokenizer.convert_ids_to_tokens(top5_ids.tolist())

        # Actual generated token
        token_id = generated_ids[step_idx].item()
        token_str = tokenizer.convert_ids_to_tokens([token_id])[0]

        per_token.append({
            "step": step_idx,
            "token_id": token_id,
            "token": token_str,
            "entropy": round(entropy, 4),
            "top5": [
                {"token": t, "token_id": int(tid), "prob": round(float(p), 6)}
                for t, tid, p in zip(top5_tokens, top5_ids.tolist(), top5_probs.tolist())
            ],
        })

    # Aggregate metadata
    all_entropies = [t["entropy"] for t in per_token]
    metadata = {
        "prompt": prompt,
        "response": response,
        "num_tokens_generated": len(per_token),
        "avg_entropy": round(float(np.mean(all_entropies)) if all_entropies else 0.0, 4),
        "max_entropy": round(float(max(all_entropies)) if all_entropies else 0.0, 4),
        "per_token": per_token,
    }

    return response, metadata


# ─────────────────────────────────────────────────────────────────────────────
# Raw Inference (fully transparent, no HF model class)
# ─────────────────────────────────────────────────────────────────────────────

def raw_inference(
    prompt: str,
    weights: Dict[str, torch.Tensor],
    tokenizer,
    device: torch.device,
    max_new_tokens: int = 256,
    temperature: float = 0.1,
    top_k: int = 50,
) -> Tuple[str, dict]:
    """
    Run inference using the manual forward pass — every matrix multiply is explicit.
    Returns the generated text and per-token metadata (logits entropy, top candidates).
    """
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    generated_ids = input_ids.clone()
    token_metadata = []

    for step in range(max_new_tokens):
        with torch.no_grad():
            logits = manual_forward_pass(generated_ids, weights, device)

        # Get logits for the last position
        next_logits = logits[:, -1, :]  # (1, vocab_size)

        # Temperature scaling
        if temperature > 0:
            next_logits = next_logits / temperature

        # Top-k filtering
        if top_k > 0:
            topk_values, topk_indices = torch.topk(next_logits, top_k)
            mask = torch.full_like(next_logits, float("-inf"))
            mask.scatter_(1, topk_indices, topk_values)
            next_logits = mask

        # Sample or greedy
        probs = F.softmax(next_logits, dim=-1)

        if temperature > 0:
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = next_logits.argmax(dim=-1, keepdim=True)

        # Compute entropy for this prediction
        entropy = -(probs * (probs + 1e-10).log()).sum(dim=-1).item()

        # Top-5 candidates
        top5_probs, top5_ids = torch.topk(probs, 5)
        top5_tokens = tokenizer.convert_ids_to_tokens(top5_ids[0].tolist())
        top5_probs_list = top5_probs[0].tolist()

        token_metadata.append({
            "step": step,
            "token_id": next_token.item(),
            "token": tokenizer.decode(next_token[0]),
            "entropy": round(entropy, 4),
            "top5": [
                {"token": t, "prob": round(p, 4)}
                for t, p in zip(top5_tokens, top5_probs_list)
            ],
        })

        # Check for EOS
        if next_token.item() == tokenizer.eos_token_id:
            break

        generated_ids = torch.cat([generated_ids, next_token], dim=-1)

    # Decode full output
    output_ids = generated_ids[0][input_ids.shape[1]:]
    generated_text = tokenizer.decode(output_ids, skip_special_tokens=True)

    metadata = {
        "prompt": prompt,
        "generated_text": generated_text,
        "num_tokens_generated": len(token_metadata),
        "avg_entropy": round(np.mean([m["entropy"] for m in token_metadata]), 4) if token_metadata else 0.0,
        "per_token": token_metadata,
    }

    return generated_text, metadata


# ─────────────────────────────────────────────────────────────────────────────
# Batch Inference
# ─────────────────────────────────────────────────────────────────────────────

def batch_inference(
    input_file: str,
    output_file: str,
    weights: Dict[str, torch.Tensor],
    tokenizer,
    device: torch.device,
    max_new_tokens: int = 256,
    temperature: float = 0.1,
) -> None:
    """Run inference on a file of prompts (one per line) and save results."""
    with open(input_file, "r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f if line.strip()]

    console.print(f"[cyan]Processing {len(prompts)} prompts from {input_file}[/cyan]")

    results = []
    for i, prompt in enumerate(prompts):
        console.print(f"\n[dim]─── Prompt {i+1}/{len(prompts)} ───[/dim]")
        console.print(f"[white]{prompt}[/white]")

        start = time.perf_counter()
        text, metadata = raw_inference(
            prompt=prompt,
            weights=weights,
            tokenizer=tokenizer,
            device=device,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        elapsed = time.perf_counter() - start

        metadata["inference_time_seconds"] = round(elapsed, 3)
        results.append(metadata)

        console.print(f"[green]{text}[/green]")
        console.print(f"[dim]({elapsed:.2f}s, {metadata['num_tokens_generated']} tokens, "
                      f"avg entropy: {metadata['avg_entropy']})[/dim]")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    console.print(f"\n[green]Results saved to {output_file}[/green]")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ZITEP Inference Engine — run the pruned deterministic circuit",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the ZITEP-pruned .safetensors file",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Single prompt for inference",
    )
    parser.add_argument(
        "--input-file",
        type=str,
        default=None,
        help="File with one prompt per line (batch mode)",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="zitep_inference_results.json",
        help="Output file for batch results (default: zitep_inference_results.json)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="hf",
        choices=["raw", "hf"],
        help="Inference mode: 'hf' (HuggingFace generate, recommended) or 'raw' (manual forward pass)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Maximum new tokens to generate (default: 256)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Sampling temperature (0 = greedy, default: 0.1)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Top-k filtering (default: 50)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Compute device (default: cuda)",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="google/gemma-3-1b-it",
        help="Tokenizer source (default: google/gemma-3-1b-it)",
    )
    parser.add_argument(
        "--zitep-dir",
        type=str,
        default="./zitep_output",
        help="Path to ZITEP output directory with analysis JSONs (default: ./zitep_output)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    console.print(Panel.fit(
        "[bold cyan]ZITEP Inference Engine[/bold cyan]\n"
        f"[dim]Mode: {args.mode} | Device: {device} | Temp: {args.temperature}[/dim]",
        border_style="cyan",
    ))

    if not args.prompt and not args.input_file:
        console.print("[red]Error: provide either --prompt or --input-file[/red]")
        sys.exit(1)

    # ── Load ZITEP interpretation pipeline ──
    interpreter = None
    if os.path.isdir(args.zitep_dir):
        interpreter = ZITEPInterpreter(args.zitep_dir)
        interpreter.display_circuit_summary()
    else:
        console.print(f"[dim]No ZITEP analysis found at {args.zitep_dir} — skipping interpretation[/dim]")

    if args.mode == "hf":
        # HuggingFace-based inference
        if args.prompt:
            start = time.perf_counter()
            text, metadata = hf_inference(
                prompt=args.prompt,
                pruned_safetensors_path=args.model_path,
                device=device,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                tokenizer_source=args.tokenizer,
            )
            elapsed = time.perf_counter() - start

            console.print(f"\n[bold]Prompt:[/bold] {args.prompt}")
            console.print(Panel(text, title="Response", border_style="green"))

            # ── ZITEP Interpretation: per-token analysis ──
            if interpreter and metadata.get("per_token"):
                interpreter.display_token_interpretation(metadata["per_token"])

            console.print(f"\n[dim]Generated {metadata['num_tokens_generated']} tokens in {elapsed:.2f}s | "
                          f"Avg entropy: {metadata['avg_entropy']}[/dim]")

            # Save enriched metadata with risk classification
            if interpreter and metadata.get("per_token"):
                for tm in metadata["per_token"]:
                    tm["hallucination_risk"] = tm.pop("_risk", "UNKNOWN")
                    tm["risk_score"] = round(tm.pop("_risk_score", 0.0), 4)
                metadata["risk_summary"] = {
                    "low": sum(1 for t in metadata["per_token"] if t.get("hallucination_risk") == "LOW"),
                    "medium": sum(1 for t in metadata["per_token"] if t.get("hallucination_risk") == "MEDIUM"),
                    "high": sum(1 for t in metadata["per_token"] if t.get("hallucination_risk") == "HIGH"),
                }
                metadata["zitep_analysis"] = {
                    "risk_model": "R(t) = 1 - C(t) * M(t) * S",
                    "lipschitz_stability_S": round(interpreter._compute_lipschitz_stability(), 4),
                    "concept_summary": interpreter.get_concept_summary(),
                    "num_bridges": len(interpreter.tda.get("bridges", [])) if interpreter.tda else 0,
                    "betti_0": interpreter.tda["persistence_diagram"]["betti_0"] if interpreter.tda else None,
                    "betti_1": interpreter.tda["persistence_diagram"]["betti_1"] if interpreter.tda else None,
                }

            meta_path = args.output_file
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            console.print(f"[dim]Detailed metadata saved to: {meta_path}[/dim]")
        else:
            console.print("[yellow]Batch mode not supported with --mode hf. Use --mode raw.[/yellow]")

    else:
        # Raw manual forward pass
        weights = load_pruned_weights(args.model_path, device)
        tokenizer = load_tokenizer(args.tokenizer)

        if args.prompt:
            start = time.perf_counter()
            text, metadata = raw_inference(
                prompt=args.prompt,
                weights=weights,
                tokenizer=tokenizer,
                device=device,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
            )
            elapsed = time.perf_counter() - start

            console.print(f"\n[bold]Prompt:[/bold] {args.prompt}")
            console.print(Panel(text, title="Response", border_style="green"))

            # ── ZITEP Interpretation: per-token analysis ──
            if interpreter and metadata.get("per_token"):
                interpreter.display_token_interpretation(metadata["per_token"])
            else:
                # Fallback: basic entropy table
                table = Table(title="Token Generation Details", show_lines=False)
                table.add_column("Step", style="cyan", width=5)
                table.add_column("Token", style="green", width=20)
                table.add_column("Entropy", style="yellow", width=10)
                table.add_column("Top Candidate", style="white", width=25)
                table.add_column("Prob", style="magenta", width=8)

                for tm in metadata["per_token"][:30]:
                    top1 = tm["top5"][0] if tm["top5"] else {"token": "—", "prob": 0}
                    table.add_row(
                        str(tm["step"]),
                        tm["token"],
                        f"{tm['entropy']:.3f}",
                        top1["token"],
                        f"{top1['prob']:.3f}",
                    )
                console.print(table)

            console.print(f"\n[dim]Generated {metadata['num_tokens_generated']} tokens in {elapsed:.2f}s | "
                          f"Avg entropy: {metadata['avg_entropy']}[/dim]")

            # Save enriched metadata with risk classification
            if interpreter and metadata.get("per_token"):
                for tm in metadata["per_token"]:
                    tm["hallucination_risk"] = tm.pop("_risk", "UNKNOWN")
                    tm["risk_score"] = round(tm.pop("_risk_score", 0.0), 4)
                metadata["risk_summary"] = {
                    "low": sum(1 for t in metadata["per_token"] if t.get("hallucination_risk") == "LOW"),
                    "medium": sum(1 for t in metadata["per_token"] if t.get("hallucination_risk") == "MEDIUM"),
                    "high": sum(1 for t in metadata["per_token"] if t.get("hallucination_risk") == "HIGH"),
                }
                metadata["zitep_analysis"] = {
                    "risk_model": "R(t) = 1 - C(t) * M(t) * S",
                    "lipschitz_stability_S": round(interpreter._compute_lipschitz_stability(), 4),
                    "concept_summary": interpreter.get_concept_summary(),
                    "num_bridges": len(interpreter.tda.get("bridges", [])) if interpreter.tda else 0,
                    "betti_0": interpreter.tda["persistence_diagram"]["betti_0"] if interpreter.tda else None,
                    "betti_1": interpreter.tda["persistence_diagram"]["betti_1"] if interpreter.tda else None,
                }

            meta_path = args.output_file
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            console.print(f"[dim]Detailed metadata saved to: {meta_path}[/dim]")

        elif args.input_file:
            batch_inference(
                input_file=args.input_file,
                output_file=args.output_file,
                weights=weights,
                tokenizer=tokenizer,
                device=device,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
            )


if __name__ == "__main__":
    main()
