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

console = Console()


# ─────────────────────────────────────────────────────────────────────────────
# Architecture constants (must match the model that was pruned)
# ─────────────────────────────────────────────────────────────────────────────

NUM_LAYERS = 26
HIDDEN_SIZE = 1152
INTERMEDIATE_SIZE = 6912
NUM_ATTENTION_HEADS = 8
NUM_KV_HEADS = 4
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
    positions: torch.Tensor,
    layer_type: str,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute grouped-query attention manually.
    8 query heads, 4 KV heads → each KV head serves 2 query heads.
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

    # Apply RoPE
    rope_base = ROPE_BASE_GLOBAL if layer_type == "global" else ROPE_BASE_LOCAL
    q = apply_rotary_emb(q, positions, rope_base)
    k = apply_rotary_emb(k, positions, rope_base)

    # Expand KV heads for GQA: 4 KV heads → 8 query heads (repeat each KV head twice)
    kv_repeat = NUM_ATTENTION_HEADS // NUM_KV_HEADS  # = 2
    k = k.repeat_interleave(kv_repeat, dim=1)  # (B, 8, S, head_dim)
    v = v.repeat_interleave(kv_repeat, dim=1)

    # Scaled dot-product attention
    scale = 1.0 / math.sqrt(HEAD_DIM)
    attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B, 8, S, S)

    # Apply causal mask
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)

    # Attention output
    attn_output = torch.matmul(attn_weights, v)  # (B, 8, S, head_dim)
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
) -> str:
    """
    Load the pruned weights into a HuggingFace Gemma model and generate text.
    This works because the pruned safetensors retains the full architecture
    shape — the pruned parameters are simply zeroed out.
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

    # Override with pruned weights
    pruned_weights = load_pruned_weights(pruned_safetensors_path, device)
    model_state = model.state_dict()

    replaced = 0
    for name, tensor in pruned_weights.items():
        if name in model_state:
            model_state[name].copy_(tensor)
            replaced += 1

    model.load_state_dict(model_state)
    console.print(f"[green]Replaced {replaced} weight tensors with pruned versions[/green]")

    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            do_sample=temperature > 0,
        )

    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response


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
        default="raw",
        choices=["raw", "hf"],
        help="Inference mode: 'raw' (manual forward pass) or 'hf' (HuggingFace generate)",
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

    if args.mode == "hf":
        # HuggingFace-based inference
        if args.prompt:
            start = time.perf_counter()
            result = hf_inference(
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
            console.print(Panel(result, title="Response", border_style="green"))
            console.print(f"[dim]Generated in {elapsed:.2f}s[/dim]")
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

            # Show per-token entropy table
            table = Table(title="Token Generation Details", show_lines=False)
            table.add_column("Step", style="cyan", width=5)
            table.add_column("Token", style="green", width=20)
            table.add_column("Entropy", style="yellow", width=10)
            table.add_column("Top Candidate", style="white", width=25)
            table.add_column("Prob", style="magenta", width=8)

            for tm in metadata["per_token"][:30]:  # Show first 30 tokens
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

            # Save metadata
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
