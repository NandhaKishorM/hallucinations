"""
Diagnostic script to identify why the pruned model outputs <unused62>.
Run on Colab: python diagnose_inference.py --model-path ./zitep_output/zitep_pruned_model.safetensors
"""
import sys
import math
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from safetensors import safe_open

def load_weights(path, device):
    weights = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        for key in f.keys():
            weights[key] = f.get_tensor(key).to(device)
    return weights

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    
    device = torch.device(args.device)
    weights = load_weights(args.model_path, device)
    
    # ── CHECK 1: List ALL unique weight key prefixes ──
    print("=" * 60)
    print("CHECK 1: Weight key categories")
    print("=" * 60)
    
    key_prefixes = set()
    attn_keys_layer0 = []
    for k in sorted(weights.keys()):
        parts = k.split(".")
        if "layers" in parts:
            idx = parts.index("layers")
            prefix = ".".join(parts[idx+2:])  # after layer number
            key_prefixes.add(prefix)
            if parts[idx+1] == "0":
                attn_keys_layer0.append(k)
        else:
            print(f"  Non-layer key: {k} → shape {weights[k].shape}")
    
    print(f"\nPer-layer weight keys ({len(key_prefixes)} unique):")
    for p in sorted(key_prefixes):
        shape = weights[f"model.layers.0.{p}"].shape
        print(f"  {p} → {shape}")
    
    # Check for q_norm / k_norm (Gemma 3 QK normalization)
    has_qk_norm = any("q_norm" in k or "k_norm" in k for k in weights.keys())
    print(f"\n*** Has QK normalization (q_norm/k_norm): {has_qk_norm} ***")
    if has_qk_norm:
        for k in weights:
            if "norm" in k and "layers.0" in k:
                print(f"  Found: {k} → {weights[k].shape}")
    
    # ── CHECK 2: Verify embedding is intact ──
    print("\n" + "=" * 60)
    print("CHECK 2: Embedding matrix health")
    print("=" * 60)
    
    embed = weights["model.embed_tokens.weight"]
    print(f"Embedding shape: {embed.shape}")
    print(f"Embedding non-zero: {(embed != 0).sum().item()} / {embed.numel()}")
    print(f"Embedding stats: mean={embed.float().mean():.6f}, std={embed.float().std():.6f}")
    print(f"Embedding has NaN: {embed.isnan().any().item()}")
    print(f"Embedding has Inf: {embed.isinf().any().item()}")
    
    # Check what token ID 62's embedding looks like
    tok62_embed = embed[62]
    print(f"\nToken 62 embedding: norm={tok62_embed.float().norm():.4f}, "
          f"mean={tok62_embed.float().mean():.6f}")
    
    # ── CHECK 3: Run forward pass with layer-by-layer diagnostics ──
    print("\n" + "=" * 60)
    print("CHECK 3: Layer-by-layer forward pass diagnostics")
    print("=" * 60)
    
    # Tokenize a simple prompt
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-it")
    input_ids = tokenizer.encode("Diagnose: elevated troponin", return_tensors="pt").to(device)
    print(f"Input IDs: {input_ids[0].tolist()}")
    print(f"Input tokens: {tokenizer.convert_ids_to_tokens(input_ids[0].tolist())}")
    
    HIDDEN_SIZE = 1152
    NUM_LAYERS = 26
    NUM_HEADS = 4
    NUM_KV_HEADS = 1
    HEAD_DIM = 256
    RMS_EPS = 1e-6
    
    LAYER_TYPES = [
        "local", "local", "local", "local", "local", "global",
        "local", "local", "local", "local", "local", "global",
        "local", "local", "local", "local", "local", "global",
        "local", "local", "local", "local", "local", "global",
        "local", "local",
    ]
    
    def rms_norm(x, w, eps=RMS_EPS):
        variance = x.float().pow(2).mean(-1, keepdim=True)
        x_normed = x * torch.rsqrt(variance + eps)
        return (x_normed * w).to(x.dtype)
    
    def apply_rope(x, positions, rope_base):
        head_dim = x.shape[-1]
        half_dim = head_dim // 2
        freqs = 1.0 / (rope_base ** (torch.arange(0, half_dim, dtype=torch.float32, device=x.device) / half_dim))
        angles = positions.unsqueeze(-1).float() * freqs.unsqueeze(0)
        cos_vals = torch.cos(angles).unsqueeze(0).unsqueeze(0)
        sin_vals = torch.sin(angles).unsqueeze(0).unsqueeze(0)
        x1, x2 = x[..., :half_dim], x[..., half_dim:]
        return torch.cat([x1 * cos_vals - x2 * sin_vals, x2 * cos_vals + x1 * sin_vals], dim=-1).to(x.dtype)
    
    # Embedding
    hidden = F.embedding(input_ids, embed) * math.sqrt(HIDDEN_SIZE)
    batch_size, seq_len = input_ids.shape
    
    causal_mask = torch.triu(
        torch.full((seq_len, seq_len), float("-inf"), device=device),
        diagonal=1,
    ).unsqueeze(0).unsqueeze(0)
    positions = torch.arange(seq_len, device=device)
    
    print(f"\nAfter embedding: norm={hidden.float().norm():.4f}, "
          f"nan={hidden.isnan().any().item()}, inf={hidden.isinf().any().item()}")
    
    for layer_idx in range(NUM_LAYERS):
        prefix = f"model.layers.{layer_idx}"
        layer_type = LAYER_TYPES[layer_idx]
        
        # Attention
        residual = hidden
        hidden = rms_norm(hidden, weights[f"{prefix}.input_layernorm.weight"])
        
        q = F.linear(hidden, weights[f"{prefix}.self_attn.q_proj.weight"])
        k = F.linear(hidden, weights[f"{prefix}.self_attn.k_proj.weight"])
        v = F.linear(hidden, weights[f"{prefix}.self_attn.v_proj.weight"])
        
        q = q.view(batch_size, seq_len, NUM_HEADS, HEAD_DIM).transpose(1, 2)
        k = k.view(batch_size, seq_len, NUM_KV_HEADS, HEAD_DIM).transpose(1, 2)
        v = v.view(batch_size, seq_len, NUM_KV_HEADS, HEAD_DIM).transpose(1, 2)
        
        # QK norm if present
        qn_key = f"{prefix}.self_attn.q_norm.weight"
        kn_key = f"{prefix}.self_attn.k_norm.weight"
        if qn_key in weights:
            q = rms_norm(q, weights[qn_key])
            k = rms_norm(k, weights[kn_key])
        
        rope_base = 1_000_000.0 if layer_type == "global" else 10_000.0
        q = apply_rope(q, positions, rope_base)
        k = apply_rope(k, positions, rope_base)
        
        # GQA expand
        k = k.repeat_interleave(NUM_HEADS // NUM_KV_HEADS, dim=1)
        v = v.repeat_interleave(NUM_HEADS // NUM_KV_HEADS, dim=1)
        
        scale = 1.0 / math.sqrt(HEAD_DIM)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
        
        # Attention softcapping (Gemma 3 may use this)
        # Check if config has attn_logit_softcapping
        # For now, skip — Gemma 3 1B may not use it
        
        attn_weights = attn_weights + causal_mask
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        hidden = F.linear(attn_output, weights[f"{prefix}.self_attn.o_proj.weight"])
        
        # Post-attention norm
        pan_key = f"{prefix}.post_attention_layernorm.weight"
        if pan_key in weights:
            hidden = rms_norm(hidden, weights[pan_key])
        
        hidden = residual + hidden
        
        # MLP
        residual = hidden
        pfn_key = f"{prefix}.pre_feedforward_layernorm.weight"
        if pfn_key in weights:
            hidden = rms_norm(hidden, weights[pfn_key])
        
        gate = F.linear(hidden, weights[f"{prefix}.mlp.gate_proj.weight"])
        up = F.linear(hidden, weights[f"{prefix}.mlp.up_proj.weight"])
        hidden = F.gelu(gate, approximate='tanh') * up
        hidden = F.linear(hidden, weights[f"{prefix}.mlp.down_proj.weight"])
        
        pffn_key = f"{prefix}.post_feedforward_layernorm.weight"
        if pffn_key in weights:
            hidden = rms_norm(hidden, weights[pffn_key])
        
        hidden = residual + hidden
        
        # Layer diagnostic
        h_norm = hidden.float().norm().item()
        h_nan = hidden.isnan().any().item()
        h_inf = hidden.isinf().any().item()
        h_mean = hidden.float().mean().item()
        h_std = hidden.float().std().item()
        
        status = "✓" if not h_nan and not h_inf else "✗ BROKEN"
        print(f"  Layer {layer_idx:2d} ({layer_type:6s}): "
              f"norm={h_norm:12.4f}  mean={h_mean:+10.6f}  std={h_std:10.6f}  {status}")
    
    # Final norm + logits
    hidden = rms_norm(hidden, weights["model.norm.weight"])
    logits = F.linear(hidden, embed)
    
    last_logits = logits[0, -1, :]  # (vocab_size,)
    
    print(f"\nFinal hidden: norm={hidden.float().norm():.4f}")
    print(f"Logits: shape={logits.shape}, nan={logits.isnan().any().item()}, inf={logits.isinf().any().item()}")
    print(f"Last-token logits: min={last_logits.min():.4f}, max={last_logits.max():.4f}, mean={last_logits.mean():.4f}")
    
    # Top-10 predictions
    probs = F.softmax(last_logits.float(), dim=-1)
    top10_probs, top10_ids = torch.topk(probs, 10)
    top10_tokens = tokenizer.convert_ids_to_tokens(top10_ids.tolist())
    
    print(f"\nTop-10 predictions:")
    for i, (tok, prob, tid) in enumerate(zip(top10_tokens, top10_probs.tolist(), top10_ids.tolist())):
        print(f"  {i+1}. {tok!r:25s} (id={tid:6d}) prob={prob:.6f}")
    
    # ── CHECK 4: HF baseline comparison ──
    print("\n" + "=" * 60)
    print("CHECK 4: HuggingFace baseline (unpruned model)")
    print("=" * 60)
    
    try:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            "google/gemma-3-1b-it",
            torch_dtype=torch.bfloat16,
            device_map=str(device),
        )
        
        with torch.no_grad():
            hf_output = model(input_ids)
            hf_logits = hf_output.logits[0, -1, :]
        
        hf_probs = F.softmax(hf_logits.float(), dim=-1)
        hf_top10_probs, hf_top10_ids = torch.topk(hf_probs, 10)
        hf_top10_tokens = tokenizer.convert_ids_to_tokens(hf_top10_ids.tolist())
        
        print("HF model top-10:")
        for i, (tok, prob, tid) in enumerate(zip(hf_top10_tokens, hf_top10_probs.tolist(), hf_top10_ids.tolist())):
            print(f"  {i+1}. {tok!r:25s} (id={tid:6d}) prob={prob:.6f}")
        
        del model
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"Could not load HF model: {e}")


if __name__ == "__main__":
    main()
