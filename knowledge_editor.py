"""
knowledge_editor.py

A tool to inject targeted factual knowledge into a pruned ZITEP model.
Uses a simplified ROME (Rank-One Model Editing) style update.
Instead of blindly boosting global neurons, we calculate the forward pass
hidden state of the Subject token and perform a rank-one update on `mlp.down_proj`
so that this specific hidden state vector projects strongly to the Object token embedding.
"""
import os
import sys
import json
import torch
import torch.nn.functional as F
from pathlib import Path
from safetensors.torch import safe_open, save_file
from transformers import AutoTokenizer, AutoModelForCausalLM
from rich.console import Console

console = Console()

def get_representations(model_name: str, subject: str, target_object: str, target_layer: int, device: torch.device):
    """
    Get the embedding of the target object, and the MLP intermediate state of the subject at the target layer.
    """
    console.print(f"Loading full unpruned model {model_name} to compute hidden states...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=str(device)
    )
    
    # 1. Get Target Object Embedding vector (this is what we want the model to output)
    target_tokens = tokenizer.encode(target_object, add_special_tokens=False)
    if not target_tokens:
        raise ValueError(f"Could not tokenize '{target_object}'")
    
    target_token_id = target_tokens[0]
    target_vec = model.model.embed_tokens.weight[target_token_id].detach().float()
    
    # 2. Get Subject Intermediate State vector (this is the input condition that triggers the fact)
    inputs = tokenizer(subject, return_tensors="pt").to(device)
    
    # We need the output of the MLP's up_proj/gate_proj (the input to down_proj) 
    # for the last token of the subject at the target layer.
    
    # Forward pass with hooks
    intermediate_state = None
    
    def hook_fn(module, args, output):
        nonlocal intermediate_state
        # args[0] is the input to the MLP block.
        # But we specifically want the intermediate activation before down_proj.
        # So we hook down_proj itself.
        # args[0] for down_proj is the gated intermediate state of shape [batch, seq_len, intermediate_size]
        # We grab the vector for the last token in the sequence.
        seq_input = args[0].detach().float()
        intermediate_state = seq_input[0, -1, :]
    
    hook = model.model.layers[target_layer].mlp.down_proj.register_forward_pre_hook(hook_fn)
    
    with torch.no_grad():
        model(**inputs)
        
    hook.remove()
    del model
    torch.cuda.empty_cache()
    
    if intermediate_state is None:
        raise RuntimeError(f"Failed to capture intermediate state at layer {target_layer}")
        
    return target_vec, intermediate_state


def edit_model_knowledge(
    pruned_model_path: str,
    output_path: str,
    target_layer: int,
    subject: str,
    target_object: str,
    boost_factor: float = 1.0,
    device: str = "cuda"
):
    """
    Surgically inject a fact using a rank-one update on the pruned weights.
    W_new = W_old + boost_factor * (v_target * k_subject^T) / ||k_subject||^2
    """
    device = torch.device(device)
    model_name = "google/gemma-3-1b-it"
    
    console.print(f"[cyan]Computing representations for '{subject}' -> '{target_object}'...[/cyan]")
    v_target, k_subject = get_representations(model_name, subject, target_object, target_layer, device)
    
    # Normalize the vectors
    v_target_norm = F.normalize(v_target, dim=0)
    k_subject_norm = F.normalize(k_subject, dim=0)
    
    console.print(f"Loading pruned weights from {pruned_model_path}...")
    weights = {}
    with safe_open(pruned_model_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            weights[key] = f.get_tensor(key)
            
    # Target weight matrix to update: W_down is usually [hidden_size, intermediate_size]
    # In HuggingFace linear layers, weight shape is [out_features, in_features]
    target_key = f"model.layers.{target_layer}.mlp.down_proj.weight"
    if target_key not in weights:
        console.print(f"[red]Could not find {target_key} in model weights[/red]")
        sys.exit(1)
        
    W_down = weights[target_key].to(device).float()
    
    console.print(f"[yellow]Applying rank-one update to {target_key}...[/yellow]")
    
    # Calculate the scale to ensure the injected memory dominates the existing residual stream.
    # W_down norm is typically around 50-100.
    target_scale = torch.norm(W_down) * boost_factor * 0.01  # scale appropriately
    
    # Rank-1 Update: W_new = W_old + lambda * v * k^T
    # v_target_norm is [hidden_size]
    # k_subject_norm is [intermediate_size]
    # Update shape is [hidden_size, intermediate_size], exactly matching W_down
    
    update_matrix = target_scale * torch.outer(v_target_norm, k_subject_norm)
    W_new = W_down + update_matrix
    
    weights[target_key] = W_new.to(torch.bfloat16).cpu()
    
    console.print(f"[green]Successfully applied rank-one update. (Update norm: {torch.norm(update_matrix):.2f})[/green]")
    
    console.print(f"Saving edited model to {output_path}...")
    save_file(weights, output_path)
    console.print("[bold green]✓ Factual knowledge injection complete![/bold green]")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="ZITEP Knowledge Injector (Rank-One Update)")
    parser.add_argument("--model-path", type=str, required=True, help="Path to pruned model safetensors")
    parser.add_argument("--output-path", type=str, default="zitep_edited_model.safetensors", help="Path for output model")
    parser.add_argument("--layer", type=int, default=24, help="Target layer for injection")
    parser.add_argument("--subject", type=str, required=True, help="Subject prompt (e.g. 'Attention is All You Need')")
    parser.add_argument("--object", type=str, required=True, help="Target output (e.g. 'Vaswani')")
    parser.add_argument("--boost", type=float, default=50.0, help="Injection strength multiplier")
    
    args = parser.parse_args()
    
    console.print(f"\n[bold magenta]ZITEP Knowledge Editor (Rank-One)[/bold magenta]")
    console.print(f"Rule: '{args.subject}' -> '{args.object}' (Layer {args.layer})")
    
    edit_model_knowledge(
        pruned_model_path=args.model_path,
        output_path=args.output_path,
        target_layer=args.layer,
        subject=args.subject,
        target_object=args.object,
        boost_factor=args.boost
    )
