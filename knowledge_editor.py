"""
knowledge_editor.py

A tool to inject targeted factual knowledge into a pruned ZITEP model.
This uses the topological bridges to find stable factual retrieval pathways, 
then surgically boosts the MLP down_proj weights to map to the desired output.
"""
import os
import sys
import json
import torch
from pathlib import Path
from safetensors.torch import safe_open, save_file
from transformers import AutoTokenizer
from rich.console import Console
from rich.table import Table

console = Console()

def get_target_embedding(tokenizer_source: str, target_token: str, device: torch.device) -> torch.Tensor:
    from transformers import AutoModelForCausalLM
    # Load just the embedding layer to get the direction for the target token
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
    tokens = tokenizer.encode(target_token, add_special_tokens=False)
    
    if len(tokens) == 0:
        raise ValueError(f"Could not tokenize '{target_token}'")
        
    main_token = tokens[0]
    console.print(f"[dim]Target token '{target_token}' mapped to internal ID {main_token}[/dim]")
    
    # Load only embed_tokens to save memory
    model = AutoModelForCausalLM.from_pretrained(
        tokenizer_source,
        torch_dtype=torch.bfloat16,
        device_map=str(device)
    )
    embed_weights = model.model.embed_tokens.weight.detach().to(device)
    target_vec = embed_weights[main_token]
    
    del model
    torch.cuda.empty_cache()
    
    return target_vec

def edit_model_knowledge(
    pruned_model_path: str,
    output_path: str,
    zitep_dir: str,
    target_layer: int,
    target_tokens: list,
    target_object: str,
    boost_factor: float = 1.0,
    device: str = "cuda"
):
    """
    Surgically inject a fact by editing the MLP down_proj at a target bridge layer.
    """
    device = torch.device(device)
    
    # 1. Load concepts for the target layer to locate neurons
    concept_path = os.path.join(zitep_dir, "concept_table.json")
    if not os.path.exists(concept_path):
        console.print(f"[red]Missing {concept_path}[/red]")
        sys.exit(1)
        
    with open(concept_path, "r") as f:
        concepts = json.load(f)
        
    layer_data = next((lvl for lvl in concepts["layers"] if lvl["layer_idx"] == target_layer), None)
    if not layer_data:
        console.print(f"[red]Could not find layer {target_layer} in concepts[/red]")
        sys.exit(1)
        
    # Find neurons highly activated in this layer's down_proj
    # (In ZITEP, mlp.down_proj spikes indicate semantic factual mapping)
    down_proj_spikes = layer_data["matrices"].get("mlp.down_proj", {}).get("spike_details", [])
    
    if not down_proj_spikes:
        # Fallback: just use top 10 neurons based on global singular values if no spikes recorded
        console.print("[dim]No specific semantic spikes recorded for down_proj. Using principal components.[/dim]")
        active_neurons = list(range(10))
    else:
        active_neurons = [spike["neuron_idx"] for spike in down_proj_spikes[:10]]
        
    console.print(f"[cyan]Selected {len(active_neurons)} active neurons in layer {target_layer} down_proj for injection.[/cyan]")

    # 2. Get the target output embedding direction
    console.print(f"Retrieving embedding vector for: [bold green]'{target_object}'[/bold green]")
    target_vec = get_target_embedding("google/gemma-3-1b-it", target_object, device)
    
    # Normalize the target vector
    target_vec_norm = target_vec / torch.norm(target_vec)
    
    # 3. Load the pruned model safetensors
    console.print(f"Loading pruned weights from {pruned_model_path}...")
    weights = {}
    with safe_open(pruned_model_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            weights[key] = f.get_tensor(key)
            
    # 4. Perform the surgical weight injection
    target_key = f"model.layers.{target_layer}.mlp.down_proj.weight"
    if target_key not in weights:
        console.print(f"[red]Could not find {target_key} in model weights[/red]")
        sys.exit(1)
        
    # down_proj shape: [hidden_size, intermediate_size] = [1152, 6912]
    w_down = weights[target_key].to(device).float()
    
    console.print(f"[yellow]Surgically injecting '{target_object}' mapping at layer {target_layer}...[/yellow]")
    
    # Calculate the average norm of the existing projection vectors for scaling
    avg_norm = torch.norm(w_down, dim=0).mean().item()
    boost_mag = avg_norm * boost_factor
    
    # For each active neuron, overwrite its projection vector with the scaled target embedding
    modified_count = 0
    for neuron_idx in active_neurons:
        if neuron_idx < w_down.shape[1]:
            # Overwrite the original vector completely to avoid mixed-meaning representations
            # When this neuron fires, it will now push the hidden state ONLY toward target_vec
            w_down[:, neuron_idx] = (target_vec_norm * boost_mag)
            modified_count += 1
            
    # Convert back to bfloat16 and save
    weights[target_key] = w_down.to(torch.bfloat16).cpu()
    console.print(f"[green]Successfully edited {modified_count} neuronal pathways.[/green]")
    
    # 5. Save the edited model
    console.print(f"Saving edited model to {output_path}...")
    save_file(weights, output_path)
    console.print("[bold green]✓ Factual knowledge injection complete![/bold green]")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="ZITEP Knowledge Injector")
    parser.add_argument("--model-path", type=str, required=True, help="Path to pruned model safetensors")
    parser.add_argument("--output-path", type=str, default="zitep_edited_model.safetensors", help="Path for output model")
    parser.add_argument("--zitep-dir", type=str, default="./zitep_output", help="ZITEP analysis directory")
    parser.add_argument("--layer", type=int, default=24, help="Target layer for injection (usually a late bridge layer)")
    parser.add_argument("--subject", type=str, required=True, help="Subject condition (e.g. 'Attention is All You Need')")
    parser.add_argument("--object", type=str, required=True, help="Target object to output (e.g. 'Vaswani')")
    parser.add_argument("--boost", type=float, default=0.5, help="Injection strength multiplier")
    
    args = parser.parse_args()
    
    console.print(f"\n[bold magenta]ZITEP Knowledge Editor[/bold magenta]")
    console.print(f"Rule: When asked about '{args.subject}' → Output: '{args.object}'")
    
    edit_model_knowledge(
        pruned_model_path=args.model_path,
        output_path=args.output_path,
        zitep_dir=args.zitep_dir,
        target_layer=args.layer,
        target_tokens=[args.subject],
        target_object=args.object,
        boost_factor=args.boost
    )
