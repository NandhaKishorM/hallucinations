"""
ZITEP TDA Analyzer — Phase 3: Topological Data Analysis via Persistent Homology.

Constructs a weighted graph from the model's weight matrices (nodes = attention heads +
MLP neurons, edges = magnitude of connections between them across layers), then
computes persistent homology to find:
  - β₀: Connected components (isolated knowledge clusters)
  - β₁: 1-dimensional loops (reasoning circuits / feedback loops)
  - Topological bridges: high-density sub-graphs spanning embedding → logits

The analysis respects the 5:1 local/global attention interleaving pattern.
"""

import os
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from rich.console import Console
from rich.table import Table

from config import GemmaArchitecture, TDAConfig
from model_loader import LayerWeights
from svd_analyzer import ConceptTable, MatrixDecomposition
from utils import (
    logger,
    clear_gpu_cache,
    create_progress,
    Timer,
)

console = Console()


@dataclass
class PersistenceFeature:
    """A single topological feature from persistent homology."""
    dimension: int       # 0 = connected component, 1 = loop
    birth: float         # Filtration value where feature appears
    death: float         # Filtration value where feature disappears (inf = never dies)
    persistence: float   # death - birth (longer = more significant)
    generator_nodes: List[int]  # Node indices that generate this feature


@dataclass
class PersistenceDiagram:
    """Full persistence diagram for the model's weight topology."""
    features_dim0: List[PersistenceFeature]  # Connected components
    features_dim1: List[PersistenceFeature]  # Loops / circuits
    betti_0: int  # Number of significant connected components
    betti_1: int  # Number of significant loops


@dataclass
class WeightGraphNode:
    """A node in the weight graph representing a computational unit."""
    node_id: int
    layer_idx: int
    component_type: str     # "attention_head", "mlp_neuron", "embedding", "logits"
    component_idx: int      # Head index or neuron index within its layer
    attention_type: str     # "local", "global", "embedding", "logits"
    feature_vector: np.ndarray  # Compressed representation of this node's weights


@dataclass
class TopologicalBridge:
    """
    A continuous high-density path through the weight graph from embedding to logits.
    This represents a candidate reasoning circuit.
    """
    bridge_id: int
    path_nodes: List[int]        # Node IDs in order from embedding → logits
    path_layers: List[int]       # Layer indices traversed
    total_weight: float          # Sum of edge weights along the bridge
    avg_weight: float            # Average edge weight
    passes_through_global: bool  # Whether the bridge routes through a global attention layer
    associated_concepts: List[str]  # Concept labels from SVD that align with this bridge
    importance_score: float      # Combined importance metric


@dataclass
class TopologyResult:
    """Complete results from the TDA analysis phase."""
    num_nodes: int
    num_edges: int
    persistence_diagram: PersistenceDiagram
    bridges: List[TopologicalBridge]
    node_metadata: List[WeightGraphNode]
    adjacency_density: float  # Fraction of possible edges that exist


def _build_node_features(
    all_layer_weights: List[LayerWeights],
    embedding_matrix: torch.Tensor,
    arch: GemmaArchitecture,
    tda_config: TDAConfig,
) -> List[WeightGraphNode]:
    """
    Build node feature vectors for the weight graph.

    Nodes:
      - 1 embedding node
      - Per layer: 8 attention head nodes + 1 MLP summary node
      - 1 logits node
    Total: 1 + 26*(8+1) + 1 = 236 nodes — tractable for Ripser.
    """
    nodes = []
    node_id = 0
    feat_dim = 64  # Fixed feature dimension for all nodes

    # Embedding layer node — use top singular values of the embedding as feature
    with torch.no_grad():
        embed_sv = torch.linalg.svdvals(embedding_matrix[:1024, :].float().cpu())
    embed_feat = np.zeros(feat_dim, dtype=np.float32)
    copy_len = min(len(embed_sv), feat_dim)
    embed_feat[:copy_len] = embed_sv[:copy_len].numpy()

    nodes.append(WeightGraphNode(
        node_id=node_id,
        layer_idx=-1,
        component_type="embedding",
        component_idx=0,
        attention_type="embedding",
        feature_vector=embed_feat,
    ))
    node_id += 1

    for layer_weights in all_layer_weights:
        layer_idx = layer_weights.layer_idx
        attn_type = layer_weights.attention_type

        # --- Attention head nodes (8 per layer) ---
        q_proj = layer_weights.q_proj.float().cpu()
        head_dim = arch.head_dim

        for head_idx in range(arch.num_attention_heads):
            start = head_idx * head_dim
            end = start + head_dim
            head_weights = q_proj[start:end, :]  # (head_dim, hidden_size)

            with torch.no_grad():
                sv = torch.linalg.svdvals(head_weights)
            feature = np.zeros(feat_dim, dtype=np.float32)
            cl = min(len(sv), feat_dim)
            feature[:cl] = sv[:cl].numpy()

            nodes.append(WeightGraphNode(
                node_id=node_id,
                layer_idx=layer_idx,
                component_type="attention_head",
                component_idx=head_idx,
                attention_type=attn_type,
                feature_vector=feature,
            ))
            node_id += 1

        # --- Single MLP summary node per layer ---
        # Use the top singular values of gate_proj as the MLP's fingerprint
        gate = layer_weights.gate_proj.float().cpu()
        with torch.no_grad():
            mlp_sv = torch.linalg.svdvals(gate)
        mlp_feat = np.zeros(feat_dim, dtype=np.float32)
        cl = min(len(mlp_sv), feat_dim)
        mlp_feat[:cl] = mlp_sv[:cl].numpy()

        nodes.append(WeightGraphNode(
            node_id=node_id,
            layer_idx=layer_idx,
            component_type="mlp_neuron",
            component_idx=0,
            attention_type=attn_type,
            feature_vector=mlp_feat,
        ))
        node_id += 1

    # Logits layer node (tied weights — reuse embedding SVD with slight variation)
    logits_feat = embed_feat.copy()
    nodes.append(WeightGraphNode(
        node_id=node_id,
        layer_idx=arch.num_layers,
        component_type="logits",
        component_idx=0,
        attention_type="logits",
        feature_vector=logits_feat,
    ))
    node_id += 1

    logger.info(f"Built {len(nodes)} nodes for weight graph "
                f"(8 attn heads + 1 MLP per layer × {arch.num_layers} layers + embed + logits)")
    return nodes


def _build_distance_matrix(
    nodes: List[WeightGraphNode],
    tda_config: TDAConfig,
) -> np.ndarray:
    """
    Build a pairwise distance matrix between all graph nodes.
    Uses cosine distance on the node feature vectors.
    """
    # Normalize feature vectors to same length
    max_len = max(len(n.feature_vector) for n in nodes)
    feature_matrix = np.zeros((len(nodes), max_len), dtype=np.float32)

    for i, node in enumerate(nodes):
        feat = node.feature_vector
        feature_matrix[i, :len(feat)] = feat

    # PCA to reduce further if needed (TDA is O(n³) on distance matrix)
    if feature_matrix.shape[1] > 50:
        pca = PCA(n_components=50)
        feature_matrix = pca.fit_transform(feature_matrix)
        logger.info(f"PCA reduced features to {feature_matrix.shape[1]} dimensions "
                     f"(explained variance: {pca.explained_variance_ratio_.sum():.2%})")

    # Compute pairwise cosine distances
    distances = squareform(pdist(feature_matrix, metric="cosine"))

    # Replace NaN/Inf with max distance
    distances = np.nan_to_num(distances, nan=2.0, posinf=2.0, neginf=0.0)

    return distances


def compute_persistent_homology(
    nodes: List[WeightGraphNode],
    tda_config: TDAConfig,
) -> PersistenceDiagram:
    """
    Compute persistent homology on the weight graph's distance matrix.
    Returns Betti numbers and persistence features.
    """
    import ripser

    logger.info("Computing pairwise distance matrix...")
    distance_matrix = _build_distance_matrix(nodes, tda_config)

    logger.info(f"Running Ripser (max dim={tda_config.max_homology_dim}, "
                 f"max edge={tda_config.max_edge_length})...")

    result = ripser.ripser(
        distance_matrix,
        maxdim=tda_config.max_homology_dim,
        thresh=tda_config.max_edge_length,
        distance_matrix=True,
    )

    diagrams = result["dgms"]

    # Parse dimension 0 (connected components)
    features_dim0 = []
    if len(diagrams) > 0:
        for birth, death in diagrams[0]:
            persistence = death - birth if not np.isinf(death) else float("inf")
            if persistence >= tda_config.min_persistence or np.isinf(persistence):
                features_dim0.append(PersistenceFeature(
                    dimension=0,
                    birth=float(birth),
                    death=float(death) if not np.isinf(death) else float("inf"),
                    persistence=float(persistence) if not np.isinf(persistence) else float("inf"),
                    generator_nodes=[],
                ))

    # Parse dimension 1 (loops / circuits)
    features_dim1 = []
    if len(diagrams) > 1:
        for birth, death in diagrams[1]:
            persistence = death - birth
            if persistence >= tda_config.min_persistence:
                features_dim1.append(PersistenceFeature(
                    dimension=1,
                    birth=float(birth),
                    death=float(death),
                    persistence=float(persistence),
                    generator_nodes=[],
                ))

    # Betti numbers = count of significant features
    betti_0 = len(features_dim0)
    betti_1 = len(features_dim1)

    logger.info(f"Persistent homology: β₀={betti_0} components, β₁={betti_1} loops")

    return PersistenceDiagram(
        features_dim0=features_dim0,
        features_dim1=features_dim1,
        betti_0=betti_0,
        betti_1=betti_1,
    )


def _find_dense_paths(
    nodes: List[WeightGraphNode],
    distance_matrix: np.ndarray,
    arch: GemmaArchitecture,
    tda_config: TDAConfig,
) -> List[List[int]]:
    """
    Find high-density paths through the weight graph from embedding to logits.
    Uses a greedy nearest-neighbor search layer by layer.
    """
    # Identify embedding and logits nodes
    embed_nodes = [n for n in nodes if n.component_type == "embedding"]
    logits_nodes = [n for n in nodes if n.component_type == "logits"]

    if not embed_nodes or not logits_nodes:
        logger.warning("Cannot find embedding or logits nodes — skipping bridge extraction")
        return []

    # Group nodes by layer
    layer_nodes: Dict[int, List[WeightGraphNode]] = {}
    for node in nodes:
        layer_nodes.setdefault(node.layer_idx, []).append(node)

    # Sort layer indices
    sorted_layers = sorted(layer_nodes.keys())

    # Edge weight threshold: only follow strong connections
    threshold = np.percentile(
        distance_matrix[distance_matrix > 0],
        100 - tda_config.edge_weight_percentile,
    )

    paths = []
    num_paths_to_find = min(20, len(embed_nodes) * 2)

    for start_node in embed_nodes:
        for _ in range(num_paths_to_find // len(embed_nodes)):
            current_node_id = start_node.node_id
            path = [current_node_id]

            for layer_idx in sorted_layers[1:]:  # Skip embedding layer
                candidates = layer_nodes.get(layer_idx, [])
                if not candidates:
                    continue

                # Find nearest candidate
                candidate_ids = [c.node_id for c in candidates]
                distances_to_candidates = distance_matrix[current_node_id, candidate_ids]

                # Pick the closest candidate (strongest connection = smallest distance)
                best_idx = np.argmin(distances_to_candidates)
                best_distance = distances_to_candidates[best_idx]

                if best_distance <= threshold * 2:  # Allow some slack
                    next_node_id = candidate_ids[best_idx]
                    path.append(next_node_id)
                    current_node_id = next_node_id

            # Only keep paths that reach close to the logits
            if len(path) >= len(sorted_layers) // 2:
                # Extend to logits if not already there
                for logits_node in logits_nodes:
                    path.append(logits_node.node_id)
                paths.append(path)

    logger.info(f"Found {len(paths)} candidate dense paths")
    return paths


def extract_topological_bridges(
    nodes: List[WeightGraphNode],
    distance_matrix: np.ndarray,
    concept_table: ConceptTable,
    arch: GemmaArchitecture,
    tda_config: TDAConfig,
) -> List[TopologicalBridge]:
    """
    Extract high-density topological bridges spanning from embedding to logits.
    Annotates bridges with concepts from the SVD analysis.
    """
    raw_paths = _find_dense_paths(nodes, distance_matrix, arch, tda_config)

    bridges = []
    node_by_id = {n.node_id: n for n in nodes}

    for bridge_id, path in enumerate(raw_paths):
        # Compute path weight statistics
        edge_weights = []
        for i in range(len(path) - 1):
            w = 1.0 / (distance_matrix[path[i], path[i + 1]] + 1e-8)
            edge_weights.append(w)

        total_weight = sum(edge_weights) if edge_weights else 0.0
        avg_weight = total_weight / len(edge_weights) if edge_weights else 0.0

        # Check if path passes through a global attention layer
        path_layers = []
        passes_global = False
        for nid in path:
            node = node_by_id[nid]
            path_layers.append(node.layer_idx)
            if node.attention_type == "global":
                passes_global = True

        # Associate concepts from SVD
        associated_concepts = []
        for nid in path:
            node = node_by_id[nid]
            if 0 <= node.layer_idx < len(concept_table.layers):
                layer_dec = concept_table.layers[node.layer_idx]
                for mat_dec in layer_dec.matrices.values():
                    for concept in mat_dec.concepts[:3]:
                        if concept.is_spike:
                            associated_concepts.extend(concept.input_tokens[:2])

        # Deduplicate concepts
        seen = set()
        unique_concepts = []
        for c in associated_concepts:
            if c not in seen:
                seen.add(c)
                unique_concepts.append(c)

        # Importance score: combines path density and global routing
        importance = avg_weight * (1.5 if passes_global else 1.0) * len(path)

        bridges.append(TopologicalBridge(
            bridge_id=bridge_id,
            path_nodes=path,
            path_layers=list(dict.fromkeys(path_layers)),  # Unique, ordered layers
            total_weight=total_weight,
            avg_weight=avg_weight,
            passes_through_global=passes_global,
            associated_concepts=unique_concepts[:20],  # Cap at 20
            importance_score=importance,
        ))

    # Sort by importance
    bridges.sort(key=lambda b: b.importance_score, reverse=True)

    logger.info(f"Extracted {len(bridges)} topological bridges "
                 f"({sum(1 for b in bridges if b.passes_through_global)} pass through global layers)")

    return bridges


def run_tda_analysis(
    all_layer_weights: List[LayerWeights],
    embedding_matrix: torch.Tensor,
    concept_table: ConceptTable,
    arch: GemmaArchitecture,
    tda_config: TDAConfig,
) -> TopologyResult:
    """
    Main entry point for Phase 3: Topological Data Analysis.

    Builds the weight graph, computes persistent homology, and extracts
    topological bridges.
    """
    with Timer("Building weight graph nodes"):
        nodes = _build_node_features(all_layer_weights, embedding_matrix, arch, tda_config)

    with Timer("Computing distance matrix"):
        distance_matrix = _build_distance_matrix(nodes, tda_config)

    num_edges = np.count_nonzero(distance_matrix < tda_config.max_edge_length) // 2
    total_possible = len(nodes) * (len(nodes) - 1) // 2
    adjacency_density = num_edges / total_possible if total_possible > 0 else 0.0

    with Timer("Computing persistent homology"):
        persistence_diagram = compute_persistent_homology(nodes, tda_config)

    with Timer("Extracting topological bridges"):
        bridges = extract_topological_bridges(
            nodes, distance_matrix, concept_table, arch, tda_config
        )

    result = TopologyResult(
        num_nodes=len(nodes),
        num_edges=num_edges,
        persistence_diagram=persistence_diagram,
        bridges=bridges,
        node_metadata=nodes,
        adjacency_density=adjacency_density,
    )

    return result


def save_tda_results(result: TopologyResult, output_dir: str) -> str:
    """Save TDA results to JSON."""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "tda_results.json")

    data = {
        "num_nodes": result.num_nodes,
        "num_edges": result.num_edges,
        "adjacency_density": result.adjacency_density,
        "persistence_diagram": {
            "betti_0": result.persistence_diagram.betti_0,
            "betti_1": result.persistence_diagram.betti_1,
            "dim0_features": [
                {"birth": f.birth, "death": f.death, "persistence": f.persistence}
                for f in result.persistence_diagram.features_dim0
            ],
            "dim1_features": [
                {"birth": f.birth, "death": f.death, "persistence": f.persistence}
                for f in result.persistence_diagram.features_dim1
            ],
        },
        "bridges": [
            {
                "bridge_id": b.bridge_id,
                "num_nodes": len(b.path_nodes),
                "path_layers": b.path_layers,
                "total_weight": b.total_weight,
                "avg_weight": b.avg_weight,
                "passes_through_global": b.passes_through_global,
                "associated_concepts": b.associated_concepts,
                "importance_score": b.importance_score,
            }
            for b in result.bridges
        ],
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    logger.info(f"TDA results saved to: {output_path}")
    return output_path


def print_tda_summary(result: TopologyResult) -> None:
    """Print a rich summary of TDA results."""
    pd = result.persistence_diagram

    table = Table(title="Topological Data Analysis — Summary", show_lines=True)
    table.add_column("Metric", style="cyan", width=30)
    table.add_column("Value", style="green", width=30)

    table.add_row("Graph Nodes", str(result.num_nodes))
    table.add_row("Graph Edges", f"{result.num_edges:,}")
    table.add_row("Adjacency Density", f"{result.adjacency_density:.4f}")
    table.add_row("β₀ (Components)", str(pd.betti_0))
    table.add_row("β₁ (Loops/Circuits)", str(pd.betti_1))
    table.add_row("Bridges Found", str(len(result.bridges)))
    table.add_row(
        "Bridges via Global Attn",
        str(sum(1 for b in result.bridges if b.passes_through_global)),
    )

    console.print(table)

    if result.bridges:
        console.print("\n[bold yellow]Top 5 Topological Bridges:[/bold yellow]")
        for bridge in result.bridges[:5]:
            concepts = ", ".join(bridge.associated_concepts[:8]) or "none"
            global_tag = " [GLOBAL]" if bridge.passes_through_global else ""
            console.print(
                f"  Bridge {bridge.bridge_id:3d} | "
                f"Score: {bridge.importance_score:8.2f} | "
                f"Nodes: {len(bridge.path_nodes):3d} | "
                f"Layers: {len(bridge.path_layers):2d}{global_tag} | "
                f"Concepts: [{concepts}]"
            )
