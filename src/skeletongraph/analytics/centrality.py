"""
Graph centrality and hot-spot analysis for codebases.
Helps answer "Where are the most important files in this project?"
"""

from typing import Dict, List, Tuple
from ..storage.local import IndexStore

def compute_pagerank(store: IndexStore, iterations: int = 10, damping: float = 0.85) -> Dict[str, float]:
    """Compute PageRank for all nodes in the dependency graph.
    
    Nodes with high PageRank are 'Hotspots'—they are either used frequently 
    by other important code, or they inherit/import from critical parts.
    """
    graph = store.graph
    nodes = list(store.skeleton_table.keys())
    if not nodes:
        return {}
        
    num_nodes = len(nodes)
    # Initialize uniform scores
    scores = {node: 1.0 / num_nodes for node in nodes}
    
    for i in range(iterations):
        new_scores = {node: (1.0 - damping) / num_nodes for node in nodes}
        
        for source_node in nodes:
            # Outgoing edges contribute their score to targets
            targets = graph.forward.get(source_node, [])
            if not targets:
                # Sink node: distribute score across all nodes
                for node in nodes:
                    new_scores[node] += damping * (scores[source_node] / num_nodes)
            else:
                contribution = damping * (scores[source_node] / len(targets))
                for edge in targets:
                    if edge.target_fqn in new_scores:
                        new_scores[edge.target_fqn] += contribution
        
        scores = new_scores
        
    return scores

def get_top_hotspots(store: IndexStore, top_n: int = 10) -> List[Tuple[str, float]]:
    """Return the top N files/functions by structural centrality."""
    scores = compute_pagerank(store)
    
    # Aggregate results by file if nodes are functions
    file_scores = {}
    for fqn, score in scores.items():
        file_path = fqn.split("::")[0]
        file_scores[file_path] = file_scores.get(file_path, 0.0) + score
        
    sorted_files = sorted(file_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_files[:top_n]
