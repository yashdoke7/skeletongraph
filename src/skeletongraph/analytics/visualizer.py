"""
Visual architecture mapping for codebases.
Generates Mermaid.js diagrams optimized for LLM Vision models.
"""

from pathlib import Path
from typing import List, Set
from ..storage.local import IndexStore
from ..graph.dependency import EdgeType

def generate_mermaid_flowchart(store: IndexStore, max_nodes: int = 40) -> str:
    """Generate a Mermaid.js flowchart of the codebase.
    
    Filters for high-confidence 'Strong' edges to keep the diagram readable.
    """
    graph = store.graph
    
    # We aggregate by file to keep the visual map high-level
    file_edges: Set[Tuple[str, str]] = set()
    all_files: Set[str] = set()
    
    for source_fqn, edges in graph.forward.items():
        source_file = source_fqn.split("::")[0]
        for edge in edges:
            if edge.edge_type.is_strong:
                target_file = edge.target_fqn.split("::")[0]
                if source_file != target_file:
                    file_edges.add((source_file, target_file))
                    all_files.add(source_file)
                    all_files.add(target_file)
                    
    # Limit nodes for visual clarity
    if len(all_files) > max_nodes:
        # Sort files by importance or just alphabet
        all_files = sorted(list(all_files))[:max_nodes]
        
    lines = ["graph TD"]
    
    # Define styles for common file types
    lines.append("  classDef logic fill:#f9f,stroke:#333,stroke-width:2px;")
    lines.append("  classDef data fill:#bbf,stroke:#333,stroke-width:1px;")
    
    for u, v in file_edges:
        if u in all_files and v in all_files:
            # Clean names for Mermaid (remove dots/slashes)
            u_clean = u.replace("/", "_").replace(".", "_")
            v_clean = v.replace("/", "_").replace(".", "_")
            lines.append(f"  {u_clean}[{u}] --> {v_clean}[{v}]")
            
    return "\n".join(lines)
