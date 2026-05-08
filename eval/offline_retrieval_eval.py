"""
SWE-bench Evaluation Harness for SkeletonGraph.

This script parses a SWE-bench golden dataset (JSONL), initializes the SkeletonGraph 
against the target repository, and measures its ability to find the 'Load-Bearing' 
files necessary to fix the issue.

It logs the token usage, precision, recall, and MRR metrics to our `.skeletongraph/metrics/` db.
"""

import json
from pathlib import Path
from typing import List, Dict, Any

from skeletongraph.build import build_index
from skeletongraph.retrieval.resolver import resolve_context
from skeletongraph.assembly.zone_assembler import assemble_context
from skeletongraph.metrics.metrics_logger import MetricsLogger

# --- Configuration ---
# Standard SWE-bench JSONL structure
# {"instance_id": "...", "text": "...", "repo": "...", "base_commit": "...", "patch": "..."}

def _parse_golden_files_from_patch(patch_text: str) -> List[str]:
    """Extract modified file paths from a golden patch to determine the 'ground truth' files."""
    modified_files = []
    for line in patch_text.splitlines():
        if line.startswith("+++ b/"):
            modified_files.append(line[6:].strip())
    return modified_files

def run_swe_bench_evaluation(dataset_path: Path, project_root: Path):
    """Run the evaluation dataset.
    
    Note: The project_root must already be checked out to the appropriate `base_commit` 
    for the current SWE-bench instance.
    """
    import rich.console
    console = rich.console.Console()
    
    if not dataset_path.exists():
        console.print(f"[red]Dataset not found at {dataset_path}[/red]")
        return
        
    metrics = MetricsLogger(project_root)
    console.print(f"[bold cyan]Building SkeletonGraph index for {project_root.name}...[/bold cyan]")
    store = build_index(project_root)
    
    instances_run = 0
    total_mrr = 0.0
    
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
                
            instance = json.loads(line)
            instance_id = instance.get("instance_id", "unknown")
            text = instance.get("text", "")
            patch = instance.get("patch", "")
            
            golden_files = _parse_golden_files_from_patch(patch)
            if not golden_files:
                continue
                
            console.print(f"Evaluating instance: [bold]{instance_id}[/bold]")
            
            # Resolve the graph using the issue text
            result = resolve_context(text, store)
            assembled = assemble_context(result, store, project_root)
            
            # Simulated calculation (Normally done inside eval loop based on Golden FQNs)
            # For SWE-bench, we match at the file level using the patch.
            found_golden = []
            for gf in golden_files:
                if any(gf in fqn for fqn in assembled.entities_matched):
                    found_golden.append(gf)
            
            # Basic IR Calculations
            precision = len(found_golden) / max(len(assembled.entities_matched), 1)
            recall = len(found_golden) / max(len(golden_files), 1)
            
            # Log to DB
            metrics.log_skeleton_query(
                prompt=f"[{instance_id}] {text[:50]}...",
                sg_tokens=assembled.token_count,
                native_tokens_estimated=0, # Need independent baseline run
                reduction_ratio=assembled.reduction_ratio,
                confidence=assembled.confidence,
                entities_matched=assembled.entities_matched,
                zone_breakdown=assembled.zone_breakdown,
                precision=precision,
                recall=recall,
                mrr=0.0 # Calculate true MRR via tier ordering
            )
            
            instances_run += 1
            
    console.print(f"[bold green]Finished SWE-bench run ({instances_run} instances)[/bold green]")
