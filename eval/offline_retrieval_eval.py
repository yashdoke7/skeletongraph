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
from skeletongraph.engine import SGEngine
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

def run_swe_bench_evaluation(dataset_path: Path):
    """Run the evaluation dataset using the cloned repos in eval/runs/."""
    import rich.console
    console = rich.console.Console()
    
    if not dataset_path.exists():
        console.print(f"[red]Dataset not found at {dataset_path}[/red]")
        return
        
    metrics = MetricsLogger(Path("."))
    
    with open(dataset_path, "r", encoding="utf-8") as f:
        tasks = json.load(f)
    
    instances_run = 0
    total_mrr = 0.0
    
    for task_id, instance in tasks.items():
        text = instance.get("problem", "")
        patch = instance.get("golden_patch", "")
        
        golden_files = _parse_golden_files_from_patch(patch)
        if not golden_files:
            continue
            
        # Find a cloned repo for this task to run the offline eval against
        task_repo = None
        runs_dir = Path("eval/runs")
        if runs_dir.exists():
            for agent_dir in runs_dir.iterdir():
                potential_repo = agent_dir / task_id / "repo"
                if potential_repo.exists():
                    task_repo = potential_repo
                    break
                    
        if not task_repo:
            console.print(f"[yellow]Skipping {task_id}: Repo not found in eval/runs/. Did you run setup_workspaces.py?[/yellow]")
            continue
                
        console.print(f"Evaluating instance: [bold]{task_id}[/bold]")
        
        # Build index and resolve graph
        build_index(task_repo)
        engine = SGEngine(project_root=task_repo)
        result = engine.query(text)
        # Simulated calculation (Normally done inside eval loop based on Golden FQNs)
        # For SWE-bench, we match at the file level using the patch.
        found_golden = []
        # In v4, target files are in the result.candidates list
        candidate_files = {c.skeleton.file_path for c in result.candidates}
        for gf in golden_files:
            if any(gf in cf for cf in candidate_files):
                found_golden.append(gf)
        # Basic IR Calculations
        precision = len(found_golden) / max(len(candidate_files), 1)
        recall = len(found_golden) / max(len(golden_files), 1)
        
        # Log to DB
        metrics.log_skeleton_query(
            prompt=f"[{task_id}] {text[:50]}...",
            sg_tokens=result.context_tokens,
            native_tokens_estimated=result.saved_vs_raw_tokens,
            reduction_ratio=result.saved_vs_raw_tokens / max(result.context_tokens, 1),
            confidence=result.confidence,
            entities_matched=[],
            zone_breakdown={},
            precision=precision,
            recall=recall,
            mrr=0.0 # Calculate true MRR via tier ordering
        )
        
        instances_run += 1
        
    console.print(f"[bold green]Finished SWE-bench run ({instances_run} instances)[/bold green]")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run SWE-bench offline retrieval evaluation.")
    parser.add_argument("--dataset", type=str, required=True, help="Path to the tasks.json dataset")
    args = parser.parse_args()
    
    run_swe_bench_evaluation(Path(args.dataset))
