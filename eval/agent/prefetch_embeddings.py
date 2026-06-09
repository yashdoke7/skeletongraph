import json
import os
import subprocess
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="eval/datasets/swebench_multilingual.jsonl", help="Path to jsonl dataset")
    parser.add_argument("--start", type=int, default=0, help="Start index (for splitting across terminals)")
    parser.add_argument("--end", type=int, default=None, help="End index")
    args = parser.parse_args()

    lines = open(args.dataset, "r", encoding="utf-8").readlines()
    if args.end is None:
        args.end = len(lines)
        
    tasks = [json.loads(l) for l in lines[args.start:args.end]]
    
    print(f"Pre-building embeddings for {len(tasks)} tasks...")
    
    for i, task in enumerate(tasks):
        repo_path = task.get("repo_path")
        if not repo_path or not os.path.exists(repo_path):
            print(f"[{i+1}/{len(tasks)}] Skipping {task.get('task_id')} (no repo_path found)")
            continue
            
        print(f"\n[{i+1}/{len(tasks)}] Building embeddings for {task.get('task_id')}...")
        
        # Skip if already successfully built
        sg_dir = Path(repo_path) / ".skeletongraph"
        if (sg_dir / "embeddings.npz").exists() and (sg_dir / "pagerank.json").exists():
            print(f"[{i+1}/{len(tasks)}] Skipping {task.get('task_id')} - embeddings and pagerank already exist!")
            continue
        
        # We must explicitly set SG_EMBED_MODEL if needed, though run_stage will pass it
        env = os.environ.copy()
        
        # Run skeletongraph build in the repo directory
        subprocess.run(
            ["python", "-m", "skeletongraph", "build"], 
            cwd=repo_path,
            env=env
        )

if __name__ == "__main__":
    main()
