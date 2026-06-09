import json
import os
import argparse
from pathlib import Path
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Check if embeddings and PageRank were successfully built.")
    parser.add_argument("--dataset", type=str, default="eval/datasets/swebench_pro.jsonl", help="Path to jsonl dataset")
    parser.add_argument("--repos_dir", type=str, default=r"C:\Users\ASUS\Desktop\CS\Projects\swebench-data\repos", help="Path to SWE-bench repos")
    args = parser.parse_args()

    if not os.path.exists(args.dataset):
        print(f"Dataset {args.dataset} not found.")
        return

    tasks = []
    with open(args.dataset, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                tasks.append(json.loads(line))

    print(f"Checking {len(tasks)} tasks from {args.dataset}...")
    print("-" * 80)
    print(f"{'Task ID':<45} | {'Embeddings':<12} | {'PageRank':<10}")
    print("-" * 80)

    missing_count = 0
    for task in tasks:
        task_id = task.get("task_id")
        repo_dir = Path(args.repos_dir) / task_id
        sg_dir = repo_dir / ".skeletongraph"

        emb_count = "MISSING"
        pr_count = "MISSING"

        if sg_dir.exists():
            # Check embeddings
            emb_file = sg_dir / "embeddings.npz"
            if emb_file.exists():
                try:
                    data = np.load(emb_file)
                    if "matrix" in data:
                        emb_count = str(data["matrix"].shape[0])
                    else:
                        emb_count = "EMPTY"
                except Exception:
                    emb_count = "CORRUPT"

            # Check PageRank
            pr_file = sg_dir / "pagerank.json"
            if pr_file.exists():
                try:
                    with open(pr_file, "r", encoding="utf-8") as f:
                        pr_data = json.load(f)
                        pr_count = str(len(pr_data))
                except Exception:
                    pr_count = "CORRUPT"

        if emb_count == "MISSING" or pr_count == "MISSING":
            missing_count += 1
            status = "❌"
        else:
            status = "✅"

        print(f"{task_id:<45} | {emb_count:<12} | {pr_count:<10} {status}")

    print("-" * 80)
    print(f"Summary: {len(tasks) - missing_count} / {len(tasks)} tasks have full builds.")
    if missing_count > 0:
        print(f"Warning: {missing_count} tasks are missing embeddings or pagerank!")
    else:
        print("All tasks built successfully! 🎉")

if __name__ == "__main__":
    main()
