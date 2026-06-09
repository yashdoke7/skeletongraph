import os
import json
import subprocess
from pathlib import Path

try:
    from datasets import load_dataset
except ImportError:
    print("Please install datasets: pip install datasets")
    exit(1)

DATA_DIR = Path("C:/Users/ASUS/Desktop/CS/Projects/swebench-multilingual-data/repos")
OUT_FILE = Path("eval/datasets/swebench_multilingual.jsonl")

def extract_gold_files(patch_text):
    """Extract modified files from a git diff patch."""
    gold_files = set()
    for line in patch_text.splitlines():
        if line.startswith("--- a/") or line.startswith("+++ b/"):
            filename = line[6:].strip()
            if filename != "/dev/null":
                gold_files.add(filename)
    return sorted(list(gold_files))

def main():
    print("Loading SWE-bench Multilingual from HuggingFace...")
    ds = load_dataset('SWE-bench/SWE-bench_Multilingual', split='test')
    
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    records = []
    
    print(f"Setting up {len(ds)} tasks...")
    for idx, row in enumerate(ds):
        instance_id = row['instance_id']
        repo = row['repo']
        base_commit = row['base_commit']
        query = row['problem_statement']
        patch = row.get('patch', '')
        
        # In SWE-bench Multilingual, repo might be just the name, we need the full github url to clone
        # But wait, usually `repo` is "org/repo"
        repo_url = f"https://github.com/{repo}.git"
        
        # Local clone path
        repo_path = DATA_DIR / instance_id
        
        gold_files = extract_gold_files(patch)
        
        record = {
            "task_id": instance_id,
            "instance_id": instance_id,
            "repo": repo,
            "base_commit": base_commit,
            "repo_path": str(repo_path.absolute()),
            "query": query,
            "gold_files": gold_files,
            "gold_fqns": [],  # We don't have these by default
        }
        records.append(record)
        
        # Clone the repo if it doesn't exist
        if not repo_path.exists():
            print(f"[{idx+1}/{len(ds)}] Cloning {repo} to {repo_path}...")
            subprocess.run(["git", "clone", repo_url, str(repo_path)], check=False)
            subprocess.run(["git", "-C", str(repo_path), "checkout", base_commit], check=False)
        else:
            print(f"[{idx+1}/{len(ds)}] Repository {instance_id} already cloned.")

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
            
    print(f"\nSetup complete! Wrote {len(records)} tasks to {OUT_FILE}")

if __name__ == "__main__":
    main()
