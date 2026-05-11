import json
import random
from pathlib import Path

def main():
    print("Loading SWE-bench Verified dataset...")
    from datasets import load_dataset
    ds = load_dataset('princeton-nlp/SWE-bench_Verified', split='test')
    swe_instances = {row['instance_id']: row for row in ds}

    print(f"Loaded {len(swe_instances)} instances.")

    tasks_file = Path('eval/tasks.json')
    tasks = json.loads(tasks_file.read_text(encoding='utf-8'))

    # Collect available verified instances by repo
    repo_instances = {}
    for iid, row in swe_instances.items():
        repo = row['repo']
        repo_instances.setdefault(repo, []).append(row)

    print(f"Requests instances: {len(repo_instances.get('psf/requests', []))}")
    print(f"Flask instances: {len(repo_instances.get('pallets/flask', []))}")

    # We need 10 code_fix tasks total (5 fast, 5 standard)
    # We will use the 8 requests instances and 2 flask instances.
    requests_pool = list(repo_instances.get('psf/requests', []))
    flask_pool = list(repo_instances.get('pallets/flask', []))
    
    # Sort them to be deterministic
    requests_pool.sort(key=lambda x: x['instance_id'])
    flask_pool.sort(key=lambda x: x['instance_id'])

    code_fix_pool = requests_pool + flask_pool[:2]

    new_tasks = {}
    code_fix_idx = 0

    # Base commit to use for all non-code_fix tasks so they are anchored to a real SWE-bench state
    default_base_commit = requests_pool[0]['base_commit'] if requests_pool else "a0df2cbb6a5f3bd3e43d14a4ed5d18f13d877402"

    for old_id, task in tasks.items():
        if task['dataset'].startswith('code_fix/'):
            # It's a SWE-bench evaluation task
            if code_fix_idx < len(code_fix_pool):
                swe_task = code_fix_pool[code_fix_idx]
                new_id = swe_task['instance_id'].replace('psf__', '').replace('pallets__', '')
                
                new_tasks[new_id] = {
                    "repo": swe_task['repo'],
                    "commit": swe_task['base_commit'],
                    "dataset": task['dataset'], # Keep our fast/standard designation
                    "expected_mode": task['expected_mode'],
                    "expected_sg_tokens": task['expected_sg_tokens'],
                    "test_cmd": "python eval/scripts/swe_bench_harness.py", # A placeholder since SWE-bench uses docker to run tests
                    "problem": swe_task['problem_statement'],
                    "golden_patch": swe_task['patch'],
                    "test_patch": swe_task['test_patch'],
                    "fail_to_pass": swe_task['FAIL_TO_PASS'],
                    "pass_to_pass": swe_task['PASS_TO_PASS'],
                    "target_function": None,
                    "eval_type": "correctness"
                }
                code_fix_idx += 1
        else:
            # It's a custom evaluation task (Planning, Review, Refactor, Debug)
            # SWE-bench does not contain these. Keep our prompt, but sync the commit.
            task['commit'] = default_base_commit
            new_tasks[old_id] = task

    tasks_file.write_text(json.dumps(new_tasks, indent=2), encoding='utf-8')
    print(f"Successfully synced eval/tasks.json. Replaced code_fix tasks with {code_fix_idx} actual SWE-bench tasks.")

if __name__ == '__main__':
    main()
