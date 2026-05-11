#!/usr/bin/env python3
import json
import subprocess
import sys
from pathlib import Path

def main():
    print("=======================================")
    print("      SWE-bench Test Harness v1.0      ")
    print("=======================================\n")

    cwd = Path.cwd()
    # The user should be running this from eval/runs/<agent>/<task_id>/
    task_json_path = cwd / "task.json"
    
    if not task_json_path.exists():
        print(f"Error: Could not find {task_json_path}.")
        print("Please run this script from inside your task directory:")
        print("Example: cd eval/runs/copilot/requests-1142 && python ../../../../scripts/swe_bench_harness.py")
        sys.exit(1)

    task_info = json.loads(task_json_path.read_text(encoding="utf-8"))
    task_id = task_info["task_id"]
    agent = task_info["agent"]
    
    # Let's grab the test_patch and fail_to_pass tests from the master tasks.json
    # Assume we are at eval/runs/<agent>/<task_id>/
    project_root = cwd.parent.parent.parent.parent
    master_tasks_path = project_root / "eval" / "tasks.json"
    
    if not master_tasks_path.exists():
        print(f"Error: Master tasks file not found at {master_tasks_path}")
        sys.exit(1)
        
    master_tasks = json.loads(master_tasks_path.read_text(encoding="utf-8"))
    if task_id not in master_tasks:
        print(f"Error: Task {task_id} not found in master tasks list.")
        sys.exit(1)
        
    task_data = master_tasks[task_id]
    test_patch = task_data.get("test_patch")
    fail_to_pass_str = task_data.get("fail_to_pass", "[]")
    pass_to_pass_str = task_data.get("pass_to_pass", "[]")
    
    if not test_patch:
        print(f"Note: Task {task_id} does not have a SWE-bench test_patch.")
        print("This is likely a qualitative or structural eval task.")
        sys.exit(0)
        
    fail_to_pass = json.loads(fail_to_pass_str)
    
    print(f"Task: {task_id}")
    print(f"Agent: {agent}")
    print(f"Tests to verify: {len(fail_to_pass)} FAIL_TO_PASS tests")
    
    repo_dir = cwd / "repo"
    if not repo_dir.exists():
        print("Error: repo/ directory not found. Is the workspace set up?")
        sys.exit(1)
        
    # Write the patch
    patch_path = repo_dir / "swe_bench_tests.patch"
    patch_path.write_text(test_patch, encoding="utf-8")
    
    print("\n[1/3] Applying SWE-bench test patch...")
    apply_res = subprocess.run(["git", "apply", "swe_bench_tests.patch"], cwd=repo_dir, capture_output=True, text=True)
    if apply_res.returncode != 0:
        print("Warning: Could not apply test patch cleanly. It might be already applied.")
        print(apply_res.stderr)
        
    print(f"\n[2/3] Running {len(fail_to_pass)} verification tests...")
    
    # We will just pass the specific test node IDs to pytest
    test_args = ["python", "-m", "pytest", "-v"] + fail_to_pass
    
    print(f"Command: {' '.join(test_args)}")
    print("-" * 50)
    
    test_res = subprocess.run(test_args, cwd=repo_dir)
    
    print("-" * 50)
    
    print("\n[3/3] Reverting SWE-bench test patch...")
    revert_res = subprocess.run(["git", "apply", "-R", "swe_bench_tests.patch"], cwd=repo_dir, capture_output=True, text=True)
    if revert_res.returncode != 0:
        print("Warning: Could not revert test patch. You may need to reset it manually.")
    
    # Cleanup patch file
    patch_path.unlink(missing_ok=True)
    
    if test_res.returncode == 0:
        print("\n\u2705 SUCCESS: All SWE-bench verification tests PASSED!")
        print("The agent's code correctly solved the issue.")
        sys.exit(0)
    else:
        print("\n\u274c FAILED: SWE-bench verification tests did not pass.")
        print("The agent's code did not completely solve the issue or introduced an error.")
        sys.exit(1)

if __name__ == "__main__":
    main()
