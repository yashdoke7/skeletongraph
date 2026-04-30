import os
import stat
import json
import shutil
import subprocess
from pathlib import Path
from skeletongraph.eval.datasets.swe_bench import load_swe_bench

def remove_readonly(func, path, exc_info):
    """Clear the readonly bit and reattempt the removal. Handles Windows WinError 32 / WinError 5."""
    try:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    except Exception as e:
        print(f"Warning: Could not remove {path}. Error: {e}")

def setup_swe_30():
    sg_root = Path(__file__).parent.resolve()
    temp_dir = sg_root.parent / "temp" / "repos"
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    plan = [
        ("psf/requests", 8),
        ("pallets/flask", 3),
        ("pytest-dev/pytest", 10),
        ("pylint-dev/pylint", 5),
        ("django/django", 4),
    ]
    
    agents = ["antigravity", "claude_code", "codex", "copilot", "cursor"]
    
    print("Loading SWE-bench tasks...")
    selected_tasks = []
    for repo, limit in plan:
        tasks = load_swe_bench(repos=[repo], limit=limit)
        selected_tasks.extend(tasks)
        
    print(f"Loaded {len(selected_tasks)} tasks.")

    # 1. Clone base repos to temp directory for caching
    unique_repos = set((t.repo, t.repo_url) for t in selected_tasks)
    for repo, repo_url in unique_repos:
        repo_name = repo.split("/")[-1]
        repo_path = temp_dir / repo_name
        if not repo_path.exists():
            print(f"Cloning {repo} to cache...")
            subprocess.run(["git", "clone", repo_url, str(repo_path)], check=True)
        else:
            print(f"{repo} already cached.")

    prompts_content = "# SWE-bench 30 Tasks Master Execution Guide\n\n"
    prompts_content += "This document contains all instructions, commands, and prompts for the 30 tasks across all 5 agents.\n\n"
    prompts_content += "## Pre-requisites\n"
    prompts_content += "Ensure SkeletonGraph is installed globally in your environment:\n"
    prompts_content += "```powershell\n"
    prompts_content += "pip install -e .\n"
    prompts_content += "```\n\n"

    # Process all agents
    for agent in agents:
        prompts_content += f"# ==========================================\n"
        prompts_content += f"# AGENT: {agent.upper()}\n"
        prompts_content += f"# ==========================================\n\n"
        
        for task in selected_tasks:
            task_id = task.task_id
            repo_name = task.repo.split("/")[-1]
            
            run_dir = sg_root / "public_eval_runs" / "runs" / agent / task_id
            trace_dir = sg_root / "benchmark_traces" / agent / task_id
            
            # Create directories and JSON placeholders
            run_dir.mkdir(parents=True, exist_ok=True)
            trace_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "native_export.json").write_text("{}", encoding="utf-8")
            (run_dir / "sg_session.json").write_text("{}", encoding="utf-8")
            (trace_dir / "native_trace.json").write_text("{}", encoding="utf-8")
            (trace_dir / "sg_trace.json").write_text("{}", encoding="utf-8")
            
            agent_repo = run_dir / repo_name
            
            # Copy repo if not exists
            if not agent_repo.exists():
                print(f"Copying {repo_name} for {agent} -> {task_id}...")
                cached_repo = temp_dir / repo_name
                
                # To avoid read-only Git file errors on Windows, we copy everything except .git,
                # then initialize a clean git repo. SWE-bench tasks only need the code at that commit.
                try:
                    # Clean up existing aborted copies if any
                    if agent_repo.exists():
                        shutil.rmtree(agent_repo, onerror=remove_readonly)
                    
                    shutil.copytree(cached_repo, agent_repo, ignore=shutil.ignore_patterns('.git'))
                    
                    # Initialize git and commit to give agents a clean slate
                    subprocess.run(["git", "init"], cwd=str(agent_repo), check=True, capture_output=True)
                    subprocess.run(["git", "add", "."], cwd=str(agent_repo), check=True, capture_output=True)
                    subprocess.run(["git", "commit", "-m", "Initial base commit"], cwd=str(agent_repo), check=True, capture_output=True)
                    
                except Exception as e:
                    print(f"Failed to copy {repo_name}: {e}")
                    continue
            
            # Write prompt.txt
            task_meta_dir = sg_root / "public_eval_runs" / "tasks" / task_id
            task_meta_dir.mkdir(parents=True, exist_ok=True)
            (task_meta_dir / "prompt.txt").write_text(task.problem_statement, encoding="utf-8")
            
            # Append to markdown instructions
            prompts_content += f"## Task: {task_id} | Repo: {task.repo}\n\n"
            
            # Commands block
            prompts_content += "### Setup & Commands\n"
            prompts_content += "```powershell\n"
            prompts_content += f"cd public_eval_runs/runs/{agent}/{task_id}/{repo_name}\n\n"
            
            prompts_content += "# --- BASELINE RUN ---\n"
            if agent == "claude_code":
                prompts_content += "claude\n"
            elif agent == "cursor":
                prompts_content += "cursor .\n"
            elif agent == "copilot":
                prompts_content += "code .\n"
            else:
                prompts_content += f"# Launch {agent} here\n"
            prompts_content += "git diff --binary > ../native.patch\n"
            prompts_content += "git reset --hard\n"
            prompts_content += "git clean -fdx\n\n"
            
            prompts_content += "# --- SKELETONGRAPH RUN ---\n"
            prompts_content += "skeletongraph build\n"
            if agent == "claude_code":
                prompts_content += "claude -m mcp.json  # Make sure MCP config maps to SG\n"
            else:
                prompts_content += f"# Launch {agent} with SG MCP enabled here\n"
            prompts_content += "git diff --binary > ../sg.patch\n"
            prompts_content += "git reset --hard\n"
            prompts_content += "git clean -fdx\n"
            prompts_content += "```\n\n"
            
            # Prompt block
            prompts_content += "### Task Prompt (Use for both runs)\n"
            prompts_content += "Copy the following text into the agent:\n"
            prompts_content += f"> {task.problem_statement.strip()}\n\n"
            prompts_content += "---\n\n"

    # Aggregation commands
    prompts_content += "# ==========================================\n"
    prompts_content += "# AGGREGATION & EVALUATION\n"
    prompts_content += "# ==========================================\n\n"
    prompts_content += "After pasting all traces into the `benchmark_traces` folders, run:\n\n"
    prompts_content += "```powershell\n"
    for agent in agents:
        prompts_content += f"skeletongraph eval-benchmark --dataset swe-bench-verified --traces-dir benchmark_traces/{agent} --output benchmark_results/swe_verified_30_{agent}\n"
    prompts_content += "```\n"

    prompts_file = sg_root / "swe_30_tasks_commands.md"
    prompts_file.write_text(prompts_content, encoding="utf-8")
    print(f"\nSetup complete! All 30 tasks for all 5 agents generated.")
    print(f"Open {prompts_file.name} for the master list of commands.")

if __name__ == "__main__":
    setup_swe_30()
