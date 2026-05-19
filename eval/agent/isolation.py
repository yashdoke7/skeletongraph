"""Per-run state isolation.

The HierMem lesson: in a multi-task eval, state leaking from task N into task
N+1 silently corrupts results. Every (task, arm, repeat) run here gets a fresh,
private copy of the repo and zero carried-over SkeletonGraph state.

A run workspace is a full file copy of the task repo at its base commit, placed
under WORKSPACE_ROOT/<run_id>/repo. Nothing is shared between run workspaces.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from .config import WORKSPACE_ROOT

# SG state that must NOT survive into a fresh run.
_SG_ARTIFACTS = [
    ".skeletongraph",
    ".mcp.json",
    ".claude",
    "summary_queue.jsonl",
    "summary_drain.lock",
]


def run_id(task_id: str, arm: str, repeat: int = 0, model: str = "qwen-32b") -> str:
    return f"{task_id}__{arm}__{model}__r{repeat}"


def prepare_workspace(task: dict, arm: str, repeat: int = 0,
                      model: str = "qwen-32b") -> Path:
    """Create a clean, isolated workspace for one run. Returns the repo path.

    The source repo (eval/datasets/repos/<task_id>) is a git worktree at the
    base commit; we copy it so the agent's edits never touch the shared source
    and concurrent runs never collide.
    """
    rid = run_id(task["task_id"], arm, repeat, model)
    work = WORKSPACE_ROOT / rid
    repo = work / "repo"

    if work.exists():
        shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True, exist_ok=True)

    src = Path(task["repo_path"])
    # Exclude .git: the source repos are git WORKTREES, so their .git is a
    # FILE pointing into a shared cache — copying it leaves a broken pointer.
    # We re-init a clean git repo below so `git diff` still works.
    shutil.copytree(src, repo,
                    ignore=shutil.ignore_patterns(*_SG_ARTIFACTS, ".git"))

    _strip_sg_state(repo)
    _init_clean_git(repo)
    return repo


def _strip_sg_state(repo: Path) -> None:
    """Delete any SkeletonGraph artifacts so the run starts with zero SG state."""
    for name in _SG_ARTIFACTS:
        p = repo / name
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
        elif p.exists():
            p.unlink()


def _init_clean_git(repo: Path) -> None:
    """Make the workspace a self-contained git repo with one baseline commit.

    The agent edits files; `git diff` against this baseline is the patch we
    submit for verification.
    """
    git_path = repo / ".git"
    if git_path.is_dir():
        shutil.rmtree(git_path, ignore_errors=True)
    elif git_path.exists():          # git worktree: .git is a FILE, not a dir
        git_path.unlink()
    env = {"GIT_AUTHOR_NAME": "sg-eval", "GIT_AUTHOR_EMAIL": "eval@local",
           "GIT_COMMITTER_NAME": "sg-eval", "GIT_COMMITTER_EMAIL": "eval@local"}
    for cmd in (
        ["git", "init", "-q"],
        ["git", "add", "-A"],
        ["git", "commit", "-q", "-m", "baseline", "--no-verify"],
    ):
        subprocess.run(cmd, cwd=repo, check=True, capture_output=True,
                       env={**_os_environ(), **env})


def diff_patch(repo: Path) -> str:
    """The agent's changes as a unified diff against the baseline commit."""
    r = subprocess.run(["git", "diff", "HEAD"], cwd=repo,
                       capture_output=True, text=True)
    return r.stdout


def cleanup_workspace(repo: Path) -> None:
    """Remove a finished run's workspace (call after the trajectory is saved)."""
    work = repo.parent
    shutil.rmtree(work, ignore_errors=True)


def assert_isolation(repo_a: Path, repo_b: Path) -> bool:
    """Sanity check: two prepared workspaces share no path. Pre-flight gate."""
    return repo_a.resolve() != repo_b.resolve() and \
        repo_a.parent.resolve() != repo_b.parent.resolve()


def _os_environ() -> dict:
    import os
    return dict(os.environ)
