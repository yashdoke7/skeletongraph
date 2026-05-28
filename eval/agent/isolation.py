"""Per-run state isolation.

The HierMem lesson: in a multi-task eval, state leaking from task N into task
N+1 silently corrupts results. Every (task, arm, repeat) run here gets a fresh,
private copy of the repo and zero carried-over SkeletonGraph state.

A run workspace is a full file copy of the task repo at its base commit, placed
under WORKSPACE_ROOT/<run_id>/repo. Nothing is shared between run workspaces.
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
from pathlib import Path

from .config import WORKSPACE_ROOT

# Resolve git executable once at import time.  On Windows, subprocess.run with
# a custom env= sometimes cannot resolve bare "git" even though git is on PATH
# (conda/venv PATH stripping, DLL search path, etc.).  Using the absolute path
# returned by shutil.which bypasses that lookup entirely.
_GIT = shutil.which("git") or "git"

# SG + retrieval-backend state that must NOT survive into a fresh run.
#
# A backend cache committed at baseline and rewritten by retrieval will appear
# in `git diff HEAD` and corrupt the SWE-bench `model_patch` with binary hunks
# (`Binary files differ`) that `git apply` rejects → "error" verify verdicts.
# Hybrid's `.hybrid_index/embeddings.npz` caused this on 29/30 NIM v2 runs.
# Every backend that caches to disk gets listed here AND in the workspace
# .gitignore (see _init_clean_git) — belt and braces.
_SG_ARTIFACTS = [
    ".skeletongraph",
    ".mcp.json",
    ".claude",
    "summary_queue.jsonl",
    "summary_drain.lock",
    "SG_EVAL_RUNBOOK.md",   # per-repo IDE prompt sheet — must not leak into eval
    ".hybrid_index",        # eval/backends/hybrid.py BM25+dense cache
    ".graphify",            # eval/backends/graphify.py knowledge-graph cache
    ".bm25_cache",          # eval/backends/bm25_flat.py (if ever persists)
    ".aider_cache",         # eval/backends/aider_repomap.py override target
    ".aider.tags.cache.v3", # aider default cache (current major version)
    ".aider.tags.cache.v4", # future-proof — aider may bump the version
]

# Files written into the workspace's `.gitignore` before the baseline commit.
# Anything matching here is invisible to `git diff HEAD`, so retrieval backends
# can cache wherever they want without polluting the agent's patch.
_WORKSPACE_GITIGNORE = """\
# CodeMemBench / SG eval — keep retrieval-backend caches out of the patch
.skeletongraph/
.hybrid_index/
.graphify/
.bm25_cache/
.aider_cache/
.aider.tags.cache.v*/
__pycache__/
*.pyc
*.pyo
.pytest_cache/
.mypy_cache/
"""


def _rmtree_safe(path: Path) -> None:
    """shutil.rmtree that survives Windows read-only files (git pack objects etc.).

    git marks objects in .git/objects as read-only on Windows.
    shutil.rmtree(ignore_errors=True) silently skips those files, leaving the
    directory alive. The onerror handler clears the flag and retries — so the
    tree is always fully gone after this call (or raises if it genuinely cannot).
    """
    def _on_error(func, p, exc_info):
        try:
            os.chmod(p, stat.S_IWRITE)
            func(p)
        except Exception:
            pass   # truly unreachable files: accept and move on
    if path.exists():
        shutil.rmtree(str(path), onerror=_on_error)


def run_id(task_id: str, arm: str, repeat: int = 0, model: str = "main") -> str:
    return f"{task_id}__{arm}__{model}__r{repeat}"


def prepare_workspace(task: dict, arm: str, repeat: int = 0,
                      model: str = "main") -> Path:
    """Create a clean, isolated workspace for one run. Returns the repo path.

    The source repo (eval/datasets/repos/<task_id>) is a git worktree at the
    base commit; we copy it so the agent's edits never touch the shared source
    and concurrent runs never collide.
    """
    rid = run_id(task["task_id"], arm, repeat, model)
    work = WORKSPACE_ROOT / rid
    repo = work / "repo"

    _rmtree_safe(work)
    work.mkdir(parents=True, exist_ok=True)

    src = Path(task["repo_path"])
    # Exclude .git: the source repos are git WORKTREES, so their .git is a
    # FILE pointing into a shared cache — copying it leaves a broken pointer.
    # We re-init a clean git repo below so `git diff` still works.
    #
    # symlinks=False (default): follow symlinks and copy the actual content.
    # This avoids creating Windows symlinks in the workspace, which would
    # require Developer Mode and cause `git add -A` to fail (exit 128).
    # ignore_dangling_symlinks=True: silently skip symlinks whose target does
    # not exist (common in Linux-native repos for test fixtures).  Without
    # this, copytree raises FileNotFoundError: [WinError 2] for every dangling
    # link it encounters — which was the original failure mode.
    shutil.copytree(src, repo,
                    ignore=shutil.ignore_patterns(*_SG_ARTIFACTS, ".git"),
                    symlinks=False,
                    ignore_dangling_symlinks=True)

    _strip_sg_state(repo)
    _init_clean_git(repo)
    return repo


def _strip_sg_state(repo: Path) -> None:
    """Delete any SkeletonGraph artifacts so the run starts with zero SG state."""
    for name in _SG_ARTIFACTS:
        p = repo / name
        if p.is_dir():
            _rmtree_safe(p)
        elif p.exists():
            p.unlink()


def _init_clean_git(repo: Path) -> None:
    """Make the workspace a self-contained git repo with one baseline commit.

    The agent edits files; `git diff` against this baseline is the patch we
    submit for verification.
    """
    git_path = repo / ".git"
    if git_path.is_dir():
        _rmtree_safe(git_path)
    elif git_path.exists():          # git worktree: .git is a FILE, not a dir
        git_path.unlink()
    env = {"GIT_AUTHOR_NAME": "sg-eval", "GIT_AUTHOR_EMAIL": "eval@local",
           "GIT_COMMITTER_NAME": "sg-eval", "GIT_COMMITTER_EMAIL": "eval@local",
           # Prevent git from reading system/user .gitconfig — avoids broken
           # config files or safeDirectory complaints in the subprocess env.
           "GIT_CONFIG_NOSYSTEM": "1"}

    # Write a workspace .gitignore BEFORE `git add -A` so retrieval-backend
    # caches never enter the baseline (and therefore never appear in
    # `git diff HEAD` later). If the repo already has a .gitignore, append.
    gi = repo / ".gitignore"
    existing = gi.read_text(encoding="utf-8", errors="replace") if gi.exists() else ""
    if "# CodeMemBench / SG eval" not in existing:
        gi.write_text(existing + ("\n" if existing and not existing.endswith("\n") else "")
                      + _WORKSPACE_GITIGNORE, encoding="utf-8")

    for cmd in (
        [_GIT, "init", "-q"],
        # core.longpaths: Windows `git add` fails with exit 128 on paths >260
        # chars (deeply-nested test fixtures) unless long paths are enabled.
        # core.autocrlf=false: never rewrite line endings (keeps the baseline
        # byte-identical so `git diff` only shows the agent's real edits).
        [_GIT, "config", "core.longpaths", "true"],
        [_GIT, "config", "core.autocrlf", "false"],
        [_GIT, "add", "-A"],
        [_GIT, "commit", "-q", "-m", "baseline", "--no-verify"],
    ):
        r = subprocess.run(cmd, cwd=repo, capture_output=True, text=True,
                           env={**_os_environ(), **env})
        if r.returncode != 0:
            # Surface git's real message (CalledProcessError hides stderr).
            raise RuntimeError(
                f"git {' '.join(str(c) for c in cmd[1:])} failed (exit "
                f"{r.returncode}) in {repo}: {(r.stderr or r.stdout).strip()[:300]}")


def diff_patch(repo: Path) -> str:
    """The agent's changes as a unified diff against the baseline commit."""
    r = subprocess.run([_GIT, "diff", "HEAD"], cwd=repo,
                       capture_output=True, text=True)
    return r.stdout


def cleanup_workspace(repo: Path) -> None:
    """Remove a finished run's workspace (call after the trajectory is saved)."""
    work = repo.parent
    _rmtree_safe(work)


def assert_isolation(repo_a: Path, repo_b: Path) -> bool:
    """Sanity check: two prepared workspaces share no path. Pre-flight gate."""
    return repo_a.resolve() != repo_b.resolve() and \
        repo_a.parent.resolve() != repo_b.parent.resolve()


def _os_environ() -> dict:
    import os
    return dict(os.environ)
