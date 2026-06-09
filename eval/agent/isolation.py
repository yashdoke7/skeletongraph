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
import time
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
    ".graphify",            # legacy graphify cache
    "graphify-out",         # eval/backends/graphify.py extract output (graph.json)
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
graphify-out/
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
    # The graphify arm needs its PREBUILT graph (graphify-out/graph.json) inside
    # the isolated workspace. Otherwise the backend finds no graph, re-extracts
    # from scratch (a slow LLM build that times out), and every query returns
    # "graph file not found" → all-zero results. The graph is gitignored in the
    # workspace (_WORKSPACE_GITIGNORE), so copying it cannot leak into the patch.
    # Every OTHER arm still excludes + strips it (clean SG-free start).
    keep_graphify = (arm == "graphify" or arm.startswith("graphify"))
    # Dense arms reuse the PREBUILT embeddings (embeddings.npz) in the base repo
    # instead of re-encoding every workspace. The doc vectors load from disk; only
    # the QUERY is encoded at search time (cheap). REQUIRES SG_EMBED_MODEL set at
    # eval time to match the prebuilt vectors' dimension (Jina = 768). Covers ALL
    # arms that use embeddings, not just sg-embed. NOTE: only safe for
    # Verified/Python — the prebuilt index predates the multi-language FQN fix, so
    # REBUILD (don't keep) for Pro / non-Python.
    keep_sg = any(t in arm for t in ("embed", "dense", "fusion")) or arm == "summary-dense"
    copy_excludes = [a for a in _SG_ARTIFACTS
                     if not (keep_graphify and a == "graphify-out") and
                     not (keep_sg and a == ".skeletongraph")]
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
    #
    # Retry: on a Defender-scanned Desktop path with many concurrent workers, a
    # newly-written file can be transiently locked mid-copy ("[WinError 32]" /
    # partial tree). Retrying the whole copy (after wiping the partial) clears it.
    for attempt in range(3):
        try:
            shutil.copytree(src, repo,
                            ignore=shutil.ignore_patterns(*copy_excludes, ".git"),
                            symlinks=False,
                            ignore_dangling_symlinks=True)
            break
        except Exception:
            _rmtree_safe(repo)
            if attempt == 2:
                raise
            time.sleep(0.6 * (attempt + 1))

    _strip_sg_state(repo, keep_graphify=keep_graphify, keep_sg=keep_sg)
    _init_clean_git(repo)
    return repo


def _strip_sg_state(repo: Path, keep_graphify: bool = False,
                    keep_sg: bool = False) -> None:
    """Delete any SkeletonGraph artifacts so the run starts with zero SG state.

    keep_graphify: preserve a prebuilt graphify-out/ (the graphify arm needs it).
    keep_sg: preserve a prebuilt .skeletongraph/ (dense arms reuse its prebuilt
    embeddings instead of re-encoding — see prepare_workspace)."""
    for name in _SG_ARTIFACTS:
        if keep_graphify and name == "graphify-out":
            continue
        if keep_sg and name == ".skeletongraph":
            continue
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

    sequence = (
        [_GIT, "init", "-q"],
        # core.longpaths: Windows `git add` fails with exit 128 on paths >260
        # chars (deeply-nested test fixtures) unless long paths are enabled.
        # core.autocrlf=false: never rewrite line endings (keeps the baseline
        # byte-identical so `git diff` only shows the agent's real edits).
        [_GIT, "config", "core.longpaths", "true"],
        [_GIT, "config", "core.autocrlf", "false"],
        [_GIT, "add", "-A"],
        [_GIT, "commit", "-q", "-m", "baseline", "--no-verify"],
    )
    # Retry the whole init→add→commit. On a Defender-scanned path under heavy
    # concurrency, git intermittently can't stat a just-written file ("unable to
    # stat 'setup.py'") or loses the half-created .git ("not a git repository").
    # These are transient locks, not corruption — re-init + retry clears them.
    last = ""
    for attempt in range(4):
        err = None
        for cmd in sequence:
            r = subprocess.run(cmd, cwd=repo, capture_output=True, text=True,
                               env={**_os_environ(), **env})
            if r.returncode != 0:
                err = (f"git {' '.join(str(c) for c in cmd[1:])} failed (exit "
                       f"{r.returncode}) in {repo}: {(r.stderr or r.stdout).strip()[:300]}")
                break
        if err is None:
            return                      # baseline committed cleanly
        last = err
        gp = repo / ".git"             # wipe the half-built repo before retrying
        if gp.is_dir():
            _rmtree_safe(gp)
        if attempt < 3:
            time.sleep(0.6 * (attempt + 1))
    raise RuntimeError(last + "  (after 4 attempts — likely AV/file-lock; "
                       "lower --workers or exclude the workspace dir from Defender)")


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
