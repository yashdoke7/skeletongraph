"""Drive REAL Claude Code (headless) on SWE-bench, with SkeletonGraph wired in
as an MCP server providing the engine-side `sg-rerank` retrieval.

This is the "real frontier agent uses SG" arm — distinct from the controlled
ReAct harness (run_stage.py). Instead of our 5-tool ReAct loop talking to a
vLLM/NIM endpoint, we launch the actual `claude` CLI in print/headless mode on
a per-task editable repo copy. SG is the project's MCP server (.mcp.json), so
Claude reaches code through sg_search/sg_get/sg_expand (sg-rerank) exactly as a
real user would. Native Read/Grep/Edit stay enabled — the honest "SG available,
prefer it" setup, not a forced ablation.

Pipeline per task:
  1. prepare_repo  — persistent editable copy at base commit, clean git baseline,
                     `sg build` index + `sg install --ide claude-code` (.mcp.json
                     + hooks + CLAUDE.md). SG artifacts are gitignored so they
                     never pollute the patch. Idempotent (skip if prepared).
  2. run_claude    — `claude -p` (stream-json) in that dir, SG as strict MCP.
  3. extract_patch — `git add -A && git diff --cached` (captures new files too;
                     gitignored SG state excluded).
  4. write a run JSON in the SAME schema verify.py / aggregate.py consume, so the
     sg-rerank/Claude arm folds into the existing pass@1 + tables unchanged.

Isolate results from the vLLM/NIM runs with a distinct tag:
    $env:SG_EVAL_RUN_TAG = "claude_sgrerank"

Run 4-5 of these in parallel terminals, each pinned to a task shard:
    python -m eval.agent.run_claude_code --dataset <swebench_100.jsonl> --shard 1/5
    python -m eval.agent.run_claude_code --dataset <swebench_100.jsonl> --shard 2/5
    ... (shards 3/5, 4/5, 5/5 in their own terminals)

Pre-stage every editable copy first (one-time, satisfies "make sure they are
copied") without running any agent:
    python -m eval.agent.run_claude_code --dataset <swebench_100.jsonl> --prepare-only

Then score with the existing harness (same tag):
    python -m eval.agent.verify --all --only-arms sg-rerank --run-tag claude_sgrr
    python -m eval.agent.aggregate
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from . import config
from .isolation import _GIT, _rmtree_safe, run_id
from .run_agent import load_tasks

SG = shutil.which("sg") or "sg"
CLAUDE = shutil.which("claude") or "claude"

ARM_SG = "sg-rerank"     # MCP server pinned to SG_MCP_RETRIEVAL=rerank (BM25+structural, no dense)
ARM_FUSION = "sg-fusion"  # MCP server pinned to SG_MCP_RETRIEVAL=fusion (3-way RRF incl. dense)
# sg-fusion-v3: IDENTICAL config to sg-fusion (same retrieval mode, same prompt,
# same body_top=0) — the only difference is the underlying product code
# (engine.py::_expand_function, fixed 2026-07) now renders per-line line numbers
# in sg_expand's function body, matching _expand_range's existing format. Fixes
# the measured cause of 81% of fusion's redundant Read calls (the agent had no
# line numbers to build an Edit from and re-Read the file just to get them).
# Distinct arm name (not a reused "sg-fusion-v2" — that name already means the
# DIFFERENT, discarded body_top=1+disallow_grep deterrent variant, see
# project_sg_unification memory) purely so pre-fix (existing sg-fusion results)
# and post-fix runs land in separate files under the same tag, not commingled.
# See project_fusion_cost_diagnosis memory for the full measurement + rationale.
ARM_FUSION_V3 = "sg-fusion-v3"
ARM_CBMEM = "cbmem"      # competitor: Codebase-Memory MCP server (tree-sitter knowledge graph)
ARM_SERENA = "serena"    # competitor: Serena MCP server (LSP-based symbol navigation, 25k stars)
ARM_GITNEXUS = "gitnexus"  # competitor: GitNexus MCP server (knowledge graph, own SWE-bench claim)
ARM_NATIVE = "native"    # Claude Code on its own — no SG, native tools only
ARMS = (ARM_SG, ARM_FUSION, ARM_FUSION_V3, ARM_CBMEM, ARM_SERENA, ARM_GITNEXUS, ARM_NATIVE)
# SG's own MCP-server family (all launch `sg serve`, differ only in retrieval mode).
# cbmem is a competitor MCP server; native has no MCP. Used to branch prepare/run.
_SG_ARMS = frozenset({ARM_SG, ARM_FUSION, ARM_FUSION_V3})

# Every non-native arm launches the same MCP server binary — what actually
# differs is which retrieval algorithm it serves. The server defaults to
# "fusion" (product default — see retrieval/fusion.py), so without pinning
# this explicitly per arm, an "sg-rerank"-labeled run silently tests whatever
# the server's current default is, not what its own name promises. Bit us
# once already: every claude-code "sg-rerank" run after the fusion port
# landed was actually measuring fusion. Pin it here so the arm label is a
# guarantee, not a hope.
_RETRIEVAL_MODE = {ARM_SG: "rerank", ARM_FUSION: "fusion", ARM_FUSION_V3: "fusion"}

# SG artifacts + standard caches kept OUT of the agent's patch. Written to the
# copy's .gitignore BEFORE the baseline commit, so `git add -A` never stages
# them and `git diff` never shows them — same belt-and-braces idea as
# isolation._WORKSPACE_GITIGNORE, plus the Claude-Code-specific files.
_GITIGNORE = """\
# SkeletonGraph + Claude Code eval — keep SG/agent config out of the patch
.skeletongraph/
.mcp.json
.claude/
CLAUDE.md
.sg_prepared
.hybrid_index/
.gitnexus/
__pycache__/
*.pyc
*.pyo
.pytest_cache/
.mypy_cache/
"""

# Reinforces CLAUDE.md in case -p mode does not surface project memory the same
# way an interactive session does. Kept short — the MCP tool descriptions and
# CLAUDE.md carry the detail.
_SG_APPEND_SYSTEM = (
    "SkeletonGraph (SG) is wired in as an MCP server for this repo. To locate "
    "code, use sg_search (a whole-task context assembler, not grep); it returns "
    "the edit targets as exact anchors (file::symbol + line range), NOT bodies. "
    "To read a body, call sg_expand(target=\"<fqn>\") — it returns the exact "
    "current source with file:line, so edit DIRECTLY from that. Do NOT re-Read or "
    "re-grep a symbol whose body sg_expand already gave you; that just repeats "
    "work and adds turns. sg_expand accepts several FQNs at once (comma-separated) "
    "— batch them in one call instead of one per function. sg_overview is OPTIONAL "
    "— call it only if you actually need project orientation (unfamiliar codebase, "
    "architecture or cross-cutting work); skip it for a focused bug fix. Use native "
    "Grep/Read only for what SG did not return (e.g. finding where to insert NEW "
    "code)."
)

# ── cbmem competitor arm: Codebase-Memory as a real Claude Code MCP server ────
# cbmem's persistent index store is shared between the CLI (which we index with,
# in prepare) and the stdio MCP server Claude Code launches from .mcp.json — so a
# repo indexed here is queryable in-session. Every cbmem tool needs a `project`
# slug; we compute the real slug at prepare/run time and pin it into the prompt.
_CBMEM_APPEND_SYSTEM = (
    "Codebase-Memory (cbmem) is wired in as an MCP server for this repo — a "
    "tree-sitter code knowledge graph. It is your code-search tool; prefer it "
    "over native grep to locate code. Tools (ALWAYS pass project=\"{slug}\"): "
    "search_graph(project=\"{slug}\", query=\"<what you're looking for>\") — find "
    "functions/classes/routes; get_code_snippet(project=\"{slug}\", "
    "qualified_name=\"<name from a search result>\") — read a symbol's source; "
    "search_code(project=\"{slug}\", pattern=\"<text>\") — graph-augmented grep; "
    "trace_path(project=\"{slug}\", function_name=\"<name>\") — follow callers/"
    "callees. Locate the code with cbmem, read it with get_code_snippet, then edit "
    "directly. Only fall back to native Read for a file you must edit."
)


def _cbmem_mod():
    from backends import cbmem as _c
    return _c


def _cbmem_slug(repo: Path) -> str:
    """Index the repo into cbmem's graph (no-op if already built) and return the
    ACTUAL registered project slug — cbmem derives its own slug, so we read it back
    rather than compute it."""
    c = _cbmem_mod()
    return c._ensure_indexed(c._bin(), Path(repo).resolve())


def _write_cbmem_mcp(repo: Path) -> None:
    """Write .mcp.json pointing Claude Code at the cbmem stdio MCP server."""
    binp = _cbmem_mod()._bin().replace("\\", "/")
    cfg = {"mcpServers": {"codebase-memory": {
        "type": "stdio", "command": binp, "args": []}}}
    (repo / ".mcp.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")


# ── Serena competitor arm: LSP-based symbol navigation, 25k stars ────────────
# Unlike cbmem (persistent global graph store), Serena is per-project and
# activates the project at MCP-server launch via --project — no separate index
# step. Requires: pip install serena-agent in its own venv (SERENA_BIN ->
# .../Scripts/serena.exe) + Node.js on PATH (pyright's LSP backend, launched via
# uvx, is a Node program under the hood — install Node if `uvx --from pyright
# pyright-langserver --version` fails).
_SERENA_APPEND_SYSTEM = (
    "Serena is wired in as an MCP server for this repo — an LSP-based semantic "
    "code navigator (real language-server symbol resolution, not text search). "
    "It is your code-search tool; prefer it over native grep to locate code. "
    "Tools: get_symbols_overview(relative_path) — top-level symbols in a file "
    "(call this on a file before diving in); find_symbol(name_path_pattern, "
    "include_body=true) — locate a function/class BY NAME and read its exact "
    "source in one call; find_referencing_symbols(name_path, relative_path) — "
    "find every caller of a symbol; search_for_pattern(substring_pattern) — "
    "text/regex search when you don't know a symbol name. Locate and read code "
    "with these tools, then edit directly. Only fall back to native Read for a "
    "file you must edit."
)

# ── GitNexus competitor arm: knowledge-graph MCP, own SWE-bench claim ────────
# Unlike cbmem (global store queried by explicit `project` slug) and Serena
# (per-project via --project flag), GitNexus's MCP server has no project flag —
# it resolves the active repo from the launch cwd (confirmed live: `gitnexus mcp`
# started with cwd=<repo> auto-selects that repo without a `repo` arg). We ALSO
# register the repo under a deterministic --name alias at analyze time and tell
# Claude to pass repo="<alias>" explicitly, as a belt-and-braces fallback in case
# cwd auto-detection is ever ambiguous (e.g. a stale global registry entry).
_GITNEXUS_APPEND_SYSTEM = (
    "GitNexus is wired in as an MCP server for this repo — a knowledge-graph "
    "code navigator. It is your code-search tool; prefer it over native grep to "
    "locate code. Tools (pass repo=\"{name}\" on every call): "
    "query(search_query=\"<what you're looking for>\", repo=\"{name}\") — hybrid "
    "BM25+semantic search, returns matching symbols with file paths; "
    "context(name=\"<symbol>\", repo=\"{name}\") — 360-degree view of a symbol "
    "(callers/callees); trace(from=\"<symbol>\", to=\"<symbol>\", repo=\"{name}\") "
    "— shortest path between two symbols; impact(target=\"<symbol>\", "
    "repo=\"{name}\") — blast-radius of changing a symbol. Locate code with "
    "these tools, then edit directly. Only fall back to native Read for a file "
    "you must edit."
)


def _gitnexus_cmd() -> list:
    """Argv prefix to invoke GitNexus. On Windows, npm's global `gitnexus.cmd`
    wrapper can't be spawned directly by CreateProcess without a shell (neither
    Claude Code's MCP client nor Python's subprocess resolves .cmd association
    without shell=True) — resolve straight to `node <package>/dist/cli/index.js`
    instead, same "point at the real executable" fix used for Serena/cbmem's
    .exe. GITNEXUS_CMD overrides with a space-separated argv prefix if the
    default resolution doesn't fit a given machine."""
    env = os.environ.get("GITNEXUS_CMD")
    if env:
        return env.split()
    node = shutil.which("node")
    if not node:
        raise RuntimeError(
            "node not found on PATH — GitNexus needs Node.js. Install it, or "
            "set GITNEXUS_CMD to a full 'node /path/to/index.js'-style prefix.")
    cmd_wrapper = shutil.which("gitnexus.cmd") or shutil.which("gitnexus")
    if not cmd_wrapper:
        raise RuntimeError(
            "GitNexus not found. Install with `npm install -g gitnexus@latest` "
            "and ensure npm's global bin dir is on PATH, or set GITNEXUS_CMD.")
    entry = Path(cmd_wrapper).with_name("node_modules") / "gitnexus" / "dist" / "cli" / "index.js"
    if not entry.is_file():
        raise RuntimeError(f"GitNexus JS entrypoint not found at {entry} — "
                           f"install layout may differ; set GITNEXUS_CMD directly.")
    return [node, str(entry)]


def _gitnexus_analyze(repo: Path) -> None:
    """Index the repo (idempotent — incremental if the git tree is unchanged).
    --index-only skips AGENTS.md/CLAUDE.md/skills injection into the tracked
    tree (would otherwise pollute the patch, same care as SG's gitignored
    install). --name registers a deterministic alias for the system-prompt
    repo= fallback."""
    cmd = _gitnexus_cmd() + ["analyze", str(repo), "--index-only",
                             "--name", repo.name]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", cwd=str(repo))
    if r.returncode != 0:
        raise RuntimeError(f"`gitnexus analyze` failed ({r.returncode}): "
                           f"{(r.stderr or r.stdout).strip()[:300]}")


def _write_gitnexus_mcp(repo: Path) -> None:
    cmd = _gitnexus_cmd()
    cfg = {"mcpServers": {"gitnexus": {
        "type": "stdio", "command": cmd[0].replace("\\", "/"),
        "args": [*(a.replace("\\", "/") for a in cmd[1:]), "mcp"]}}}
    (repo / ".mcp.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")


def _serena_bin() -> str:
    env = os.environ.get("SERENA_BIN")
    if env:
        return env
    found = shutil.which("serena")
    if found:
        return found
    raise RuntimeError(
        "Serena binary not found. Install with `pip install serena-agent` in "
        "its own venv and set SERENA_BIN to the full path of serena(.exe), or "
        "put it on PATH.")


def _write_serena_mcp(repo: Path) -> None:
    """Write .mcp.json pointing Claude Code at the Serena stdio MCP server,
    pre-activated on this exact repo copy via --project."""
    binp = _serena_bin().replace("\\", "/")
    repo_path = str(Path(repo).resolve()).replace("\\", "/")
    cfg = {"mcpServers": {"serena": {
        "type": "stdio", "command": binp,
        "args": ["start-mcp-server", "--transport", "stdio",
                 "--project", repo_path,
                 "--enable-web-dashboard", "False",
                 "--open-web-dashboard", "False"]}}}
    (repo / ".mcp.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

# Scope-discipline block — BYTE-IDENTICAL in both prompts on purpose. It is
# about task scope, not retrieval, so both arms must get exactly the same words
# or the cost comparison stops being apples-to-apples. Motivated directly by the
# observed failure: richer retrieval made the SG arm AWARE of adjacent code (the
# mathtext render path, .pyi type stubs) and it "helpfully" extended the fix and
# synced stubs / added changelog notes the issue never asked for — same core
# outcome, ~2x the turns/tokens. This tells BOTH arms to stop at the fix the
# issue actually requires, so any remaining cost delta is retrieval, not
# gold-plating. The "the fix does not require" qualifier keeps genuinely needed
# stub/related edits allowed — it forbids thoroughness-for-its-own-sake, not
# correctness.
_SCOPE_BLOCK = """\
- Make the smallest change that correctly fixes the issue, and nothing more.
- Stop as soon as the issue's described behaviour is correct. Do NOT keep \
searching or reading "to be thorough" once you already have the code needed to \
make the fix — only look further if you are still missing something THIS fix \
requires.
- Do NOT add changelog / whatsnew / release-note entries, update type-stub \
(.pyi) files, write documentation, or reformat/refactor code that the fix does \
not require.
- Only touch extra files or extend to related features/dependencies if the \
issue explicitly asks for it.
- Do NOT run or write tests — the test environment is not available."""

_SG_PROMPT = """Fix the following GitHub issue in this repository by editing the \
source files directly.

--- ISSUE ---
{issue}

Guidelines:
- Prefer the SkeletonGraph MCP tools (sg_overview, sg_search, sg_get, sg_expand) \
to locate the relevant code before reading or grepping.
{scope}
- When the fix is complete, stop.
"""

# Native baseline — Claude Code on its own. No SG mention, so the agent uses its
# own tools (Grep/Read/Edit/...) exactly as it would for any user. This is the
# control the SG-wrapped arm is measured against. Same scope block as the SG arm
# (see _SCOPE_BLOCK) so the ONLY prompt difference is the SG-tool guidance.
_NATIVE_PROMPT = """Fix the following GitHub issue in this repository by editing \
the source files directly.

--- ISSUE ---
{issue}

Guidelines:
{scope}
- When the fix is complete, stop.
"""


# ── git helpers (env mirrors isolation._init_clean_git) ──────────────────────

_GIT_ENV = {
    "GIT_AUTHOR_NAME": "sg-eval", "GIT_AUTHOR_EMAIL": "eval@local",
    "GIT_COMMITTER_NAME": "sg-eval", "GIT_COMMITTER_EMAIL": "eval@local",
    "GIT_CONFIG_NOSYSTEM": "1",
}


def _git(repo: Path, *args: str, check: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run([_GIT, *args], cwd=str(repo), capture_output=True,
                          text=True, encoding="utf-8", errors="replace",
                          env={**os.environ, **_GIT_ENV}, check=check)


def _copies_root(task: dict) -> Path:
    """Persistent editable copies live next to the source clones, in a sibling
    _claude_repos dir (NOT the ephemeral _agent_work tree)."""
    return Path(task["repo_path"]).resolve().parent.parent / "_claude_repos"


def _repo_dir(task: dict, arm: str = ARM_SG) -> Path:
    # sg-rerank keeps the bare task dir (back-compat with already-prepared
    # copies); every other arm namespaces so SG and native copies never clash.
    base = _copies_root(task)
    return base / (task["task_id"] if arm == ARM_SG
                   else f"{task['task_id']}__{arm}")


# ── prepare: editable copy + clean baseline + (SG arm only) index + MCP ──────

def prepare_repo(task: dict, arm: str = ARM_SG, rebuild: bool = False,
                 verbose: bool = True) -> Path:
    """Create (or reuse) a persistent editable copy.

    SG arm: + `sg build` index + `sg install --ide claude-code` (.mcp.json,
    hooks, CLAUDE.md), all gitignored. Native arm: clean repo only — NO SG, so
    it is a fair "Claude on its own" control. Idempotent via a `.sg_prepared`
    marker; pass rebuild=True to wipe and redo.
    """
    repo = _repo_dir(task, arm)
    marker = repo / ".sg_prepared"

    if marker.exists() and not rebuild:
        # Refresh Claude Code hooks/MCP config (idempotent, gitignored) so copies
        # prepared before a hook/install change pick it up — e.g. the SG-first
        # PreToolUse gate — without a slow re-index.
        if arm in _SG_ARMS:
            try:
                _sg(repo, "install", "--ide", "claude-code", "--path", str(repo))
            except Exception:
                pass
            if _RETRIEVAL_MODE.get(arm) == "fusion":
                _warm_dense_cache(repo)   # no-op if already warm (cache present)
        elif arm == ARM_CBMEM:
            _cbmem_slug(repo)             # ensure indexed (persistent; cheap if built)
            _write_cbmem_mcp(repo)
        elif arm == ARM_SERENA:
            _write_serena_mcp(repo)       # no index step - Serena activates per-project at launch
        elif arm == ARM_GITNEXUS:
            _gitnexus_analyze(repo)       # cheap/incremental if already indexed
            _write_gitnexus_mcp(repo)
        reset_repo(repo)
        return repo

    src = Path(task["repo_path"]).resolve()
    if not src.is_dir():
        raise FileNotFoundError(f"source clone missing: {src}")

    if verbose:
        print(f"  prepare {task['task_id']} -> {repo}")
    _rmtree_safe(repo)
    repo.parent.mkdir(parents=True, exist_ok=True)

    # Copy the worktree, excluding .git (a worktree's .git is a FILE pointing
    # into a shared cache — copying it leaves a broken pointer) and any SG state.
    excludes = (".git", ".skeletongraph", ".mcp.json", ".claude", "CLAUDE.md",
                ".hybrid_index")
    for attempt in range(3):
        try:
            shutil.copytree(src, repo,
                            ignore=shutil.ignore_patterns(*excludes),
                            symlinks=False, ignore_dangling_symlinks=True)
            break
        except Exception:
            _rmtree_safe(repo)
            if attempt == 2:
                raise
            time.sleep(0.6 * (attempt + 1))

    # .gitignore SG/agent state BEFORE the baseline commit.
    gi = repo / ".gitignore"
    existing = gi.read_text(encoding="utf-8", errors="replace") if gi.exists() else ""
    if "# SkeletonGraph + Claude Code eval" not in existing:
        gi.write_text(existing + ("\n" if existing and not existing.endswith("\n") else "")
                      + _GITIGNORE, encoding="utf-8")

    # Clean git baseline — the agent's diff is taken against this commit.
    _init_baseline(repo)

    if arm in _SG_ARMS:
        # Build the SG index, then install Claude Code integration (.mcp.json +
        # hooks + CLAUDE.md). Both write only gitignored paths. The native arm
        # skips this entirely so it stays a genuine SG-free baseline.
        _sg(repo, "build", "--path", str(repo))
        _sg(repo, "install", "--ide", "claude-code", "--path", str(repo))
        if _RETRIEVAL_MODE.get(arm) == "fusion":
            _warm_dense_cache(repo)
    elif arm == ARM_CBMEM:
        # Index the FRESH copy into cbmem's graph + point Claude at cbmem's MCP
        # server. cbmem's index lives in its own store (not the repo), so the only
        # in-repo file is .mcp.json (already gitignored, like SG's).
        _cbmem_slug(repo)
        _write_cbmem_mcp(repo)
    elif arm == ARM_SERENA:
        _write_serena_mcp(repo)
    elif arm == ARM_GITNEXUS:
        # Index the FRESH copy — .gitnexus/ is gitignored (added to _GITIGNORE
        # above) so, like cbmem/SG, the only tracked-tree file is .mcp.json.
        _gitnexus_analyze(repo)
        _write_gitnexus_mcp(repo)

    # Safety net: prepare must leave a CLEAN tracked tree (SG state all ignored).
    dirty = _git(repo, "status", "--porcelain").stdout.strip()
    if dirty:
        print(f"  WARN {task['task_id']}: SG install touched tracked files — "
              f"patch may be polluted:\n{dirty[:400]}")

    marker.write_text("ok\n", encoding="utf-8")
    return repo


def _dense_cache_dir(repo: Path) -> Path:
    return repo / ".skeletongraph" / "dense_cache"


def _warm_dense_cache(repo: Path) -> None:
    """Fully build fusion's dense-embedding cache BEFORE the timed agent run.

    Fusion's dense leg (retrieval/dense.py) encodes the entire function corpus on
    first use and caches it to .skeletongraph/dense_cache/embcache_code.npz. COLD,
    that encode can approach or exceed the live SG_DENSE_TIMEOUT_S bound the MCP
    server guards each call with — so the agent's very first sg_search could
    silently degrade to a 2-way (bm25+structural) fusion instead of the intended
    3-way, and the LLM would be handed a weaker result than the paper claims.

    Building it here, once, with NO timeout, guarantees every in-run fusion call
    reloads a ready .npy (fast) and is the real 3-way retrieval. This is one-time,
    index-class cost OUTSIDE the agent's clock — same bucket as `sg build`, not
    charged to agent latency/tokens. Idempotent: a present cache is a no-op.

    Runs in the harness interpreter (sg-env), writing the same on-disk cache the
    MCP child process later reads — the cache key is (model, repo-content-hash),
    identical across both, so the child gets a guaranteed hit.
    """
    cache = _dense_cache_dir(repo)
    if cache.is_dir() and any(cache.glob("embcache_*.npz")):
        return   # already warm
    t0 = time.time()
    try:
        from skeletongraph.retrieval import dense as _dense
        # A real retrieve() builds + writes the corpus doc-embeddings cache. The
        # query is irrelevant — only the doc-side .npy is cached (keyed by repo
        # content + model), and that is exactly what the runtime call reloads.
        _dense.retrieve("warm-up", repo, 1)
        n = len(list(cache.glob("embcache_*.npz"))) if cache.is_dir() else 0
        print(f"  dense pre-warm {repo.name}: {round(time.time()-t0,1)}s "
              f"({n} cache file(s)) — first sg_search now full 3-way fusion")
    except Exception as e:
        # Non-fatal: without the cache the run still works, it just risks a
        # degraded first call. Surface it loudly so a paper run isn't silently
        # 2-way. (Most likely cause: sentence-transformers/model not available
        # in this interpreter — the same dep the MCP child needs anyway.)
        print(f"  WARN dense pre-warm FAILED for {repo.name} ({round(time.time()-t0,1)}s): "
              f"{e} — fusion may degrade to 2-way on first call")


def _init_baseline(repo: Path) -> None:
    seq = (
        [_GIT, "init", "-q"],
        [_GIT, "config", "core.longpaths", "true"],
        [_GIT, "config", "core.autocrlf", "false"],
        [_GIT, "add", "-A"],
        [_GIT, "commit", "-q", "-m", "baseline", "--no-verify"],
    )
    last = ""
    for attempt in range(4):
        err = None
        for cmd in seq:
            r = subprocess.run(cmd, cwd=str(repo), capture_output=True, text=True,
                               encoding="utf-8", errors="replace",
                               env={**os.environ, **_GIT_ENV})
            if r.returncode != 0:
                err = (f"git {' '.join(cmd[1:])} failed ({r.returncode}): "
                       f"{(r.stderr or r.stdout).strip()[:200]}")
                break
        if err is None:
            return
        last = err
        gp = repo / ".git"
        if gp.is_dir():
            _rmtree_safe(gp)
        if attempt < 3:
            time.sleep(0.6 * (attempt + 1))
    raise RuntimeError(last + "  (after 4 attempts — likely AV/file-lock)")


def _sg(repo: Path, *args: str) -> None:
    r = subprocess.run([SG, *args], capture_output=True, text=True,
                       encoding="utf-8", errors="replace", cwd=str(repo))
    if r.returncode != 0:
        raise RuntimeError(f"`sg {' '.join(args)}` failed ({r.returncode}): "
                           f"{(r.stderr or r.stdout).strip()[:300]}")


def reset_repo(repo: Path) -> None:
    """Return the copy to its baseline (discard the previous run's edits and any
    agent-created files). Gitignored SG state (.skeletongraph/.mcp.json) survives
    `git clean -fd` — only -x would remove ignored files — so the index is reused."""
    _git(repo, "reset", "--hard", "HEAD")
    _git(repo, "clean", "-fd")


def extract_patch(repo: Path) -> str:
    """Agent's changes as a unified diff, including new files. SG state is
    gitignored, so `git add -A` never stages it."""
    _git(repo, "add", "-A")
    patch = _git(repo, "diff", "--cached", "HEAD").stdout
    _git(repo, "reset", "-q")   # unstage; leave working tree as-is for inspection
    return patch


# ── run Claude Code headless ─────────────────────────────────────────────────

def run_claude(repo: Path, issue: str, model: str, timeout: int,
               arm: str = ARM_SG, disallow_grep: bool = False,
               body_top: int = 0) -> dict:
    """Launch `claude -p` in the repo.

    SG arms (sg-rerank / sg-fusion): SG as the strict MCP server, pinned via
    SG_MCP_RETRIEVAL to the algorithm the arm name promises (see
    _RETRIEVAL_MODE — the server's own default is a moving target as the
    product evolves, so every arm must nail it down explicitly or the label
    lies about what actually ran) + an SG-first system nudge + SG prompt.
    Native arm: `--strict-mcp-config` with NO config (disables ALL project/global
    MCP, so it is SG-free) + a neutral prompt.
    disallow_grep: block native Grep/Glob (forces search through SG / read) —
    use to test SG as the SOLE retrieval surface.

    Returns {ok, exit, transcript (stream-json objects), result, raw}.
    """
    cmd = [
        CLAUDE, "-p",
        "--output-format", "stream-json",
        "--verbose",
        "--model", model,
        "--dangerously-skip-permissions",
    ]
    env = dict(os.environ)
    if arm in _SG_ARMS:
        cmd += ["--mcp-config", str(repo / ".mcp.json"), "--strict-mcp-config",
                "--append-system-prompt", _SG_APPEND_SYSTEM]
        prompt = _SG_PROMPT.format(issue=issue, scope=_SCOPE_BLOCK)
        # Claude Code spawns the MCP server (`sg serve`) as a child process
        # inheriting this env, which is where `sg serve` reads it from.
        env["SG_MCP_RETRIEVAL"] = _RETRIEVAL_MODE[arm]
        # Payload shape: body_top=0 (default) = lean anchors only; >0 inlines the
        # top-N bodies. Passed to the MCP child the same way as the retrieval mode.
        env["SG_MCP_BODY_TOP"] = str(body_top)
    elif arm == ARM_CBMEM:
        # Competitor MCP server. Pin the real project slug into the prompt (every
        # cbmem tool needs it) + neutral user prompt (tool guidance is in the
        # append-system prompt, parallel to the SG arm).
        slug = _cbmem_slug(repo)
        cmd += ["--mcp-config", str(repo / ".mcp.json"), "--strict-mcp-config",
                "--append-system-prompt", _CBMEM_APPEND_SYSTEM.format(slug=slug)]
        prompt = _NATIVE_PROMPT.format(issue=issue, scope=_SCOPE_BLOCK)
    elif arm == ARM_SERENA:
        # Competitor MCP server — project is pre-activated via --project in the
        # .mcp.json args, so no per-call slug needed (unlike cbmem).
        cmd += ["--mcp-config", str(repo / ".mcp.json"), "--strict-mcp-config",
                "--append-system-prompt", _SERENA_APPEND_SYSTEM]
        prompt = _NATIVE_PROMPT.format(issue=issue, scope=_SCOPE_BLOCK)
    elif arm == ARM_GITNEXUS:
        # Competitor MCP server — repo auto-selected from launch cwd (this
        # subprocess.run call below sets cwd=str(repo)); repo={name} in the
        # prompt is the explicit fallback (see _GITNEXUS_APPEND_SYSTEM comment).
        cmd += ["--mcp-config", str(repo / ".mcp.json"), "--strict-mcp-config",
                "--append-system-prompt",
                _GITNEXUS_APPEND_SYSTEM.format(name=repo.name)]
        prompt = _NATIVE_PROMPT.format(issue=issue, scope=_SCOPE_BLOCK)
    else:
        # An explicit EMPTY config + --strict-mcp-config ⇒ exactly zero MCP
        # servers (no project or global leakage). Truly Claude-on-its-own.
        cmd += ["--mcp-config", '{"mcpServers":{}}', "--strict-mcp-config"]
        prompt = _NATIVE_PROMPT.format(issue=issue, scope=_SCOPE_BLOCK)
    if disallow_grep:
        cmd += ["--disallowedTools", "Grep", "Glob"]
    try:
        r = subprocess.run(cmd, cwd=str(repo), input=prompt, env=env,
                           capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=timeout)
        raw, exit_code = r.stdout, r.returncode
        timed_out = False
    except subprocess.TimeoutExpired as e:
        raw = (e.stdout or "") if isinstance(e.stdout, str) else ""
        exit_code, timed_out = -1, True

    objs = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            objs.append(json.loads(line))
        except Exception:
            pass
    result = next((o for o in reversed(objs) if o.get("type") == "result"), None)
    ok = (not timed_out) and exit_code == 0 and result is not None \
        and not result.get("is_error", False)
    return {"ok": ok, "exit": exit_code, "timed_out": timed_out,
            "transcript": objs, "result": result, "raw": raw}


def parse_transcript(objs: list, result: dict | None) -> dict:
    """Pull every comparison signal we can from a stream-json transcript:
    turn count, cumulative token usage, cost, PEAK context window, and per-tool
    call counts (split into SG-MCP vs native).

    Token model: Claude Code prompt-caches the re-sent history, so per-message
    `input_tokens` is only the fresh (uncached) slice; the real context the model
    saw each turn is input + cache_read + cache_creation. The PEAK of that across
    turns is the context-window high-water mark — the number to compare against a
    baseline that pastes everything.
    """
    tool_counts: dict = {}
    peak_context = 0
    for o in objs:
        if o.get("type") != "assistant":
            continue
        msg = o.get("message", {}) or {}
        for block in msg.get("content", []) or []:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                name = block.get("name", "?")
                tool_counts[name] = tool_counts.get(name, 0) + 1
        u = msg.get("usage", {}) or {}
        ctx = ((u.get("input_tokens", 0) or 0)
               + (u.get("cache_read_input_tokens", 0) or 0)
               + (u.get("cache_creation_input_tokens", 0) or 0))
        peak_context = max(peak_context, ctx)

    usage = (result or {}).get("usage", {}) or {}
    billed_in = usage.get("input_tokens", 0) or 0
    cache_read = usage.get("cache_read_input_tokens", 0) or 0
    cache_create = usage.get("cache_creation_input_tokens", 0) or 0
    sg_calls = sum(v for k, v in tool_counts.items()
                   if k.startswith("mcp__skeletongraph"))
    n_calls = sum(tool_counts.values())
    return {
        "n_turns": (result or {}).get("num_turns", 0) or 0,
        "billed_input": billed_in,
        "billed_output": usage.get("output_tokens", 0) or 0,
        "cached_input": cache_read,
        "cache_creation_input": cache_create,
        # Total input the model actually processed (fresh + both cache classes).
        "total_input_tokens": billed_in + cache_read + cache_create,
        "peak_context_tokens": peak_context,
        "cost_usd": (result or {}).get("total_cost_usd", 0.0) or 0.0,
        "duration_ms": (result or {}).get("duration_ms", 0) or 0,
        "tool_counts": tool_counts,
        "n_tool_calls": n_calls,
        "sg_tool_calls": sg_calls,
        "native_tool_calls": n_calls - sg_calls,
    }


# ── one task end-to-end ──────────────────────────────────────────────────────

def _model_tag(model: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", model.lower()).strip("_") or "claude"


def _edited_gold(patch: str, gold: list) -> bool:
    touched = {ln[6:].strip().replace("\\", "/") for ln in patch.splitlines()
               if ln.startswith("+++ b/")}
    return bool(touched & {g.replace("\\", "/") for g in gold})


def _patch_metrics(patch: str) -> dict:
    """Unified-diff shape (mirrors run_agent._patch_metrics)."""
    files: set = set()
    added = removed = hunks = 0
    for line in patch.splitlines():
        if line.startswith("+++ b/"):
            files.add(line[6:].strip().replace("\\", "/"))
        elif line.startswith("@@"):
            hunks += 1
        elif line.startswith("+") and not line.startswith("+++"):
            added += 1
        elif line.startswith("-") and not line.startswith("---"):
            removed += 1
    return {"lines_added": added, "lines_removed": removed,
            "files_touched": len(files), "hunks": hunks, "files": sorted(files)}


# ── retrieval metrics from the SG MCP calls in the transcript ────────────────
# aggregate.py computes rec@1/rec@cum/prec from `search_calls`, and funcR@10 from
# turns[].tool_calls[].result. The Claude arm retrieves via MCP sg_search (not the
# harness search_code), so we reconstruct those fields here from the stream-json.

def _tool_result_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(b.get("text", "") for b in content
                         if isinstance(b, dict))
    return ""


def _parse_sg_result_text(text: str):
    """Ordered, de-duped (fqns, files) from one sg_search result blob.

    SG ranks results as `## N. <file::symbol>` headers and `- \\`<file::symbol>\\``
    bullets (Other matches). We pull the FQN from both and derive the file."""
    fqns, files, seen_q, seen_p = [], [], set(), set()
    for line in text.splitlines():
        m = re.match(r"^##\s+\d+\.\s+(\S.+?)\s*$", line) \
            or re.match(r"^-\s+`([^`]+)`", line)
        if not m:
            continue
        fqn = m.group(1).strip()
        path = fqn.split("::", 1)[0].replace("\\", "/").strip()
        if not path:
            continue
        if fqn not in seen_q:
            seen_q.add(fqn); fqns.append(fqn)
        if path not in seen_p:
            seen_p.add(path); files.append(path)
    # Also count files surfaced via the "Module constants" (`NAME = … # path`)
    # and "Lexical matches" (`path:line:`) sections, so file-recall reflects the
    # constants/symbols SG returns — not just the function-graph hits.
    for line in text.splitlines():
        m = re.search(r"#\s+([\w./-]+\.\w+)\s*$", line) \
            or re.match(r"\s*([\w./-]+\.\w+):\d+:", line)
        if not m:
            continue
        p = m.group(1).replace("\\", "/").strip()
        if p and p not in seen_p:
            seen_p.add(p); files.append(p)
    return fqns, files


def _retrieval_from_transcript(objs: list, gold_files: list) -> dict:
    """Reconstruct search_calls + first/all FQN lists + first-search file recall
    metrics from the sg_search calls in a stream-json transcript."""
    gold = {g.replace("\\", "/") for g in gold_files}
    pending: dict = {}        # tool_use_id -> (query, order)
    order = 0
    calls = []                # (order, query, fqns, files)
    for o in objs:
        typ = o.get("type")
        content = (o.get("message", {}) or {}).get("content", []) or []
        if typ == "assistant":
            for b in content:
                if (isinstance(b, dict) and b.get("type") == "tool_use"
                        and str(b.get("name", "")).endswith("sg_search")):
                    pending[b.get("id")] = (
                        (b.get("input") or {}).get("query", ""), order)
                    order += 1
        elif typ == "user":
            for b in content:
                if isinstance(b, dict) and b.get("type") == "tool_result":
                    tid = b.get("tool_use_id")
                    if tid not in pending:
                        continue
                    query, od = pending.pop(tid)
                    fqns, files = _parse_sg_result_text(
                        _tool_result_text(b.get("content")))
                    calls.append((od, query, fqns, files))
    calls.sort(key=lambda c: c[0])

    search_calls, seen_gold = [], set()
    for od, query, fqns, files in calls:
        gih = sorted(gold & set(files))
        seen_gold |= set(gih)
        search_calls.append({
            "turn": od, "query": query, "hits": files, "n_hits": len(files),
            "gold_in_hits": gih,
            "precision": round(len(gih) / len(files), 4) if files else 0.0,
            "cumulative_recall": (round(len(seen_gold) / len(gold), 4)
                                  if gold else 0.0),
            "error": False,
        })
    first_files = calls[0][3] if calls else []
    first_fqns = calls[0][2] if calls else []
    all_fqns, seen = [], set()
    for _, _, fqns, _ in calls:
        for fq in fqns:
            if fq not in seen:
                seen.add(fq); all_fqns.append(fq)
    rank = next((i for i, f in enumerate(first_files, 1) if f in gold), 0)
    n_gold_first = len([f for f in first_files if f in gold])
    return {
        "search_calls": search_calls,
        "first_search_fqns": first_fqns,
        "all_search_fqns": all_fqns,
        "retrieval_hit": bool(gold & set(first_files)),
        "retrieval_precision": (round(n_gold_first / len(first_files), 4)
                                if first_files else 0.0),
        "retrieval_rank": rank,
    }


# ── native-arm retrieval reconstruction ──────────────────────────────────────
# The native arm has no sg_search, so its "retrieval" is whatever its own tools
# surface. Claude Code stream-json attaches a `tool_use_result` to every tool
# call with the STRUCTURED outcome (not just the text the model saw), so this is
# a faithful reconstruction, not a guess:
#   Grep  — files_with_matches mode: `filenames`; content mode: files appear as
#           `path:line:` prefixes in `content`; single-file content grep has
#           neither, so the file IS the input `path` (a non-empty result = a hit).
#   Glob  — `filenames` (a pure file lister; every returned path is a "hit").
#   Read  — `file.filePath` / input `file_path` — the exact file opened.
# Grep+Glob are the SEARCH tools (parallel to sg_search); Read is navigation
# (parallel to sg_get/sg_expand, which SG also excludes from search_calls). So
# retrieval_hit/rank/precision are computed over Grep+Glob only — an apples-to-
# apples "how well did each arm's SEARCH surface gold" — while `reached_gold_via`
# additionally records if native ever touched gold through ANY tool (incl. Read),
# so native's read-driven navigation isn't hidden from the paper.

def _retrieval_from_cbmem_transcript(objs: list, gold_files: list,
                                     repo_root: str) -> dict:
    """Same schema as _retrieval_from_transcript, but for the cbmem competitor
    arm: its search-shaped MCP tools (search_graph, search_code) return raw JSON
    (not SG's markdown), so reuse cbmem's OWN parser (backends.cbmem._extract_files)
    on each tool_result instead of _parse_sg_result_text. get_code_snippet is
    navigation (parallel to SG's sg_expand) — excluded, like Read is for native."""
    from backends.cbmem import _extract_files
    repo = Path(repo_root)
    gold = {g.replace("\\", "/") for g in gold_files}
    pending: dict = {}
    order = 0
    calls = []
    search_tool_names = {"search_graph", "search_code"}
    for o in objs:
        typ = o.get("type")
        content = (o.get("message", {}) or {}).get("content", []) or []
        if typ == "assistant":
            for b in content:
                if not (isinstance(b, dict) and b.get("type") == "tool_use"):
                    continue
                name = str(b.get("name", ""))
                if name.rsplit("__", 1)[-1] in search_tool_names:
                    inp = b.get("input") or {}
                    query = inp.get("query") or inp.get("pattern") or ""
                    pending[b.get("id")] = (query, order)
                    order += 1
        elif typ == "user":
            for b in content:
                if isinstance(b, dict) and b.get("type") == "tool_result":
                    tid = b.get("tool_use_id")
                    if tid not in pending:
                        continue
                    query, od = pending.pop(tid)
                    text = _tool_result_text(b.get("content"))
                    files = _extract_files(text, repo)
                    calls.append((od, query, files))
    calls.sort(key=lambda c: c[0])

    search_calls, seen_gold = [], set()
    for od, query, files in calls:
        gih = sorted(gold & set(files))
        seen_gold |= set(gih)
        search_calls.append({
            "turn": od, "query": query, "hits": files, "n_hits": len(files),
            "gold_in_hits": gih,
            "precision": round(len(gih) / len(files), 4) if files else 0.0,
            "cumulative_recall": (round(len(seen_gold) / len(gold), 4)
                                  if gold else 0.0),
            "error": False,
        })
    first_files = calls[0][2] if calls else []
    rank = next((i for i, f in enumerate(first_files, 1) if f in gold), 0)
    n_gold_first = len([f for f in first_files if f in gold])
    return {
        "search_calls": search_calls,
        "first_search_fqns": first_files,
        "all_search_fqns": [f for _, _, fs in calls for f in fs],
        "retrieval_hit": bool(gold & set(first_files)),
        "retrieval_precision": (round(n_gold_first / len(first_files), 4)
                                if first_files else 0.0),
        "retrieval_rank": rank,
    }


def _retrieval_from_serena_transcript(objs: list, gold_files: list,
                                      repo_root: str) -> dict:
    """Same schema as _retrieval_from_cbmem_transcript, for Serena. Its
    search-shaped tools (find_symbol, search_for_pattern, find_referencing_symbols,
    get_symbols_overview) return a JSON list of dicts with a "relative_path" key
    (confirmed live: [{"name_path": ..., "relative_path": "pkg\\\\math_ops.py", ...}]).
    get_symbols_overview and read_file are navigation (parallel to cbmem's
    get_code_snippet) — excluded from search_calls."""
    gold = {g.replace("\\", "/") for g in gold_files}
    pending: dict = {}
    order = 0
    calls = []
    search_tool_names = {"find_symbol", "search_for_pattern", "find_referencing_symbols"}
    for o in objs:
        typ = o.get("type")
        content = (o.get("message", {}) or {}).get("content", []) or []
        if typ == "assistant":
            for b in content:
                if not (isinstance(b, dict) and b.get("type") == "tool_use"):
                    continue
                name = str(b.get("name", ""))
                if name.rsplit("__", 1)[-1] in search_tool_names:
                    inp = b.get("input") or {}
                    query = (inp.get("name_path_pattern") or inp.get("substring_pattern")
                             or inp.get("name_path") or "")
                    pending[b.get("id")] = (query, order)
                    order += 1
        elif typ == "user":
            for b in content:
                if isinstance(b, dict) and b.get("type") == "tool_result":
                    tid = b.get("tool_use_id")
                    if tid not in pending:
                        continue
                    query, od = pending.pop(tid)
                    text = _tool_result_text(b.get("content"))
                    files = []
                    try:
                        parsed = json.loads(text)
                        items = parsed if isinstance(parsed, list) else [parsed]
                        seen = set()
                        for it in items:
                            if not isinstance(it, dict):
                                continue
                            p = str(it.get("relative_path", "")).replace("\\", "/")
                            if p and p not in seen:
                                seen.add(p); files.append(p)
                    except (json.JSONDecodeError, TypeError):
                        pass
                    calls.append((od, query, files))
    calls.sort(key=lambda c: c[0])

    search_calls, seen_gold = [], set()
    for od, query, files in calls:
        gih = sorted(gold & set(files))
        seen_gold |= set(gih)
        search_calls.append({
            "turn": od, "query": query, "hits": files, "n_hits": len(files),
            "gold_in_hits": gih,
            "precision": round(len(gih) / len(files), 4) if files else 0.0,
            "cumulative_recall": (round(len(seen_gold) / len(gold), 4)
                                  if gold else 0.0),
            "error": False,
        })
    first_files = calls[0][2] if calls else []
    rank = next((i for i, f in enumerate(first_files, 1) if f in gold), 0)
    n_gold_first = len([f for f in first_files if f in gold])
    return {
        "search_calls": search_calls,
        "first_search_fqns": first_files,
        "all_search_fqns": [f for _, _, fs in calls for f in fs],
        "retrieval_hit": bool(gold & set(first_files)),
        "retrieval_precision": (round(n_gold_first / len(first_files), 4)
                                if first_files else 0.0),
        "retrieval_rank": rank,
    }


def _gitnexus_extract_files(text: str) -> list:
    """GitNexus tool results are a JSON object (with trailing '\\n\\n---\\n**Next:**
    ...' guidance text appended after it — raw_decode grabs just the leading
    JSON, ignoring that suffix) whose shape varies by tool (query returns
    `definitions[].filePath`; context/trace/impact nest paths at different
    depths). Rather than hand-code every tool's schema, recursively walk the
    whole parsed object and collect any 'filePath'/'file_path' value — robust
    to the exact nesting confirmed live for `query`, cheap insurance for the
    others."""
    try:
        obj, _ = json.JSONDecoder().raw_decode(text.strip())
    except (json.JSONDecodeError, ValueError):
        return []
    files, seen = [], set()

    def walk(node):
        if isinstance(node, dict):
            for k in ("filePath", "file_path"):
                v = node.get(k)
                if isinstance(v, str) and v:
                    p = v.replace("\\", "/")
                    if p not in seen:
                        seen.add(p); files.append(p)
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for it in node:
                walk(it)

    walk(obj)
    return files


def _retrieval_from_gitnexus_transcript(objs: list, gold_files: list,
                                        repo_root: str) -> dict:
    """Same schema as _retrieval_from_cbmem_transcript, for GitNexus. Search-
    shaped tools are query/context/trace/impact; rename/cypher/check/
    detect_changes/route_map/tool_map/shape_check/api_impact/group_* are
    mutation or structural-audit tools, not localization — excluded, same
    reasoning as cbmem's get_code_snippet / Serena's get_symbols_overview."""
    gold = {g.replace("\\", "/") for g in gold_files}
    pending: dict = {}
    order = 0
    calls = []
    search_tool_names = {"query", "context", "trace", "impact"}
    for o in objs:
        typ = o.get("type")
        content = (o.get("message", {}) or {}).get("content", []) or []
        if typ == "assistant":
            for b in content:
                if not (isinstance(b, dict) and b.get("type") == "tool_use"):
                    continue
                name = str(b.get("name", ""))
                if name.rsplit("__", 1)[-1] in search_tool_names:
                    inp = b.get("input") or {}
                    query = (inp.get("search_query") or inp.get("name")
                             or inp.get("target") or inp.get("from") or "")
                    pending[b.get("id")] = (query, order)
                    order += 1
        elif typ == "user":
            for b in content:
                if isinstance(b, dict) and b.get("type") == "tool_result":
                    tid = b.get("tool_use_id")
                    if tid not in pending:
                        continue
                    query, od = pending.pop(tid)
                    text = _tool_result_text(b.get("content"))
                    files = _gitnexus_extract_files(text)
                    calls.append((od, query, files))
    calls.sort(key=lambda c: c[0])

    search_calls, seen_gold = [], set()
    for od, query, files in calls:
        gih = sorted(gold & set(files))
        seen_gold |= set(gih)
        search_calls.append({
            "turn": od, "query": query, "hits": files, "n_hits": len(files),
            "gold_in_hits": gih,
            "precision": round(len(gih) / len(files), 4) if files else 0.0,
            "cumulative_recall": (round(len(seen_gold) / len(gold), 4)
                                  if gold else 0.0),
            "error": False,
        })
    first_files = calls[0][2] if calls else []
    rank = next((i for i, f in enumerate(first_files, 1) if f in gold), 0)
    n_gold_first = len([f for f in first_files if f in gold])
    return {
        "search_calls": search_calls,
        "first_search_fqns": first_files,
        "all_search_fqns": [f for _, _, fs in calls for f in fs],
        "retrieval_hit": bool(gold & set(first_files)),
        "retrieval_precision": (round(n_gold_first / len(first_files), 4)
                                if first_files else 0.0),
        "retrieval_rank": rank,
    }


def _rel(path: str, repo_root: str) -> str:
    """Normalise a tool path to repo-relative forward-slash form."""
    p = str(path).replace("\\", "/").strip()
    rr = str(repo_root).replace("\\", "/").rstrip("/")
    if rr:
        low_p, low_r = p.lower(), rr.lower()
        i = low_p.find(low_r)
        if i != -1:
            p = p[i + len(rr):].lstrip("/")
    return p


def _grep_hits(inp: dict, tur: dict, repo_root: str) -> list:
    """Repo-relative files a single Grep call surfaced, in first-seen order."""
    files, seen = [], set()

    def _add(p):
        p = _rel(p, repo_root)
        if p and p not in seen:
            seen.add(p); files.append(p)

    fn = tur.get("filenames")
    if isinstance(fn, list) and fn:
        for p in fn:
            _add(p)
        return files
    content = tur.get("content") or ""
    # content-mode over a directory: every match line is `path:line:...` /
    # `path-line-...`. Anchor to line start so we don't catch inline colons.
    got_prefix = False
    for line in content.splitlines():
        m = re.match(r"^([\w./\\+-]+\.\w+)[:-]\d+[:-]", line)
        if m:
            _add(m.group(1)); got_prefix = True
    if got_prefix:
        return files
    # content-mode over a single FILE: no path prefix, but a non-empty result
    # means that specific input file matched → the input path is the hit.
    if content.strip():
        path = inp.get("path") or ""
        if path and re.search(r"\.\w+$", str(path)):
            _add(path)
    return files


def _retrieval_from_native_transcript(objs: list, gold_files: list,
                                      repo_root: str) -> dict:
    """Reconstruct native-arm retrieval (Grep/Glob search + Read navigation)
    into the SAME schema as _retrieval_from_transcript, so aggregate.py scores
    both arms identically. File-level only — native has no function ranking, so
    the function-level metrics (funcR@10/funcHit) are honestly 0 for it."""
    gold = {g.replace("\\", "/") for g in gold_files}
    tid: dict = {}      # tool_use_id -> (name, input, order)
    order = 0
    events = []         # (order, name, input, tool_use_result)
    for o in objs:
        typ = o.get("type")
        if typ == "assistant":
            for b in (o.get("message", {}) or {}).get("content", []) or []:
                if isinstance(b, dict) and b.get("type") == "tool_use":
                    tid[b.get("id")] = (b.get("name"), b.get("input") or {}, order)
                    order += 1
        elif typ == "user":
            tur = o.get("tool_use_result")
            for b in (o.get("message", {}) or {}).get("content", []) or []:
                if isinstance(b, dict) and b.get("type") == "tool_result":
                    nm, inp, od = tid.get(b.get("tool_use_id"), (None, {}, None))
                    if nm is not None and isinstance(tur, dict):
                        events.append((od, nm, inp, tur))
    events.sort(key=lambda e: (e[0] if e[0] is not None else 1e9))

    search_calls, seen_gold = [], set()
    reached, reached_seen = [], set()   # every file touched via ANY tool, in order
    reached_gold_via = None

    def _touch(paths, via):
        nonlocal reached_gold_via
        for p in paths:
            if p not in reached_seen:
                reached_seen.add(p); reached.append(p)
            if p in gold and reached_gold_via is None:
                reached_gold_via = via

    for od, nm, inp, tur in events:
        if nm == "Grep":
            hits = _grep_hits(inp, tur, repo_root)
            query = str(inp.get("pattern", ""))
        elif nm == "Glob":
            fn = tur.get("filenames")
            hits = [_rel(p, repo_root) for p in fn] if isinstance(fn, list) else []
            query = str(inp.get("pattern", ""))
        elif nm == "Read":
            fp = (tur.get("file") or {}).get("filePath") if isinstance(tur.get("file"), dict) else None
            fp = fp or inp.get("file_path") or ""
            _touch([_rel(fp, repo_root)] if fp else [], "Read")
            continue     # navigation, not a search call
        else:
            continue
        # de-dup within a call, preserve order
        seen, ordered = set(), []
        for h in hits:
            if h not in seen:
                seen.add(h); ordered.append(h)
        gih = sorted(gold & set(ordered))
        seen_gold |= set(gih)
        _touch(ordered, nm)
        search_calls.append({
            "turn": od, "query": query, "tool": nm, "hits": ordered,
            "n_hits": len(ordered), "gold_in_hits": gih,
            "precision": round(len(gih) / len(ordered), 4) if ordered else 0.0,
            "cumulative_recall": round(len(seen_gold) / len(gold), 4) if gold else 0.0,
            "error": False,
        })

    first_files = search_calls[0]["hits"] if search_calls else []
    all_files, seen = [], set()
    for sc in search_calls:
        for f in sc["hits"]:
            if f not in seen:
                seen.add(f); all_files.append(f)
    rank = next((i for i, f in enumerate(first_files, 1) if f in gold), 0)
    n_gold_first = len([f for f in first_files if f in gold])
    return {
        "search_calls": search_calls,
        # File paths (not FQNs) — native retrieves at file granularity. aggregate
        # treats these as the localization list; function-match simply won't fire,
        # so funcR@10/funcHit come out 0 for native (honest, not a bug).
        "first_search_fqns": first_files,
        "all_search_fqns": all_files,
        "retrieval_hit": bool(gold & set(first_files)),
        "retrieval_precision": (round(n_gold_first / len(first_files), 4)
                                if first_files else 0.0),
        "retrieval_rank": rank,
        # Transparency beyond the parallel search metric: did native EVER reach a
        # gold file, and through which tool (Grep/Glob = search, Read = it already
        # knew the path from the issue). Not fed to the headline metrics.
        "reached_gold": bool(gold & set(reached)),
        "reached_gold_via": reached_gold_via,
        "reached_rank": next((i for i, f in enumerate(reached, 1) if f in gold), 0),
    }


def run_one_task(task: dict, arm: str, model: str, timeout: int,
                 rebuild: bool = False, disallow_grep: bool = False,
                 keep_transcript: bool = True, body_top: int = 0) -> dict:
    model_tag = _model_tag(model)
    rid = run_id(task["task_id"], arm, 0, model_tag)
    out_path = config.RUNS_DIR / f"{rid}.json"
    config.RUNS_DIR.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    repo = prepare_repo(task, arm, rebuild=rebuild, verbose=True)
    reset_repo(repo)   # guarantee a clean tree even if a prior run left edits

    run = run_claude(repo, task["query"], model, timeout, arm, disallow_grep, body_top)
    patch = extract_patch(repo)
    meta = parse_transcript(run["transcript"], run["result"])
    pm = _patch_metrics(patch)
    # Retrieval metrics for BOTH arm families: SG arms from sg_search, native
    # from its own Grep/Glob/Read (reconstructed at file granularity) — so the
    # paper can compare SG retrieval head-to-head against Claude Code's own.
    gold_files = task.get("gold_files", [])
    if arm == ARM_NATIVE:
        ret = _retrieval_from_native_transcript(run["transcript"], gold_files, str(repo))
    elif arm == ARM_CBMEM:
        ret = _retrieval_from_cbmem_transcript(run["transcript"], gold_files, str(repo))
    elif arm == ARM_SERENA:
        ret = _retrieval_from_serena_transcript(run["transcript"], gold_files, str(repo))
    elif arm == ARM_GITNEXUS:
        ret = _retrieval_from_gitnexus_transcript(run["transcript"], gold_files, str(repo))
    else:
        ret = _retrieval_from_transcript(run["transcript"], gold_files)
    wall = round(time.time() - t0, 1)

    if run["timed_out"]:
        stopped = "timeout"
    elif run["ok"]:
        stopped = "submit"          # verify.py counts submit / max_turns
    else:
        stopped = "error"

    gold = task.get("gold_files", [])
    record = {
        "run_id": rid,
        "task_id": task["task_id"],
        "arm": arm,
        "model": model_tag,
        "model_full": model,
        "repeat": 0,
        "stopped": stopped,
        "harness": "claude-code",     # real-agent arm (SG-MCP or native)
        "disallow_grep": disallow_grep,
        "repo": task.get("repo", ""),
        "base_commit": task.get("base_commit", ""),
        "gold_files": gold,
        "model_patch": patch,
        "edited_gold_file": _edited_gold(patch, gold),
        "n_turns": meta["n_turns"],
        "billed_input": meta["billed_input"],
        "billed_output": meta["billed_output"],
        "cached_input": meta["cached_input"],
        "cache_creation_input": meta["cache_creation_input"],
        "total_input_tokens": meta["total_input_tokens"],
        "peak_context_tokens": meta["peak_context_tokens"],
        "imputed_cost": round(meta["cost_usd"], 6),
        "wall_s": wall,
        "claude_exit": run["exit"],
        # Surface WHY a non-submit run failed (e.g. "Invalid API key · Fix
        # external API key") directly in the record — previously only visible
        # by opening the raw transcript in _claude_transcripts/.
        "error": (None if stopped == "submit" else
                  ("timeout" if run["timed_out"] else
                   (run["result"] or {}).get("result"))),
        "tool_counts": meta["tool_counts"],
        "n_tool_calls": meta["n_tool_calls"],
        "sg_tool_calls": meta["sg_tool_calls"],
        "native_tool_calls": meta["native_tool_calls"],
        # ── retrieval — SG arms from sg_search, native from Grep/Glob/Read ──
        "retrieval_hit": ret["retrieval_hit"],
        "retrieval_precision": ret["retrieval_precision"],
        "retrieval_rank": ret["retrieval_rank"],
        "search_calls": ret["search_calls"],
        "n_search_calls": len(ret["search_calls"]),
        "first_search_fqns": ret["first_search_fqns"],
        "all_search_fqns": ret["all_search_fqns"],
        # native-only transparency (present only when populated) — did native
        # ever reach gold at all, and via which tool (Read = it already knew the
        # path from the issue, not a search win).
        **({"reached_gold": ret["reached_gold"],
            "reached_gold_via": ret["reached_gold_via"],
            "reached_rank": ret["reached_rank"]} if "reached_gold" in ret else {}),
        "files_read": [], "edit_attempts": [],
        # ── patch shape + consolidation (so patch% / patch figures fill) ──
        "patch_lines_added": pm["lines_added"],
        "patch_lines_removed": pm["lines_removed"],
        "patch_files_touched": pm["files_touched"],
        "patch_hunks": pm["hunks"],
        "consolidation": {"files_in_patch_count": pm["files_touched"],
                          "files_read_count": 0, "files_read_and_used_count": 0,
                          "consolidation_gap_files": 0.0},
    }
    out_path.write_text(json.dumps(record, indent=2), encoding="utf-8")

    if keep_transcript and run["raw"]:
        tdir = config.RUNS_DIR / "_claude_transcripts"
        tdir.mkdir(parents=True, exist_ok=True)
        (tdir / f"{rid}.jsonl").write_text(run["raw"], encoding="utf-8")

    try:
        write_index(config.RUNS_DIR)
    except Exception:
        pass   # index is a convenience; never fail a run over it

    return record


def write_index(runs_dir: Path) -> None:
    """Regenerate `_INDEX.md` — a traversable table of every Claude-Code run in
    this results dir, with metrics + clickable paths to each transcript and
    editable repo copy. Rewritten from all JSONs each call, so it is always a
    complete snapshot (race-tolerant across parallel shards)."""
    rows = []
    for p in sorted(runs_dir.glob("*.json")):
        if p.name.startswith("_"):
            continue
        try:
            r = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if r.get("harness") != "claude-code":
            continue
        rows.append(r)
    if not rows:
        return

    copies_root = "C:/Users/ASUS/Desktop/CS/Projects/swebench-data/_claude_repos"
    lines = [
        "# Claude Code runs (SG-MCP vs native) — index",
        "",
        f"_Auto-generated. {len(rows)} run(s). Results dir: `{runs_dir}`._",
        "",
        "- Run JSON (full record): `<run_id>.json` in this dir.",
        "- Stream-json transcript (every message + tool call): "
        "`_claude_transcripts/<run_id>.jsonl`.",
        "- Editable repo copy (inspect the agent's edits): "
        f"`{copies_root}/<task_id>/` — `git -C <copy> diff HEAD` after a run.",
        "",
        "**Columns** — peak_ctx = context-window high-water mark (fresh+cached "
        "input the model saw in one turn); total_in = all input processed; "
        "sg/nat = SkeletonGraph-MCP vs native tool calls.",
        "",
        "| task | arm | stopped | turns | sg | nat | peak_ctx | total_in | out | "
        "cost$ | wall_s | edited_gold | transcript |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    tot_cost = tot_in = tot_out = 0
    for r in rows:
        rid = r.get("run_id", "?")
        tot_cost += r.get("imputed_cost", 0) or 0
        tot_in += r.get("total_input_tokens", 0) or 0
        tot_out += r.get("billed_output", 0) or 0
        lines.append(
            f"| {r.get('task_id','?')} | {r.get('arm','?')} "
            f"| {r.get('stopped','?')} "
            f"| {r.get('n_turns',0)} | {r.get('sg_tool_calls',0)} "
            f"| {r.get('native_tool_calls',0)} "
            f"| {r.get('peak_context_tokens',0):,} "
            f"| {r.get('total_input_tokens',0):,} | {r.get('billed_output',0):,} "
            f"| {r.get('imputed_cost',0):.4f} | {r.get('wall_s',0)} "
            f"| {r.get('edited_gold_file',False)} "
            f"| `_claude_transcripts/{rid}.jsonl` |"
        )
    n = len(rows)
    lines += [
        "",
        f"**Totals** — runs: {n} · cost: ${tot_cost:.2f} · "
        f"total input: {tot_in:,} · output: {tot_out:,} · "
        f"mean cost/run: ${tot_cost / n:.4f} · "
        f"mean peak_ctx: {sum(x.get('peak_context_tokens',0) for x in rows)//n:,}",
        "",
    ]
    (runs_dir / "_INDEX.md").write_text("\n".join(lines), encoding="utf-8")


def _already_done(task: dict, arm: str, model_tag: str) -> bool:
    rid = run_id(task["task_id"], arm, 0, model_tag)
    p = config.RUNS_DIR / f"{rid}.json"
    if not p.exists():
        return False
    try:
        rec = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return False
    return rec.get("stopped") in ("submit", "max_turns")


def reprocess_retrieval(runs_dir: Path) -> None:
    """Re-derive retrieval metrics for EVERY saved run from its transcript, in
    place — no agent rerun. Backfills native-arm retrieval (Grep/Glob/Read) onto
    runs recorded before that extractor existed, and refreshes SG-arm retrieval
    if the parser improved. Reads gold_files from the run JSON and the repo root
    from the transcript's own `cwd`, so it needs nothing but this results dir."""
    tdir = runs_dir / "_claude_transcripts"
    patched = 0
    for p in sorted(runs_dir.glob("*.json")):
        if p.name.startswith("_"):
            continue
        try:
            rec = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if rec.get("harness") != "claude-code":
            continue
        tf = tdir / f"{rec['run_id']}.jsonl"
        if not tf.exists():
            print(f"  skip {rec['run_id']}: no transcript")
            continue
        objs = []
        for line in tf.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if line:
                try:
                    objs.append(json.loads(line))
                except Exception:
                    pass
        gold = rec.get("gold_files", [])
        if rec.get("arm") == ARM_NATIVE:
            cwd = next((o.get("cwd") for o in objs
                        if o.get("type") == "system" and o.get("cwd")), "")
            ret = _retrieval_from_native_transcript(objs, gold, cwd)
        elif rec.get("arm") == ARM_CBMEM:
            cwd = next((o.get("cwd") for o in objs
                        if o.get("type") == "system" and o.get("cwd")), "")
            ret = _retrieval_from_cbmem_transcript(objs, gold, cwd)
        elif rec.get("arm") == ARM_SERENA:
            cwd = next((o.get("cwd") for o in objs
                        if o.get("type") == "system" and o.get("cwd")), "")
            ret = _retrieval_from_serena_transcript(objs, gold, cwd)
        elif rec.get("arm") == ARM_GITNEXUS:
            cwd = next((o.get("cwd") for o in objs
                        if o.get("type") == "system" and o.get("cwd")), "")
            ret = _retrieval_from_gitnexus_transcript(objs, gold, cwd)
        else:
            ret = _retrieval_from_transcript(objs, gold)
        rec.update({
            "retrieval_hit": ret["retrieval_hit"],
            "retrieval_precision": ret["retrieval_precision"],
            "retrieval_rank": ret["retrieval_rank"],
            "search_calls": ret["search_calls"],
            "n_search_calls": len(ret["search_calls"]),
            "first_search_fqns": ret["first_search_fqns"],
            "all_search_fqns": ret["all_search_fqns"],
        })
        if "reached_gold" in ret:
            rec["reached_gold"] = ret["reached_gold"]
            rec["reached_gold_via"] = ret["reached_gold_via"]
            rec["reached_rank"] = ret["reached_rank"]
        p.write_text(json.dumps(rec, indent=2), encoding="utf-8")
        patched += 1
        print(f"  {rec['run_id']}: {rec['arm']:9} hit={ret['retrieval_hit']} "
              f"rank={ret['retrieval_rank']} n_search={len(ret['search_calls'])}"
              + (f" reached_via={ret.get('reached_gold_via')}" if "reached_gold" in ret else ""))
    print(f"Reprocessed {patched} run(s) in {runs_dir}. "
          f"Re-run `python -m eval.agent.aggregate` to refresh the table.")


def _parse_shard(shard: str):
    if not shard:
        return None
    try:
        k, n = (int(x) for x in shard.split("/"))
        if not (1 <= k <= n):
            raise ValueError
        return (k, n)
    except Exception:
        raise SystemExit(f"--shard must be 'k/N' with 1<=k<=N (got {shard!r})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="", help="tasks jsonl (e.g. swebench_100.jsonl); not needed with --reprocess")
    ap.add_argument("--reprocess", action="store_true",
                    help="re-derive retrieval metrics for all runs in the results "
                         "dir from their saved transcripts (backfills native-arm "
                         "retrieval), then exit. No agent rerun.")
    ap.add_argument("--arm", default=ARM_SG, choices=list(ARMS),
                    help="sg-rerank = Claude + SkeletonGraph MCP pinned to BM25+"
                         "structural rerank (default, no dense leg); "
                         "sg-fusion = same MCP pinned to 3-way RRF incl. dense; "
                         "native = Claude on its own, no SG (the baseline).")
    ap.add_argument("--disallow-grep", action="store_true",
                    help="block native Grep/Glob so search goes through SG "
                         "(SG-as-sole-retrieval). Only meaningful for sg-rerank/sg-fusion.")
    ap.add_argument("--body-top", type=int, default=0, dest="body_top",
                    help="SG_MCP_BODY_TOP — sg_search inlines the top-N function "
                         "bodies (capped). 0 (default) = lean anchors only, the "
                         "agent pulls bodies via sg_expand. Use 1 for the "
                         "lean+rank1 A/B; the default lean is what ships.")
    ap.add_argument("--model", default="sonnet",
                    help="Claude model: alias ('sonnet'/'opus') or full id "
                         "(default: sonnet)")
    ap.add_argument("--shard", default="",
                    help="'k/N' — run only the k-th of N strided task shards "
                         "(1-based). Run the SAME command in N terminals.")
    ap.add_argument("--limit", type=int, default=0, help="first N tasks only")
    ap.add_argument("--workers", type=int, default=1,
                    help="concurrent claude processes IN THIS terminal (default 1; "
                         "raise only if you want one terminal to drive several)")
    ap.add_argument("--timeout", type=int, default=1200,
                    help="seconds per task before killing claude (default 1200)")
    ap.add_argument("--force", action="store_true", help="re-run completed tasks")
    ap.add_argument("--rebuild", action="store_true",
                    help="wipe + rebuild each editable copy (re-copy, re-index)")
    ap.add_argument("--prepare-only", action="store_true",
                    help="only stage the editable copies (copy+index+install); "
                         "run no agent. Use once up front to pre-copy all repos.")
    args = ap.parse_args()

    if args.reprocess:
        reprocess_retrieval(config.RUNS_DIR)
        return
    if not args.dataset:
        raise SystemExit("--dataset is required (unless --reprocess)")

    tasks = load_tasks(Path(args.dataset))
    if args.limit > 0:
        tasks = tasks[:args.limit]
    shard = _parse_shard(args.shard)
    if shard:
        k, n = shard
        tasks = tasks[k - 1::n]

    arm = args.arm
    model_tag = _model_tag(args.model)
    if not args.force and not args.prepare_only:
        tasks = [t for t in tasks if not _already_done(t, arm, model_tag)]

    grep_note = " | grep DISALLOWED" if args.disallow_grep else ""
    print(f"Claude Code [{arm}] | model={args.model}{grep_note} "
          f"| tag={config._RUN_TAG or '(none)'} | {len(tasks)} tasks "
          f"| workers={args.workers}")
    print(f"  results -> {config.RUNS_DIR}")
    if shard:
        print(f"  shard {shard[0]}/{shard[1]} (strided)")
    if not tasks:
        print("  nothing to do (all done) — run verify + aggregate")
        return

    if args.prepare_only:
        for i, t in enumerate(tasks, 1):
            try:
                prepare_repo(t, arm, rebuild=args.rebuild, verbose=True)
                print(f"  [{i}/{len(tasks)}] prepared {t['task_id']}")
            except Exception as e:
                print(f"  [{i}/{len(tasks)}] PREPARE FAILED {t['task_id']}: "
                      f"{type(e).__name__}: {e}")
        print("Prepared. Drop --prepare-only to run the agent.")
        return

    t0 = time.time()
    done = fail = 0
    if args.workers <= 1:
        for t in tasks:
            try:
                rec = run_one_task(t, arm, args.model, args.timeout,
                                   args.rebuild, args.disallow_grep, body_top=args.body_top)
                done += 1
                _report(done + fail, len(tasks), rec)
            except Exception as e:
                fail += 1
                print(f"  [{done+fail}/{len(tasks)}] {t['task_id']} FAILED: "
                      f"{type(e).__name__}: {e}")
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = {pool.submit(run_one_task, t, arm, args.model, args.timeout,
                                args.rebuild, args.disallow_grep,
                                body_top=args.body_top): t
                    for t in tasks}
            for fut in as_completed(futs):
                t = futs[fut]
                try:
                    rec = fut.result()
                    done += 1
                    _report(done + fail, len(tasks), rec)
                except Exception as e:
                    fail += 1
                    print(f"  [{done+fail}/{len(tasks)}] {t['task_id']} FAILED: "
                          f"{type(e).__name__}: {e}")

    mins = (time.time() - t0) / 60
    print(f"\nDone: {done} ok, {fail} failed, {mins:.1f} min. Results in {config.RUNS_DIR}")
    print(f"Next:  python -m eval.agent.verify --all --only-arms {arm} "
          f"--run-tag <harness_tag>")
    print( "Then:  python -m eval.agent.aggregate")


def _report(n: int, total: int, rec: dict) -> None:
    tools = rec.get("tool_counts", {})
    sg_calls = sum(v for k, v in tools.items() if k.startswith("mcp__skeletongraph"))
    print(f"  [{n}/{total}] {rec['run_id']}: {rec['stopped']} "
          f"turns={rec['n_turns']} sg_calls={sg_calls} "
          f"edited_gold={rec['edited_gold_file']} "
          f"in={rec['billed_input']} out={rec['billed_output']} "
          f"${rec['imputed_cost']} {rec['wall_s']}s")


if __name__ == "__main__":
    main()
