"""Post-turn summary queue — non-blocking lazy generation.

Retrieved functions with missing or stale summaries are written to
.skeletongraph/summary_queue.jsonl. A daemon thread drains the queue
after each PostToolUse hook, trying:

  Tier-0.5  Ollama local LLM  (if running, config.enable_local_summary=True)
  Tier-1    Cloud LLM         (if config.auto_summarize_on_update=True and key set)
  Tier-0    Local heuristic   (always available, no API needed)

The current query is NEVER blocked. Stale summaries are served
immediately; the queue improves quality over time, turn by turn.

Queue file format — one JSON object per line:
  {
    "fqn": "src/foo.py::Bar::baz",
    "file_path": "src/foo.py",
    "line_start": 10,
    "line_end": 40,
    "signature": "def baz(self) -> str:",
    "enqueued_at": 1234567890.123
  }
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional

if TYPE_CHECKING:
    from ..config import SGConfig
    from ..parser.skeleton import SkeletonCore

_QUEUE_FILE = "summary_queue.jsonl"
_LOCK_FILE = "summary_drain.lock"
_LOCK_MAX_AGE_SECS = 120  # treat locks older than this as stale


# ── Enqueue ──────────────────────────────────────────────────────────────


def enqueue_for_summary(sg_dir: Path, sk: "SkeletonCore") -> None:
    """Append a function skeleton to the summary queue.

    Best-effort: never raises, safe to call from hot paths.
    Deduplication happens at drain time (last write per FQN wins).
    """
    try:
        queue_path = sg_dir / _QUEUE_FILE
        entry = json.dumps({
            "fqn": sk.fqn,
            "file_path": sk.file_path,
            "line_start": getattr(sk, "line_start", 0),
            "line_end": getattr(sk, "line_end", 0),
            "signature": getattr(sk, "signature", ""),
            "enqueued_at": time.time(),
        })
        # Append mode: each line is independent — safe for concurrent appends
        with queue_path.open("a", encoding="utf-8") as f:
            f.write(entry + "\n")
    except Exception:
        pass


def queue_size(sg_dir: Path) -> int:
    """Return number of pending entries (unique FQNs). Returns 0 on error."""
    queue_path = sg_dir / _QUEUE_FILE
    if not queue_path.exists():
        return 0
    try:
        seen = set()
        for line in queue_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                seen.add(entry["fqn"])
            except (json.JSONDecodeError, KeyError):
                continue
        return len(seen)
    except Exception:
        return 0


# ── Background drain ─────────────────────────────────────────────────────


def drain_queue_background(project_root: Path, config: "SGConfig") -> None:
    """Spawn a daemon thread to drain the queue asynchronously.

    Returns immediately. The daemon thread dies with the process.
    Concurrent calls are deduplicated via a lock file.
    """
    t = threading.Thread(
        target=_drain_queue_safe,
        args=(project_root, config),
        daemon=True,
        name="sg-summary-drain",
    )
    t.start()


# ── Internal ─────────────────────────────────────────────────────────────


def _drain_queue_safe(project_root: Path, config: "SGConfig") -> None:
    """Thread target: acquire lock, drain, release lock."""
    sg_dir = project_root / ".skeletongraph"
    lock_path = sg_dir / _LOCK_FILE

    try:
        # Honour existing lock unless it's stale
        if lock_path.exists():
            try:
                lock_age = time.time() - lock_path.stat().st_mtime
                if lock_age < _LOCK_MAX_AGE_SECS:
                    return  # Another drain is running
            except OSError:
                pass
            try:
                lock_path.unlink(missing_ok=True)
            except Exception:
                return  # Can't remove stale lock — give up

        # Acquire
        try:
            lock_path.write_text(str(os.getpid()), encoding="utf-8")
        except Exception:
            return

        try:
            _drain_queue(sg_dir, project_root, config)
        finally:
            try:
                lock_path.unlink(missing_ok=True)
            except Exception:
                pass

    except Exception:
        try:
            lock_path.unlink(missing_ok=True)
        except Exception:
            pass


def _drain_queue(sg_dir: Path, project_root: Path, config: "SGConfig") -> None:
    """Read the queue, generate summaries for up to max_batch entries."""
    queue_path = sg_dir / _QUEUE_FILE
    if not queue_path.exists():
        return

    # Read all entries, dedup by FQN (last occurrence wins)
    entries_by_fqn: Dict[str, dict] = {}
    try:
        raw = queue_path.read_text(encoding="utf-8")
    except Exception:
        return

    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
            fqn = entry["fqn"]
            entries_by_fqn[fqn] = entry
        except (json.JSONDecodeError, KeyError):
            continue

    if not entries_by_fqn:
        try:
            queue_path.unlink(missing_ok=True)
        except Exception:
            pass
        return

    max_batch: int = getattr(config, "summary_queue_max_batch", 10)
    all_entries = list(entries_by_fqn.values())
    to_process = all_entries[:max_batch]
    # Entries beyond max_batch remain in queue for next drain
    leftover_fqns = {e["fqn"] for e in all_entries[max_batch:]}

    # Load summary store
    from .summary_store import SummaryStore
    summary_store = SummaryStore.load(sg_dir)

    # ── Determine which tier is available ────────────────────────────────
    ollama_available = False
    ollama_model: str = getattr(config, "ollama_summary_model", "qwen2.5-coder:1.5b")
    ollama_base: str = getattr(config, "ollama_base_url", "http://localhost:11434")
    ollama_timeout: int = getattr(config, "ollama_timeout", 15)
    enable_local: bool = getattr(config, "enable_local_summary", True)

    if enable_local:
        try:
            from .ollama import is_ollama_available
            ollama_available = is_ollama_available(ollama_base)
        except Exception:
            pass

    cloud_enabled: bool = getattr(config, "auto_summarize_on_update", False)

    # ── Process batch ────────────────────────────────────────────────────
    processed_fqns: List[str] = []

    for entry in to_process:
        fqn = entry["fqn"]
        file_path: str = entry.get("file_path", "")
        line_start: int = entry.get("line_start", 0)
        line_end: int = entry.get("line_end", 0)
        signature: str = entry.get("signature", "")

        body = _read_body(project_root, file_path, line_start, line_end)
        summary: Optional[str] = None

        # ── Tier-0.5: Ollama ─────────────────────────────────────────────
        if ollama_available and body:
            try:
                from .ollama import generate_summary_ollama
                summary = generate_summary_ollama(
                    fqn=fqn,
                    signature=signature,
                    body=body,
                    model=ollama_model,
                    base_url=ollama_base,
                    timeout=ollama_timeout,
                )
            except Exception:
                pass

        # ── Tier-1: Cloud LLM ────────────────────────────────────────────
        if not summary and cloud_enabled and body:
            try:
                from ..llm.provider import LLMConfig, complete
                from ..llm.summarizer import (
                    SUMMARIZE_SYSTEM,
                    SUMMARIZE_PROMPT,
                    _clean_summary as _cloud_clean,
                )
                prompt_text = SUMMARIZE_PROMPT.format(
                    fqn=fqn,
                    signature=signature,
                    decorators="",
                    body=body[:1800] if len(body) > 1800 else body,
                )
                llm_cfg = LLMConfig(
                    model=getattr(config, "summary_model", "gemini/gemini-2.5-flash"),
                    max_tokens=80,
                    timeout=20,
                )
                resp = complete(prompt_text, system=SUMMARIZE_SYSTEM, config=llm_cfg)
                summary = _cloud_clean(resp.text) or None
            except Exception:
                pass

        # ── Tier-0: Local heuristic fallback ─────────────────────────────
        if not summary:
            summary = _heuristic_fallback(fqn, signature)

        if summary:
            summary_store.set(fqn, summary)
            summary_store.clear_pending(fqn)
            processed_fqns.append(fqn)

    # ── Persist summaries ────────────────────────────────────────────────
    if processed_fqns:
        try:
            summary_store.save(sg_dir)
        except Exception:
            pass

    # ── Rewrite queue ────────────────────────────────────────────────────
    # Keep: entries we couldn't process (leftover beyond max_batch)
    # Remove: everything we processed (regardless of success)
    remaining = [
        e for fqn, e in entries_by_fqn.items()
        if fqn not in processed_fqns and fqn in leftover_fqns
    ]

    try:
        if remaining:
            lines = [json.dumps(e) for e in remaining]
            queue_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        else:
            queue_path.unlink(missing_ok=True)
    except Exception:
        pass


def _read_body(
    project_root: Path,
    file_path: str,
    line_start: int,
    line_end: int,
) -> str:
    """Read function body lines from disk. Returns '' on any error."""
    if not file_path or not line_start:
        return ""
    try:
        full_path = project_root / file_path
        if not full_path.exists():
            return ""
        lines = full_path.read_text(encoding="utf-8", errors="replace").splitlines()
        start = max(0, line_start - 1)
        end = min(len(lines), line_end if line_end else len(lines))
        return "\n".join(lines[start:end])
    except Exception:
        return ""


def _heuristic_fallback(fqn: str, signature: str) -> str:
    """Tier-0 fallback: derive a minimal summary from name tokens."""
    name = fqn.split("::")[-1].split(".")[-1]
    try:
        from ..graph.inverted_index import tokenize_identifier
        tokens = tokenize_identifier(name)
        if tokens:
            return " ".join(tokens).capitalize() + "."
    except Exception:
        pass
    return f"{name}."
