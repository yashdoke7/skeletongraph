"""Decision log — durable memory of *why* changes were made.

Distinct from the tool-turn log (`session/log.py`, which records *what* files a
turn touched). The decision log captures rationale: "we chose X over Y because
Z". It exists so a decision made early in a project is not lost once it scrolls
out of the recent-turns window — the model can pull it back when planning.

Design contract (see temp/PIPELINE_REWIRE_PLAN.md):
  - append-only JSONL at .skeletongraph/decisions.jsonl
  - the model RECORDS a decision via a tool (invited, never forced)
  - the model READS it pull-only, topic-filtered — never pushed, never a dump
  - surfaced as "available" only in planning/architecture/explain modes

This module is pure storage + retrieval; wiring into MCP tools is separate.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

_LOG_NAME = "decisions.jsonl"


@dataclass
class Decision:
    """One recorded decision."""
    ts: float
    summary: str                              # one-line: what was decided
    rationale: str = ""                       # why — the part that ages well
    files: List[str] = field(default_factory=list)    # files it concerns
    topics: List[str] = field(default_factory=list)   # tags for retrieval

    def to_dict(self) -> dict:
        return {
            "ts": self.ts,
            "summary": self.summary,
            "rationale": self.rationale,
            "files": self.files,
            "topics": self.topics,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Decision":
        return cls(
            ts=float(d.get("ts", 0.0)),
            summary=str(d.get("summary", "")),
            rationale=str(d.get("rationale", "")),
            files=list(d.get("files", []) or []),
            topics=[t.lower() for t in (d.get("topics", []) or [])],
        )


def _log_path(sg_dir: Path) -> Path:
    return Path(sg_dir) / _LOG_NAME


# ── write ───────────────────────────────────────────────────────────────────


def record_decision(
    sg_dir: Path,
    summary: str,
    rationale: str = "",
    files: Optional[List[str]] = None,
    topics: Optional[List[str]] = None,
) -> bool:
    """Append one decision. Best-effort — never raises into the caller."""
    summary = (summary or "").strip()
    if not summary:
        return False
    sg_dir = Path(sg_dir)
    try:
        sg_dir.mkdir(parents=True, exist_ok=True)
        entry = Decision(
            ts=time.time(),
            summary=summary,
            rationale=(rationale or "").strip(),
            files=list(files or []),
            topics=[t.strip().lower() for t in (topics or []) if t.strip()],
        )
        with _log_path(sg_dir).open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry.to_dict()) + "\n")
        return True
    except Exception:
        return False


# ── read ────────────────────────────────────────────────────────────────────


def _load_all(sg_dir: Path) -> List[Decision]:
    path = _log_path(sg_dir)
    if not path.exists():
        return []
    out: List[Decision] = []
    try:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if line:
                try:
                    out.append(Decision.from_dict(json.loads(line)))
                except (json.JSONDecodeError, ValueError):
                    continue
    except Exception:
        return []
    return out


def query_decisions(
    sg_dir: Path,
    topic: Optional[str] = None,
    limit: int = 8,
) -> List[Decision]:
    """Return recent decisions, optionally filtered by topic.

    Topic match is a case-insensitive substring test against each decision's
    `topics` tags, its summary, and its rationale — so the model can ask for a
    concept ("auth", "retrieval ranking") without knowing the exact tag.
    Newest first. Pull-only: callers decide when this is worth the tokens.
    """
    decisions = _load_all(sg_dir)
    if topic:
        t = topic.strip().lower()
        decisions = [
            d for d in decisions
            if any(t in tag for tag in d.topics)
            or t in d.summary.lower()
            or t in d.rationale.lower()
        ]
    decisions.sort(key=lambda d: d.ts, reverse=True)
    return decisions[:max(0, limit)]


def format_decisions(decisions: List[Decision]) -> str:
    """Render decisions as a compact digest for injection into context."""
    if not decisions:
        return ""
    lines = ["## Prior decisions"]
    for d in decisions:
        line = f"- {d.summary}"
        if d.rationale:
            line += f" — {d.rationale}"
        if d.files:
            line += f"  [{', '.join(d.files[:3])}]"
        lines.append(line)
    return "\n".join(lines)


def decision_count(sg_dir: Path) -> int:
    """Cheap count — for deciding whether to even offer the log to the model."""
    path = _log_path(sg_dir)
    if not path.exists():
        return 0
    try:
        return sum(1 for line in
                   path.read_text(encoding="utf-8", errors="replace").splitlines()
                   if line.strip())
    except Exception:
        return 0
