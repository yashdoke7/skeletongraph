"""
Hierarchical session memory: cross-session project knowledge.

Structure:
  .skeletongraph/session/
  ├── current.md        (~100-250 tokens, this session's turns)
  ├── recent.md         (~300-500 tokens, last 3-5 sessions, key decisions)
  └── project_log.md    (~100-200 tokens, milestone history, 1 sentence each)

  .skeletongraph/domain/
  └── [name].md         (~150-250 tokens each, per-domain decisions)

Separate from retrieval/session.py which handles turn-level FQN dedup.
This module handles project-level knowledge that survives across sessions.

Post-processing is rule-based (no LLM). Captures:
  - File paths mentioned in agent responses
  - Function names modified
  - Test results (pass/fail)
  - Decision statements ("decided to...", "using X because...")
"""

from __future__ import annotations

import json
import re
import time
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)

# Token budget per file
_BUDGET_CURRENT = 250      # tokens
_BUDGET_RECENT = 500       # tokens
_BUDGET_PROJECT_LOG = 200  # tokens
_BUDGET_DOMAIN = 250       # tokens per domain

# ~4 chars per token estimate
_CHARS_PER_TOKEN = 4

# Patterns for extracting decisions from agent responses
_DECISION_PATTERNS = [
    re.compile(r"(?:decided|decision|choosing|chose|went with|going with|using)\s+(.{10,80})", re.I),
    re.compile(r"(?:because|reason:|rationale:)\s+(.{10,80})", re.I),
    re.compile(r"(?:will not|won't|avoiding|instead of)\s+(.{10,80})", re.I),
]

# Patterns for test results
_TEST_PATTERNS = [
    re.compile(r"(?:tests?\s+(?:pass|passed|passing|succeed|green))", re.I),
    re.compile(r"(?:tests?\s+(?:fail|failed|failing|red|broken))", re.I),
    re.compile(r"(\d+)\s+(?:pass|passed).*?(\d+)\s+(?:fail|failed)", re.I),
]

# File path pattern
_FILE_PATH_PATTERN = re.compile(r'[\w./\\-]+\.(?:py|js|ts|tsx|jsx|java|go|rs|cpp|cs|rb|php)')


@dataclass
class TurnEntry:
    """A single turn record in current.md."""
    turn_number: int
    summary: str
    files_modified: List[str] = field(default_factory=list)
    decisions: List[str] = field(default_factory=list)
    test_result: Optional[str] = None  # "pass", "fail", or None


@dataclass
class SessionEntry:
    """A session summary in recent.md."""
    date: str
    title: str
    decisions: List[str] = field(default_factory=list)
    files: List[str] = field(default_factory=list)
    status: str = "complete"


class SessionMemory:
    """Hierarchical session memory manager.

    Interaction flow (per turn):
      1. SessionMemory.load(sg_dir)       → reads files from disk
      2. prompt_builder uses .get_L3()    → returns text for assembly
      3. [After agent responds]
      4. SessionMemory.post_process()     → updates current.md, tags domain
      5. [On session end]
      6. SessionMemory.compress()         → moves current→recent→project_log
    """

    def __init__(self, sg_dir: Path) -> None:
        self._sg_dir = sg_dir
        self._session_dir = sg_dir / "session"
        self._domain_dir = sg_dir / "domain"
        self._current_path = self._session_dir / "current.md"
        self._recent_path = self._session_dir / "recent.md"
        self._log_path = self._session_dir / "project_log.md"
        self._turn_count = 0
        self._session_title = ""
        self._decisions_this_session: List[str] = []
        self._files_this_session: Set[str] = set()

    @classmethod
    def load(cls, sg_dir: Path) -> "SessionMemory":
        """Load session memory from disk."""
        mem = cls(sg_dir)
        mem._session_dir.mkdir(parents=True, exist_ok=True)
        mem._domain_dir.mkdir(parents=True, exist_ok=True)

        # Count existing turns in current.md
        if mem._current_path.exists():
            text = mem._current_path.read_text(encoding="utf-8", errors="replace")
            mem._turn_count = text.count("\n## Turn ")
            # Extract files mentioned
            for match in _FILE_PATH_PATTERN.finditer(text):
                mem._files_this_session.add(match.group())

        return mem

    def get_current_text(self) -> str:
        """Get current.md content for L3 assembly."""
        if self._current_path.exists():
            return self._current_path.read_text(encoding="utf-8", errors="replace").strip()
        return ""

    def get_recent_text(self) -> str:
        """Get recent.md content for L3 assembly."""
        if self._recent_path.exists():
            return self._recent_path.read_text(encoding="utf-8", errors="replace").strip()
        return ""

    def get_project_log_text(self) -> str:
        """Get project_log.md content for L3 assembly."""
        if self._log_path.exists():
            return self._log_path.read_text(encoding="utf-8", errors="replace").strip()
        return ""

    def get_domain_text(self, domain_name: str) -> str:
        """Get a specific domain note."""
        path = self._domain_dir / f"{domain_name}.md"
        if path.exists():
            return path.read_text(encoding="utf-8", errors="replace").strip()
        return ""

    def list_domains(self) -> List[str]:
        """List available domain note names."""
        if not self._domain_dir.exists():
            return []
        return [p.stem for p in self._domain_dir.glob("*.md")]

    # ── Post-Processing ──────────────────────────────────────────────────

    def post_process(
        self,
        prompt: str,
        agent_response: str,
        files_modified: Optional[List[str]] = None,
    ) -> None:
        """Process an agent turn and update session memory.

        Called after each turn. Rule-based extraction, no LLM.

        Args:
            prompt: The user's prompt for this turn.
            agent_response: The agent's full response text.
            files_modified: Optional explicit list of modified files.
        """
        self._turn_count += 1

        # Extract information from response
        extracted_files = files_modified or []
        if not extracted_files:
            extracted_files = list({m.group() for m in _FILE_PATH_PATTERN.finditer(agent_response)})

        decisions = _extract_decisions(agent_response)
        test_result = _extract_test_result(agent_response)

        # Update tracking
        self._files_this_session.update(extracted_files)
        self._decisions_this_session.extend(decisions)

        # Build turn summary
        summary_parts = []
        if extracted_files:
            summary_parts.append(f"Modified: {', '.join(extracted_files[:5])}")
        if decisions:
            summary_parts.append(f"Decision: {decisions[0]}")
        if test_result:
            summary_parts.append(f"Tests: {test_result}")
        if not summary_parts:
            # Fallback: first 80 chars of prompt
            summary_parts.append(prompt[:80].strip())

        summary = "; ".join(summary_parts)

        # Append to current.md
        turn_text = f"\n## Turn {self._turn_count}\n{summary}\n"
        if not self._session_title and self._turn_count == 1:
            self._session_title = prompt[:60].strip()

        self._session_dir.mkdir(parents=True, exist_ok=True)

        # Write/append current.md
        header = ""
        if self._turn_count == 1:
            header = (
                f"# Current Session\n"
                f"**Started:** {time.strftime('%Y-%m-%d %H:%M')}\n"
                f"**Task:** {self._session_title}\n"
            )

        if self._current_path.exists() and self._turn_count > 1:
            existing = self._current_path.read_text(encoding="utf-8", errors="replace")
            # Enforce budget — keep last N turns if over
            new_text = existing + turn_text
            if len(new_text) > _BUDGET_CURRENT * _CHARS_PER_TOKEN:
                lines = new_text.split("\n## Turn ")
                # Keep header + last 3 turns
                if len(lines) > 4:
                    kept = lines[0] + "\n## Turn " + "\n## Turn ".join(lines[-3:])
                    new_text = kept
            self._current_path.write_text(new_text, encoding="utf-8")
        else:
            self._current_path.write_text(header + turn_text, encoding="utf-8")

        # Tag decisions to domain notes
        if decisions:
            self._tag_domain_decisions(decisions, extracted_files)

    def _tag_domain_decisions(self, decisions: List[str], files: List[str]) -> None:
        """Tag decisions to relevant domain notes based on file paths."""
        if not files:
            return

        for file_path in files:
            parts = file_path.replace("\\", "/").split("/")
            for part in parts[:-1]:  # directory components
                domain_path = self._domain_dir / f"{part}.md"
                if domain_path.exists():
                    # Append decisions to existing domain note
                    existing = domain_path.read_text(encoding="utf-8", errors="replace")
                    addition = "\n\n## Recent Changes\n"
                    for d in decisions[:2]:
                        addition += f"- {time.strftime('%Y-%m-%d')}: {d}\n"

                    new_text = existing + addition
                    # Enforce budget
                    if len(new_text) > _BUDGET_DOMAIN * _CHARS_PER_TOKEN:
                        # Trim from middle (keep header + recent)
                        lines = new_text.splitlines()
                        if len(lines) > 15:
                            new_text = "\n".join(lines[:8] + ["...", ""] + lines[-6:])

                    domain_path.write_text(new_text, encoding="utf-8")
                    break  # Only tag first matching domain

    # ── Compression ──────────────────────────────────────────────────────

    def compress(self) -> None:
        """Compress session memory on session end.

        Flow:
          1. current.md has decisions? → summarize into recent.md
          2. current.md has no decisions? → discard
          3. recent.md > 500 tokens? → move oldest to project_log.md
          4. Clear current.md

        MUST be called at: MCP server shutdown, `sg session end` CLI, TTL expiry.
        Without this, recent.md grows unboundedly.
        """
        if not self._current_path.exists():
            return

        current_text = self._current_path.read_text(encoding="utf-8", errors="replace")

        # Check if session had decisions
        if self._decisions_this_session:
            # Summarize into recent.md
            date = time.strftime("%Y-%m-%d")
            title = self._session_title or "Session"
            files_str = ", ".join(sorted(self._files_this_session)[:5])

            entry = (
                f"\n## Session {date}: {title}\n"
                f"- Files: {files_str}\n"
            )
            for d in self._decisions_this_session[:3]:
                entry += f"- Decision: {d}\n"
            entry += f"- Status: complete\n"

            # Append to recent.md
            recent_text = ""
            if self._recent_path.exists():
                recent_text = self._recent_path.read_text(encoding="utf-8", errors="replace")

            if not recent_text:
                recent_text = "# Recent Decisions\n"

            recent_text += entry
            self._recent_path.write_text(recent_text, encoding="utf-8")

            # Check recent.md overflow → compress oldest to project_log.md
            if len(recent_text) > _BUDGET_RECENT * _CHARS_PER_TOKEN:
                self._overflow_to_log(recent_text)

        # Clear current.md
        self._current_path.write_text("", encoding="utf-8")
        self._turn_count = 0
        self._decisions_this_session.clear()
        self._files_this_session.clear()
        self._session_title = ""

    def _overflow_to_log(self, recent_text: str) -> None:
        """Move oldest entry from recent.md to project_log.md (1 sentence)."""
        sections = recent_text.split("\n## Session ")
        if len(sections) <= 2:
            return  # Header + 1 entry, nothing to overflow

        # Extract oldest entry (sections[1])
        oldest = sections[1]
        oldest_line = oldest.split("\n")[0].strip()  # "2024-04-28: Database Schema"

        # Compress to 1 sentence for project_log
        log_entry = f"\n## {oldest_line}\n"
        # Extract first decision if any
        for line in oldest.split("\n"):
            if line.strip().startswith("- Decision:"):
                log_entry += line.strip().replace("- Decision: ", "") + "\n"
                break

        # Append to project_log.md
        log_text = ""
        if self._log_path.exists():
            log_text = self._log_path.read_text(encoding="utf-8", errors="replace")
        if not log_text:
            log_text = "# Project Milestone Log\n"

        log_text += log_entry

        # Enforce project_log budget (max 10 entries)
        log_sections = log_text.split("\n## ")
        if len(log_sections) > 11:  # header + 10 entries
            log_text = log_sections[0] + "\n## " + "\n## ".join(log_sections[-10:])

        self._log_path.write_text(log_text, encoding="utf-8")

        # Remove oldest from recent.md
        new_recent = sections[0] + "\n## Session " + "\n## Session ".join(sections[2:])
        self._recent_path.write_text(new_recent, encoding="utf-8")


# ── Rule-based extraction helpers ────────────────────────────────────────

def _extract_decisions(text: str) -> List[str]:
    """Extract decision statements from agent response. No LLM needed."""
    decisions: List[str] = []
    for pattern in _DECISION_PATTERNS:
        for match in pattern.finditer(text):
            decision = match.group(1).strip().rstrip(".")
            if len(decision) > 15:  # Skip very short matches
                decisions.append(decision)
    # Deduplicate
    seen: Set[str] = set()
    unique: List[str] = []
    for d in decisions:
        key = d.lower()[:30]
        if key not in seen:
            seen.add(key)
            unique.append(d)
    return unique[:5]  # Max 5 decisions per turn


def _extract_test_result(text: str) -> Optional[str]:
    """Extract test pass/fail from agent response."""
    text_lower = text.lower()
    if any(w in text_lower for w in ["test passed", "tests pass", "all tests pass", "tests green"]):
        return "pass"
    if any(w in text_lower for w in ["test failed", "tests fail", "test failure", "tests red"]):
        return "fail"
    # Check for "X passed, Y failed" pattern
    match = re.search(r"(\d+)\s+(?:pass|passed).*?(\d+)\s+(?:fail|failed)", text_lower)
    if match:
        passed, failed = int(match.group(1)), int(match.group(2))
        if failed > 0:
            return f"fail ({passed} passed, {failed} failed)"
        return f"pass ({passed} passed)"
    return None
