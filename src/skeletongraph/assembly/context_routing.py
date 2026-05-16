"""Smart context routing — pick which MD files / sections to include per prompt.

The full context is expensive: constraints + architecture + project + session digest
+ top functions adds 1k–3k tokens to every prompt. Most prompts don't need all of it.

This module classifies the prompt by keyword family and returns ONLY the sections
that actually help for that kind of task.

Always-on:
  • constraints (compact)        — Zone 1, never skip
  • session digest (5 turns)     — short, useful for continuity

Conditional:
  • architecture.md              — for design/refactor/migrate/explain queries
  • project.md                   — for "what is this codebase" queries
  • decisions.md                 — for "why did we..." queries

This is heuristic — no LLM call — so it's cheap to run on every hook invocation.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Tuple

# Keyword families. Order matters — first match wins for the primary mode.
_KEYWORD_FAMILIES: List[Tuple[str, List[str]]] = [
    (
        "architecture",
        [
            "architecture", "architect", "design", "structure", "organize",
            "refactor", "restructure", "migrate", "module", "layer", "pattern",
            "dependency", "coupling", "abstraction",
        ],
    ),
    (
        "explain",
        [
            "what is", "what does", "explain", "describe", "overview",
            "how does this", "walk me through", "tell me about",
        ],
    ),
    (
        "decision",
        [
            "why did", "why do we", "why are we", "rationale", "history of",
            "decision", "tradeoff", "trade-off",
        ],
    ),
    (
        "debug",
        [
            "fix", "bug", "broken", "error", "fail", "crash", "exception",
            "traceback", "regression", "wrong", "not working",
        ],
    ),
    (
        "test",
        [
            "test", "coverage", "spec", "pytest", "unittest", "mock", "fixture",
        ],
    ),
    (
        "review",
        [
            "review", "audit", "check", "validate", "inspect", "lint",
            "security", "vulnerability",
        ],
    ),
]

# Section name → (filename, max_chars)
_SECTION_FILES: Dict[str, Tuple[str, int]] = {
    "architecture": ("architecture.md", 3200),   # ~800 tokens
    "project":      ("project.md", 1600),        # ~400 tokens
    "decisions":    ("decisions.md", 2400),      # ~600 tokens
}


def classify_prompt(prompt: str) -> str:
    """Return the primary keyword family for a prompt.

    Returns 'general' if no family matches.
    """
    p = (prompt or "").lower()
    if not p.strip():
        return "general"
    for family, keywords in _KEYWORD_FAMILIES:
        for kw in keywords:
            # Word-boundary for single words; phrase match for multi-word
            if " " in kw:
                if kw in p:
                    return family
            else:
                if re.search(rf"\b{re.escape(kw)}\b", p):
                    return family
    return "general"


def route_context_sections(
    prompt: str,
    sg_dir: Path,
) -> Dict[str, str]:
    """Pick which optional MD sections to include, based on prompt classification.

    Returns:
        Dict mapping section_name → trimmed content. May be empty for general queries.
        Caller is responsible for actually injecting these sections into the
        prompt/overview output.

    Always-on sections (constraints, session digest, top functions) are NOT
    handled here — they're always included by the hook/tool caller.
    """
    sections: Dict[str, str] = {}
    family = classify_prompt(prompt)

    if family == "architecture":
        _add_section(sections, "architecture", sg_dir)

    elif family == "explain":
        # "what is this codebase" — include project + architecture (short)
        _add_section(sections, "project", sg_dir)
        _add_section(sections, "architecture", sg_dir, char_override=1600)

    elif family == "decision":
        _add_section(sections, "decisions", sg_dir)

    # debug / test / review / general: no extra MD — constraints + session +
    # top functions are usually enough. The caller adds those unconditionally.

    return sections


def _add_section(
    sections: Dict[str, str],
    key: str,
    sg_dir: Path,
    char_override: int = 0,
) -> None:
    """Read the MD file for `key`, trim, store under sections[key]."""
    if key not in _SECTION_FILES:
        return
    filename, default_chars = _SECTION_FILES[key]
    cap = char_override or default_chars
    path = sg_dir / filename
    if not path.exists():
        return
    try:
        text = path.read_text(encoding="utf-8", errors="replace").strip()
        if not text:
            return
        if len(text) > cap:
            text = text[:cap].rstrip() + "\n... (truncated)"
        sections[key] = text
    except Exception:
        pass


def format_routed_sections(sections: Dict[str, str]) -> str:
    """Render routed sections as markdown blocks. Returns '' if empty."""
    if not sections:
        return ""
    parts = []
    titles = {
        "architecture": "## Architecture",
        "project":      "## Project",
        "decisions":    "## Decisions",
    }
    for key, text in sections.items():
        title = titles.get(key, f"## {key.title()}")
        parts.append(f"{title}\n{text}")
    return "\n\n".join(parts)
