"""
Mode modifiers: reasoning instructions injected into assembled context.

7 modifiers total:
  - 6 instruction-level (text appended near top of context)
  - 1 API-level (EXTENDED_THINKING — flag only, not text)

Modifiers shape HOW the LLM reasons, not WHAT it sees.
Selected automatically by classifier.py. Max 2 instruction-level per query.
"""

from __future__ import annotations

from typing import Dict, List


# ── Modifier Templates ──────────────────────────────────────────────────
# Each is ~60-100 tokens. Placed at position 3 in assembly
# (after task + constraints, before session/architecture/code).

MODIFIER_TEMPLATES: Dict[str, str] = {
    "BRAINSTORM": (
        "## Reasoning Mode: Brainstorm\n"
        "Before recommending an approach:\n"
        "1. Generate at least 3 distinct approaches, including at least one you'd normally dismiss\n"
        "2. For each: state the core tradeoff, not just pros/cons\n"
        "3. Flag which approach fits best given the constraints above and WHY\n"
        "4. Only then give a recommendation\n"
        "\n"
        "Do not recommend the first approach you think of. The right answer for this\n"
        "project may not be the obvious one."
    ),

    "BLAST_FIRST": (
        "## Reasoning Mode: Blast-First\n"
        "Before making any changes:\n"
        "1. List every caller, consumer, and dependent of the code being refactored\n"
        "2. For each: state whether the change will break it, may break it, or is safe\n"
        "3. Identify the highest-risk change (most callers, least test coverage)\n"
        "4. Start with the highest-risk change and confirm approach before proceeding\n"
        "\n"
        "Do not write code until the blast radius analysis is complete."
    ),

    "VERIFY_ASSUMPTIONS": (
        "## Reasoning Mode: Verify Assumptions\n"
        "Before diagnosing:\n"
        "1. List at least 3 possible causes, ordered by likelihood given recent changes\n"
        "2. For each cause: what evidence would confirm or rule it out?\n"
        "3. Check the evidence available in the context above\n"
        "4. Only after ruling out alternatives: state your diagnosis\n"
        "\n"
        "The most obvious cause is often not the actual cause in debugging.\n"
        "Favor hypotheses that explain the symptoms given RECENT CHANGES (see session memory)."
    ),

    "STEP_COMMIT": (
        "## Reasoning Mode: Step-Commit\n"
        "This is a multi-step implementation. Use this process:\n"
        "1. State the implementation plan (files to create/modify, order of changes)\n"
        "2. Wait for confirmation before writing code (just state the plan first)\n"
        "3. Implement one logical unit at a time\n"
        "4. After each unit: state what's complete, what's next, what could break\n"
        "\n"
        "Do not attempt to implement everything in one response.\n"
        "A correct partial implementation is better than a broken complete one."
    ),

    "MINIMAL": (
        "## Reasoning Mode: Minimal Change\n"
        "Make the smallest correct change that achieves the goal.\n"
        "Do not refactor surrounding code unless it's required for correctness.\n"
        "Do not improve unrelated things you notice.\n"
        "If you see something that should be fixed but is out of scope, note it briefly — don't fix it."
    ),

    "THINK_ALOUD": (
        "## Reasoning Mode: Think Aloud\n"
        "For this task, show your reasoning before your answer:\n"
        "- What are you uncertain about?\n"
        "- What assumptions are you making?\n"
        "- What would change your answer if it turned out to be wrong?\n"
        "\n"
        "Keep reasoning concise. The goal is to surface hidden assumptions,\n"
        "not to write an essay. Then give your answer."
    ),
}

# EXTENDED_THINKING is NOT a template — it's an API-level flag.
# The assembler sets metadata.extended_thinking = True.
# The caller (MCP server, hook, CLI) uses the flag if the model supports it.
# Trigger: PLANNING or DEBUG_INVESTIGATE + cross_file > 3 or dep_depth > 3.
# Only works for: Claude Code API-direct, sg-agent, sg prompt → Claude.ai.
# Does NOT work for: Cursor, Copilot, Antigravity (they control the API call).


def render_modifiers(modifier_names: List[str]) -> str:
    """Render selected modifiers into a single text block for context injection.

    Args:
        modifier_names: List of modifier names (e.g., ["BRAINSTORM", "THINK_ALOUD"])

    Returns:
        Combined modifier text, or empty string if no modifiers.
        EXTENDED_THINKING is silently skipped (it's API-level, not text).
    """
    parts = []
    for name in modifier_names:
        if name == "EXTENDED_THINKING":
            continue  # API-level, not rendered as text
        template = MODIFIER_TEMPLATES.get(name)
        if template:
            parts.append(template)

    if not parts:
        return ""

    return "\n\n".join(parts)


def estimate_modifier_tokens(modifier_names: List[str]) -> int:
    """Estimate token count for selected modifiers.

    Rough estimate: ~4 chars per token for English instruction text.
    """
    text = render_modifiers(modifier_names)
    if not text:
        return 0
    # ~4 chars per token is a conservative estimate for English instructions
    return len(text) // 4
