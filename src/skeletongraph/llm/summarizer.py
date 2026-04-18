"""
Batch function summarizer using LLM.

Generates concise, structured summaries for function bodies.
Only summarizes dirty (new/changed) functions. Summaries are stored
separately from skeletons in SummaryStore.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set

from ..parser.skeleton import SkeletonCore
from ..storage.local import IndexStore, save_index
from ..summary.summary_store import SummaryStore
from .provider import LLMConfig, LLMResponse, complete


SUMMARIZE_SYSTEM = """You are a code documentation expert. Generate a ONE-LINE summary for the given function.

Rules:
- Maximum 15 words
- Start with a verb (Returns, Validates, Computes, Handles, etc.)
- Include the key input→output transformation
- Mention side effects if any (writes to DB, raises exceptions, modifies state)
- Do NOT repeat the function name
- Do NOT include implementation details

Examples:
- "Validates JWT token and returns associated User or None if expired."
- "Computes SHA256 hash of file contents for dirty tracking."
- "Dispatches incoming WebSocket messages to registered handler callbacks."
"""

SUMMARIZE_PROMPT = """Function: {fqn}
Signature: {signature}
{decorators}
Body:
```
{body}
```

One-line summary:"""


@dataclass
class SummarizeResult:
    """Result of a batch summarization run."""
    total_functions: int
    summarized: int
    skipped: int  # Already had summaries
    errors: int
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost: float = 0.0
    duration_seconds: float = 0.0


def summarize_index(
    store: IndexStore,
    project_root: Path,
    config: Optional[LLMConfig] = None,
    force: bool = False,
    on_progress: Optional[Callable[[str, int, int], None]] = None,
    min_body_tokens: int = 20,
) -> SummarizeResult:
    """Generate summaries for all functions that need them.

    Args:
        store: The loaded index.
        project_root: For reading function bodies.
        config: LLM configuration.
        force: If True, re-summarize everything.
        on_progress: Optional callback(fqn, current, total).
        min_body_tokens: Skip functions with body smaller than this.

    Returns:
        SummarizeResult with stats.
    """
    cfg = config or LLMConfig()
    start = time.time()

    # Determine which functions need summarization
    candidates = []
    for fqn, sk in store.skeleton_table.items():
        # Skip tiny functions (getters, simple returns)
        if sk.body_token_estimate < min_body_tokens:
            continue

        # Skip if already summarized (unless force)
        if not force and store.summaries.get(fqn):
            continue

        candidates.append(sk)

    total = len(candidates)
    summarized = 0
    errors = 0
    total_in = 0
    total_out = 0
    total_cost = 0.0

    for i, sk in enumerate(candidates):
        if on_progress:
            on_progress(sk.fqn, i + 1, total)

        body = _read_body(sk, project_root)
        if not body:
            errors += 1
            continue

        # Truncate very long bodies to save tokens
        if len(body) > 2000:
            body = body[:1800] + "\n    # ... (truncated)"

        prompt = SUMMARIZE_PROMPT.format(
            fqn=sk.fqn,
            signature=sk.signature,
            decorators="\n".join(f"  {d}" for d in sk.decorators) if sk.decorators else "",
            body=body,
        )

        try:
            resp = complete(prompt, system=SUMMARIZE_SYSTEM, config=cfg)
            summary = _clean_summary(resp.text)

            if summary:
                store.summaries.set(sk.fqn, summary)
                summarized += 1
                total_in += resp.input_tokens
                total_out += resp.output_tokens
                total_cost += resp.cost
            else:
                errors += 1

        except Exception:
            errors += 1

    # Persist updated summaries
    if summarized > 0:
        sg_dir = project_root / ".skeletongraph"
        store.summaries.save(sg_dir)

    return SummarizeResult(
        total_functions=len(store.skeleton_table),
        summarized=summarized,
        skipped=len(store.skeleton_table) - total,
        errors=errors,
        total_input_tokens=total_in,
        total_output_tokens=total_out,
        total_cost=total_cost,
        duration_seconds=time.time() - start,
    )


def generate_single_summary(
    sk: SkeletonCore,
    body: str,
    config: Optional[LLMConfig] = None,
) -> Optional[str]:
    """Generate a summary for a single function. Used by MCP expand tool."""
    cfg = config or LLMConfig()

    if len(body) > 2000:
        body = body[:1800] + "\n    # ... (truncated)"

    prompt = SUMMARIZE_PROMPT.format(
        fqn=sk.fqn,
        signature=sk.signature,
        decorators="\n".join(f"  {d}" for d in sk.decorators) if sk.decorators else "",
        body=body,
    )

    try:
        resp = complete(prompt, system=SUMMARIZE_SYSTEM, config=cfg)
        return _clean_summary(resp.text)
    except Exception:
        return None


def _read_body(sk: SkeletonCore, project_root: Path) -> str:
    """Read function body from disk."""
    file_path = project_root / sk.file_path
    if not file_path.exists():
        return ""
    try:
        lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
        return "\n".join(lines[sk.line_start - 1:sk.line_end])
    except Exception:
        return ""


def _clean_summary(text: str) -> str:
    """Clean up LLM output to a single-line summary."""
    # Remove markdown formatting, quotes, etc.
    text = text.strip().strip('"').strip("'").strip("`").strip()
    # Take first line only
    text = text.split("\n")[0].strip()
    # Remove leading dash/bullet
    if text.startswith(("- ", "* ", "• ")):
        text = text[2:]
    # Cap length
    if len(text) > 200:
        text = text[:197] + "..."
    return text
