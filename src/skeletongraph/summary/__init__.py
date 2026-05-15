"""SkeletonGraph summary sub-package.

Three tiers of function summary generation:

  Tier-0    Local heuristic  (summary/local.py)     — free, instant, always available
  Tier-0.5  Ollama local LLM (summary/ollama.py)    — free, on-device, requires Ollama
  Tier-1    Cloud LLM        (llm/summarizer.py)     — paid, highest quality

Post-turn queue (summary/queue.py) drains summaries asynchronously after
each PostToolUse hook so the current query is never blocked.

Public API:
    from skeletongraph.summary import (
        build_local_summary,           # Tier-0: heuristic
        is_ollama_available,           # Tier-0.5 probe
        generate_summary_ollama,       # Tier-0.5 generation
        batch_generate_ollama,
        enqueue_for_summary,           # queue management
        drain_queue_background,
        queue_size,
        SummaryStore,                  # persistence
    )
"""

from .local import build_local_summary
from .ollama import (
    is_ollama_available,
    list_ollama_models,
    generate_summary_ollama,
    batch_generate_ollama,
)
from .queue import (
    enqueue_for_summary,
    drain_queue_background,
    queue_size,
)
from .summary_store import SummaryStore

__all__ = [
    "build_local_summary",
    "is_ollama_available",
    "list_ollama_models",
    "generate_summary_ollama",
    "batch_generate_ollama",
    "enqueue_for_summary",
    "drain_queue_background",
    "queue_size",
    "SummaryStore",
]
