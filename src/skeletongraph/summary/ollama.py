"""Tier-0.5: Ollama-based local LLM summary generation.

Zero API cost, runs on-device. Used as middle tier between Tier-0
(heuristic) and Tier-1 (cloud LLM). Only activated when Ollama is
detected running on localhost.

Default model: qwen2.5-coder:1.5b  (fast, <2s per function on CPU)
Configurable via config.ollama_summary_model.

Communication goes directly to Ollama's /api/generate and /api/tags
via urllib (no extra dependencies).
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Dict, List, Optional, Tuple

_OLLAMA_DEFAULT_BASE = "http://localhost:11434"
_PROBE_TIMEOUT = 2   # seconds — fast enough to not block hook path

_SUMMARIZE_SYSTEM = (
    "Generate a ONE-LINE function summary. "
    "Rules: max 15 words, start with a verb (Returns/Validates/Computes/Handles), "
    "include key input→output, note side effects, do NOT repeat the function name."
)

_SUMMARIZE_PROMPT = (
    "Function: {fqn}\n"
    "Signature: {signature}\n"
    "Body:\n```\n{body}\n```\n\n"
    "One-line summary:"
)


# ── Probe ────────────────────────────────────────────────────────────────


def is_ollama_available(base_url: str = _OLLAMA_DEFAULT_BASE) -> bool:
    """Return True if Ollama is reachable at base_url. Never raises."""
    try:
        req = urllib.request.Request(f"{base_url}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=_PROBE_TIMEOUT) as resp:
            return resp.status == 200
    except Exception:
        return False


def list_ollama_models(base_url: str = _OLLAMA_DEFAULT_BASE) -> List[str]:
    """Return names of models available in Ollama. Returns [] on error."""
    try:
        req = urllib.request.Request(f"{base_url}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=_PROBE_TIMEOUT) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return [m["name"] for m in data.get("models", [])]
    except Exception:
        return []


# ── Single-function summary ──────────────────────────────────────────────


def generate_summary_ollama(
    fqn: str,
    signature: str,
    body: str,
    model: str = "qwen2.5-coder:1.5b",
    base_url: str = _OLLAMA_DEFAULT_BASE,
    timeout: int = 15,
) -> Optional[str]:
    """Generate a one-line summary via Ollama /api/generate.

    Args:
        fqn: Fully-qualified function name.
        signature: Function signature string.
        body: Function body text (will be truncated if >2000 chars).
        model: Ollama model name (without 'ollama/' prefix).
        base_url: Ollama server base URL.
        timeout: Request timeout in seconds.

    Returns:
        Cleaned one-line summary string, or None on failure.
    """
    # Strip 'ollama/' prefix in case caller passes litellm-style name
    model_name = model.removeprefix("ollama/")

    if len(body) > 2000:
        body = body[:1800] + "\n    # ... (truncated)"

    full_prompt = (
        f"{_SUMMARIZE_SYSTEM}\n\n"
        + _SUMMARIZE_PROMPT.format(fqn=fqn, signature=signature, body=body)
    )

    payload = json.dumps({
        "model": model_name,
        "prompt": full_prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_predict": 80,
            "stop": ["\n", "Function:", "Signature:"],
        },
    }).encode("utf-8")

    try:
        req = urllib.request.Request(
            f"{base_url}/api/generate",
            data=payload,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            text = data.get("response", "").strip()
            cleaned = _clean_summary(text)
            return cleaned or None
    except Exception:
        return None


# ── Batch ────────────────────────────────────────────────────────────────


def batch_generate_ollama(
    items: List[Tuple[str, str, str]],   # (fqn, signature, body)
    model: str = "qwen2.5-coder:1.5b",
    base_url: str = _OLLAMA_DEFAULT_BASE,
    timeout_per: int = 15,
) -> Dict[str, str]:
    """Generate summaries for a batch of functions via Ollama.

    Returns:
        Dict mapping fqn → summary (only successful entries included).
    """
    results: Dict[str, str] = {}
    for fqn, signature, body in items:
        summary = generate_summary_ollama(
            fqn=fqn,
            signature=signature,
            body=body,
            model=model,
            base_url=base_url,
            timeout=timeout_per,
        )
        if summary:
            results[fqn] = summary
    return results


# ── Internal ─────────────────────────────────────────────────────────────


def _clean_summary(text: str) -> str:
    """Strip markdown artefacts and enforce single-line output."""
    text = text.strip().strip('"').strip("'").strip("`").strip()
    # Take only the first line
    text = text.split("\n")[0].strip()
    # Remove leading bullet/dash
    if text.startswith(("- ", "* ", "• ")):
        text = text[2:].strip()
    # Hard cap
    if len(text) > 200:
        text = text[:197] + "..."
    return text
