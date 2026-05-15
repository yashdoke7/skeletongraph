"""Local, zero-API summary generation.

These summaries are intentionally conservative. They are not meant to replace
human docstrings or LLM-written notes; they give fresh projects a useful
retrieval baseline from names, signatures, and body keywords.
"""

from __future__ import annotations

import re
from typing import Iterable, List

from ..graph.inverted_index import tokenize_identifier
from ..parser.skeleton import SkeletonCore


def build_local_summary(
    sk: SkeletonCore,
    body_keywords: Iterable[str] = (),
    max_keywords: int = 8,
) -> str:
    """Create a compact deterministic summary for one function/class."""
    if sk.docstring:
        return sk.docstring.strip().splitlines()[0]

    name = sk.fqn.split("::")[-1].split(".")[-1]
    action = _humanize_name(name)
    details: List[str] = []

    params = _extract_params(sk.signature)
    if params:
        details.append(f"parameters: {', '.join(params[:5])}")

    return_type = _extract_return_type(sk.signature)
    if return_type:
        details.append(f"returns {return_type}")

    keywords = _dedupe_keywords(body_keywords, max_keywords=max_keywords)
    if keywords:
        details.append(f"uses {', '.join(keywords)}")

    if details:
        return f"{action}; {'; '.join(details)}."
    return f"{action}."


def _humanize_name(name: str) -> str:
    tokens = tokenize_identifier(name)
    if not tokens:
        return name
    return " ".join(tokens)


def _extract_params(signature: str) -> List[str]:
    match = re.search(r"\((.*?)\)", signature)
    if not match:
        return []

    params = []
    for raw in match.group(1).split(","):
        param = raw.strip()
        if not param or param in {"self", "cls"}:
            continue
        name = param.split(":", 1)[0].split("=", 1)[0].strip()
        if name and name not in {"*", "/"}:
            params.append(name.lstrip("*"))
    return params


def _extract_return_type(signature: str) -> str:
    match = re.search(r"->\s*([^:={]+)", signature)
    if match:
        return match.group(1).strip()

    match = re.search(r":\s*([A-Za-z_][A-Za-z0-9_<>, .|\\[\\]]*)\s*[{;]?$", signature)
    if match and not signature.strip().startswith(("class ", "def ")):
        return match.group(1).strip()
    return ""


def _dedupe_keywords(keywords: Iterable[str], max_keywords: int) -> List[str]:
    seen = set()
    result = []
    for kw in keywords:
        clean = str(kw).strip().lower()
        if not clean or clean in seen:
            continue
        seen.add(clean)
        result.append(clean)
        if len(result) >= max_keywords:
            break
    return result
