"""
Inverted index: keyword → set of FQNs.

When a user prompt doesn't mention specific function names, we tokenize
the prompt and look up matching functions via keywords extracted from
function names, signatures, and summaries.

Zero LLM cost — pure string matching.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple


# Regex to split identifiers into tokens: camelCase, snake_case, PascalCase
_SPLIT_PATTERN = re.compile(r"[_\-.]|(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")

# Common stop words to exclude from index (too generic, match everything)
_STOP_WORDS = frozenset({
    "self", "cls", "args", "kwargs", "none", "true", "false",
    "return", "def", "class", "import", "from", "if", "else",
    "for", "while", "try", "except", "with", "as", "in", "is",
    "not", "and", "or", "the", "a", "an", "to", "of", "it",
    "this", "that", "str", "int", "float", "bool", "list", "dict",
    "set", "tuple", "type", "any", "optional", "void", "null",
    "undefined", "var", "let", "const", "function", "async",
    "await", "new", "get", "set",
})


def tokenize_identifier(name: str) -> List[str]:
    """Split an identifier into searchable tokens.

    Examples:
        'validate_token'  → ['validate', 'token']
        'AuthMiddleware'  → ['auth', 'middleware']
        'getUserById'     → ['get', 'user', 'by', 'id']
        'decode_jwt'      → ['decode', 'jwt']
    """
    parts = _SPLIT_PATTERN.split(name)
    tokens = []
    for part in parts:
        part = part.strip().lower()
        if part and len(part) > 1 and part not in _STOP_WORDS:
            tokens.append(part)
    return tokens


def tokenize_text(text: str) -> List[str]:
    """Tokenize free text (summaries, prompts) into searchable terms."""
    # Split on non-alphanumeric characters AND underscores for consistency
    raw = re.split(r"[^a-zA-Z0-9]+", text.lower().replace("_", " "))
    return [t for t in raw if t and len(t) > 1 and t not in _STOP_WORDS]


class InvertedIndex:
    """Maps keywords to sets of FQNs for fast lookup.

    Built from function names, signatures, and summaries.
    Searched against tokenized user prompts.
    """

    def __init__(self) -> None:
        self._index: Dict[str, Set[str]] = defaultdict(set)
        self._fqn_tokens: Dict[str, Set[str]] = {}  # For removal during updates

    def add(self, fqn: str, name: str, signature: str = "",
            summary: str = "", docstring: str = "") -> None:
        """Index a function by its name, signature, summary, and docstring.

        Args:
            fqn: Fully qualified name (the value stored in the index).
            name: Function/class name (e.g., 'validate_token').
            signature: Full signature for additional keyword extraction.
            summary: Function summary text.
            docstring: First line of docstring (high-signal, low-noise).
        """
        tokens: Set[str] = set()

        # Tokenize the function name (highest signal)
        tokens.update(tokenize_identifier(name))

        # Tokenize signature parameters
        if signature:
            # Extract parameter names from signature
            paren_match = re.search(r"\((.+?)\)", signature)
            if paren_match:
                params_str = paren_match.group(1)
                for param in params_str.split(","):
                    param_name = param.strip().split(":")[0].split("=")[0].strip()
                    tokens.update(tokenize_identifier(param_name))

        # Tokenize summary
        if summary:
            tokens.update(tokenize_text(summary))

        # Tokenize docstring (high signal: describes what the function does)
        if docstring:
            tokens.update(tokenize_text(docstring))

        # Also index the full function name as-is (for exact matches)
        full_name = fqn.split("::")[-1] if "::" in fqn else fqn
        tokens.add(full_name.lower())

        # Store tokens for this FQN (for removal)
        self._fqn_tokens[fqn] = tokens

        # Add to inverted index
        for token in tokens:
            self._index[token].add(fqn)

    def remove(self, fqn: str) -> None:
        """Remove a FQN from the index. Used during incremental updates."""
        tokens = self._fqn_tokens.pop(fqn, set())
        for token in tokens:
            if token in self._index:
                self._index[token].discard(fqn)
                if not self._index[token]:
                    del self._index[token]

    def search(
        self,
        query: str,
        top_k: int = 10,
        min_score: float = 0.1,
    ) -> List[Tuple[str, float]]:
        """Search the index with a free-text query.

        Returns FQNs ranked by relevance (number of matching tokens / total query tokens).

        Args:
            query: Free text query (user prompt or keywords).
            top_k: Maximum results to return.
            min_score: Minimum relevance score (0-1) to include.

        Returns:
            List of (fqn, score) tuples, sorted by score descending.
        """
        query_tokens = set(tokenize_text(query))
        if not query_tokens:
            return []

        # Count how many query tokens each FQN matches
        fqn_scores: Dict[str, float] = defaultdict(float)
        for token in query_tokens:
            for fqn in self._index.get(token, set()):
                fqn_scores[fqn] += 1.0

        # Normalize by query token count
        total_query_tokens = len(query_tokens)
        results = [
            (fqn, score / total_query_tokens)
            for fqn, score in fqn_scores.items()
            if score / total_query_tokens >= min_score
        ]

        # Sort by score descending, then alphabetically for stability
        results.sort(key=lambda x: (-x[1], x[0]))
        return results[:top_k]

    def lookup(self, keyword: str) -> Set[str]:
        """Direct keyword lookup. Returns all FQNs indexed under this keyword."""
        return set(self._index.get(keyword.lower(), set()))

    @property
    def term_count(self) -> int:
        """Number of unique terms in the index."""
        return len(self._index)

    @property
    def entry_count(self) -> int:
        """Number of FQNs indexed."""
        return len(self._fqn_tokens)

    def to_dict(self) -> dict:
        """Serialize for JSON storage."""
        return {
            "index": {k: sorted(v) for k, v in self._index.items()},
            "fqn_tokens": {k: sorted(v) for k, v in self._fqn_tokens.items()},
        }

    @classmethod
    def from_dict(cls, data: dict) -> InvertedIndex:
        """Deserialize from JSON."""
        idx = cls()
        for term, fqns in data.get("index", {}).items():
            idx._index[term] = set(fqns)
        for fqn, tokens in data.get("fqn_tokens", {}).items():
            idx._fqn_tokens[fqn] = set(tokens)
        return idx
