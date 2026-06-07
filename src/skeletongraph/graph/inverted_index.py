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
    # Python
    "self", "cls", "args", "kwargs", "none", "true", "false",
    "return", "def", "class", "import", "from", "if", "else",
    "for", "while", "try", "except", "with", "as", "in", "is",
    "not", "and", "or", "the", "a", "an", "to", "of", "it",
    "this", "that", "str", "int", "float", "bool", "list", "dict",
    "set", "tuple", "type", "any", "optional", "void", "null",
    "undefined", "var", "let", "const", "function", "async",
    "await", "new", "get", "set",
    # Go keywords (never useful as search tokens)
    "func", "struct", "interface", "defer", "chan", "select",
    "fallthrough", "nil", "goto", "range", "switch", "case",
    "break", "continue", "default", "package", "map",
    # TS/JS keywords
    "export", "extends", "implements", "declare", "readonly",
    "keyof", "namespace", "abstract", "enum", "module",
    "typeof", "instanceof", "yield", "super",
    # Java/C# keywords
    "public", "private", "protected", "static", "final",
    "throws", "throw", "catch", "void",
    # Rust
    "pub", "mut", "impl", "trait", "where", "unsafe", "match",
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
    """Tokenize free text (summaries, prompts) into searchable terms.
    
    Uses BUILD-TIME stop words (aggressive filtering) — suitable for
    indexing summaries and docstrings where we want clean tokens.
    """
    # Split on non-alphanumeric characters AND underscores for consistency
    raw = re.split(r"[^a-zA-Z0-9]+", text.lower().replace("_", " "))
    return [t for t in raw if t and len(t) > 1 and t not in _STOP_WORDS]


# ── Query-Time Stop Words (lighter) ─────────────────────────────────────
# Only removes true noise: articles, pronouns, PL keywords.
# KEEPS domain terms like "content", "length", "header", "session" that
# match indexed function data. This fixes the v3 MISS on NLP prompts.

_QUERY_STOP_WORDS = frozenset({
    # Python
    "self", "cls", "args", "kwargs", "none", "true", "false",
    "return", "def", "class", "import", "from", "if", "else",
    "for", "while", "try", "except", "with", "as", "in", "is",
    "not", "and", "or", "the", "a", "an", "to", "of", "it",
    "this", "that", "void", "null", "undefined",
    "var", "let", "const", "function", "async", "await", "new",
    # Go keywords (never useful as query tokens)
    "func", "struct", "interface", "defer", "chan", "select",
    "fallthrough", "nil", "goto", "range", "switch", "case",
    "break", "continue", "default", "package", "map",
    # TS/JS keywords
    "export", "extends", "implements", "declare", "readonly",
    "keyof", "namespace", "abstract", "enum", "module",
    "typeof", "instanceof", "yield", "super",
    # Java/C#
    "public", "private", "protected", "static", "final",
    "throws", "throw", "catch",
    # Rust
    "pub", "mut", "impl", "trait", "where", "unsafe", "match",
    # Keep intentionally: str, int, float, bool, list, dict, set, tuple,
    #   type, any, optional, get, set — these can match function names
})


def tokenize_query(query: str) -> List[str]:
    """Tokenize a user query with LIGHTER stop-word filtering.
    
    Unlike tokenize_text() which is for build-time indexing (aggressive filtering),
    this keeps domain terms that could match indexed functions.
    
    Example:
        "Content-Length is always being sent on GET requests"
        tokenize_text → []  (v3: all words filtered!)
        tokenize_query → ['content', 'length', 'always', 'being', 'sent', 'get', 'requests']
    """
    raw = re.split(r"[^a-zA-Z0-9]+", query.lower().replace("_", " "))
    return [t for t in raw if t and len(t) > 1 and t not in _QUERY_STOP_WORDS]


def extract_body_keywords(body: str, language: str = "python") -> List[str]:
    """Extract high-signal keywords from a function body.

    Captures string literals (like 'Content-Length'), dict key access patterns,
    method calls, and error patterns — things that are never in the function
    name or signature but are exactly what users search for.

    Language-aware: extracts patterns specific to Python, Go, TypeScript/JS.
    Zero LLM cost — pure regex extraction.
    """
    keywords: List[str] = []

    # 1. String literals in quotes — universal across all languages
    for match in re.finditer(r'["\']([A-Za-z][A-Za-z0-9_-]{2,}(?:[- ][A-Za-z0-9_-]+)*)["\']', body):
        val = match.group(1)
        if val.lower() not in _STOP_WORDS and '/' not in val and '\\' not in val:
            parts = re.split(r'[-_ ]', val.lower())
            keywords.extend(p for p in parts if len(p) > 1 and p not in _STOP_WORDS)

    # 2. Dict/header/bracket access — universal
    for match in re.finditer(r'\[\s*["\']([A-Za-z][A-Za-z0-9_-]+)["\']\s*\]', body):
        val = match.group(1)
        parts = re.split(r'[-_ ]', val.lower())
        keywords.extend(p for p in parts if len(p) > 1 and p not in _STOP_WORDS)

    # === Python-specific ===
    if language == "python":
        for match in re.finditer(r'raise\s+([A-Z][a-zA-Z]+)', body):
            keywords.extend(tokenize_identifier(match.group(1)))
        for match in re.finditer(r'self\.([a-z_][a-z_0-9]*)', body):
            attr = match.group(1)
            if len(attr) > 2 and attr not in _STOP_WORDS:
                keywords.extend(tokenize_identifier(attr))
        for match in re.finditer(r'self\.([a-z_][a-z_0-9]*)\s*\(', body):
            method = match.group(1)
            if len(method) > 2 and method not in _STOP_WORDS:
                keywords.extend(tokenize_identifier(method))

    # === Go-specific ===
    elif language == "go":
        # Error creation: fmt.Errorf(), errors.New(), errors.Wrap()
        for match in re.finditer(r'(?:fmt\.Errorf|errors\.New|errors\.Wrap)\s*\(\s*"([^"]+)"', body):
            parts = re.split(r'[^a-zA-Z0-9]+', match.group(1).lower())
            keywords.extend(p for p in parts if len(p) > 2 and p not in _STOP_WORDS)
        # Package-qualified calls AND receiver method calls: pkg.Func(), r.Method()
        # Both have the same syntax in Go: lowercase.Uppercase()
        seen_go = set()
        for match in re.finditer(r'\b([a-z][a-zA-Z0-9]*)\.\s*([A-Z][a-zA-Z0-9]+)', body):
            qualifier, name = match.group(1), match.group(2)
            key = (qualifier, name)
            if key in seen_go:
                continue
            seen_go.add(key)
            if len(qualifier) > 1 and qualifier not in _STOP_WORDS:
                keywords.append(qualifier)
            keywords.extend(tokenize_identifier(name))
        # Panic messages
        for match in re.finditer(r'panic\s*\(\s*"([^"]+)"', body):
            parts = re.split(r'[^a-zA-Z0-9]+', match.group(1).lower())
            keywords.extend(p for p in parts if len(p) > 2 and p not in _STOP_WORDS)

    # === TypeScript / JavaScript ===
    elif language in ("typescript", "javascript", "tsx", "jsx"):
        # Method chains: this.service.doSomething()
        for match in re.finditer(r'this\.([a-zA-Z_][a-zA-Z0-9_]*)(?:\.([a-zA-Z_][a-zA-Z0-9_]*))?', body):
            prop = match.group(1)
            method = match.group(2)
            if prop and len(prop) > 1 and prop not in _STOP_WORDS:
                keywords.extend(tokenize_identifier(prop))
            if method and len(method) > 1 and method not in _STOP_WORDS:
                keywords.extend(tokenize_identifier(method))
        # throw new Error("...")
        for match in re.finditer(r'throw\s+new\s+([A-Z][a-zA-Z]+)', body):
            keywords.extend(tokenize_identifier(match.group(1)))
        # Template literals with identifiers: ${variable}
        for match in re.finditer(r'\$\{([a-zA-Z_][a-zA-Z0-9_.]+)\}', body):
            for part in match.group(1).split('.'):
                if len(part) > 1 and part.lower() not in _STOP_WORDS:
                    keywords.extend(tokenize_identifier(part))
        # React/DOM patterns: createElement, querySelector
        for match in re.finditer(r'\b(querySelector|getElementById|createElement|addEventListener|removeEventListener|dispatch|emit|subscribe)\b', body):
            keywords.extend(tokenize_identifier(match.group(1)))
        # Import-like: require('module'), import('module')
        for match in re.finditer(r'(?:require|import)\s*\(\s*["\']([^"\'/][^"\']*)["\'\)]', body):
            mod = match.group(1).split('/')[-1]  # last segment
            if len(mod) > 1 and mod.lower() not in _STOP_WORDS:
                keywords.append(mod.lower())

    # === Universal: identifier calls not caught above ===
    # Captures any functionCall() pattern for all languages
    for match in re.finditer(r'\b([a-zA-Z_][a-zA-Z0-9_]{2,})\s*\(', body):
        name = match.group(1)
        name_lower = name.lower()
        if name_lower not in _STOP_WORDS and name_lower not in {'if', 'for', 'while', 'switch', 'catch', 'func', 'function', 'return', 'print', 'println', 'printf', 'sprintf', 'log', 'console'}:
            keywords.extend(tokenize_identifier(name))

    # Deduplicate while preserving order (first occurrence wins)
    seen = set()
    deduped = []
    for kw in keywords:
        if kw not in seen:
            seen.add(kw)
            deduped.append(kw)
    return deduped


class InvertedIndex:
    """Maps keywords to sets of FQNs for fast lookup.

    Built from function names, signatures, and summaries.
    Searched against tokenized user prompts.
    """

    def __init__(self) -> None:
        self._index: Dict[str, Set[str]] = defaultdict(set)
        self._fqn_tokens: Dict[str, Set[str]] = {}  # For removal during updates

    def add(self, fqn: str, name: str, signature: str = "",
            summary: str = "", docstring: str = "",
            body_keywords: Optional[List[str]] = None) -> None:
        """Index a function by its name, signature, summary, docstring, and body keywords.

        Args:
            fqn: Fully qualified name (the value stored in the index).
            name: Function/class name (e.g., 'validate_token').
            signature: Full signature for additional keyword extraction.
            summary: Function summary text.
            docstring: First line of docstring (high-signal, low-noise).
            body_keywords: Keywords extracted from function body (string literals, etc.).
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

        # Body keywords (string literals, dict keys, exceptions from the code)
        if body_keywords:
            tokens.update(kw.lower() for kw in body_keywords if len(kw) > 1)

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
        min_score: float = 0.05,
    ) -> List[Tuple[str, float]]:
        """Search the index with a free-text query using IDF-weighted scoring.

        Rare tokens (like 'content_length') score higher than common tokens
        (like 'requests') to avoid false positives from high-frequency terms.

        Args:
            query: Free text query (user prompt or keywords).
            top_k: Maximum results to return.
            min_score: Minimum relevance score (0-1) to include.

        Returns:
            List of (fqn, score) tuples, sorted by score descending.
        """
        query_tokens = set(tokenize_query(query))
        if not query_tokens:
            return []

        import math
        total_docs = max(len(self._fqn_tokens), 1)

        # Compute IDF weight for each query token:
        # tokens that appear in fewer documents are more discriminative
        token_idf: Dict[str, float] = {}
        for token in query_tokens:
            doc_freq = len(self._index.get(token, set()))
            if doc_freq > 0:
                # Standard IDF: log(N / df) + 1 (smoothed)
                token_idf[token] = math.log(total_docs / doc_freq) + 1.0
            # If doc_freq == 0, token doesn't exist in index, skip it

        if not token_idf:
            return []

        # Score each FQN by sum of IDF weights + breadth bonus
        fqn_idf_sum: Dict[str, float] = defaultdict(float)
        fqn_match_count: Dict[str, int] = defaultdict(int)  # distinct tokens matched
        for token, idf in token_idf.items():
            for fqn in self._index.get(token, set()):
                fqn_idf_sum[fqn] += idf
                fqn_match_count[fqn] += 1

        # Breadth bonus: matching 3 different query tokens is much better than
        # matching 1 token that happens to have high IDF.
        # Score = IDF_sum * sqrt(num_distinct_matches)
        total_idf = sum(token_idf.values())
        results = []
        for fqn, idf_sum in fqn_idf_sum.items():
            breadth = math.sqrt(fqn_match_count[fqn])
            score = (idf_sum * breadth) / (total_idf * math.sqrt(len(token_idf)))
            if score >= min_score:
                results.append((fqn, score))

        # Sort by score descending, then alphabetically for stability
        results.sort(key=lambda x: (-x[1], x[0]))
        return results[:top_k]

    def lookup(self, keyword: str) -> Set[str]:
        """Direct keyword lookup. Returns all FQNs indexed under this keyword."""
        return set(self._index.get(keyword.lower(), set()))

    def build_bm25_corpus(self) -> Dict[str, str]:
        """Build a lightweight corpus for BM25 from indexed tokens.

        Uses the token set per FQN, which already includes docstrings,
        signatures, and body keywords. Each token appears once to keep
        the corpus compact and deterministic.
        """
        return {
            fqn: " ".join(sorted(tokens))
            for fqn, tokens in self._fqn_tokens.items()
        }

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
