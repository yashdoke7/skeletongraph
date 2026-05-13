"""
Core skeleton data structures.

Two-layer design:
  - SkeletonCore: identity + signature + metadata (always in memory, used for graph traversal)
  - SummaryStore: FQN → summary (loaded on demand, only for Tier 2 assembly)

FileSkeleton and ClassSkeleton are container types that group SkeletonCore entries
with file-level / class-level metadata.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

from .node_kinds import NodeKind


def _hash_text(text: str) -> str:
    """SHA256 hash of text content for dirty tracking."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class SkeletonCore:
    """Minimal representation of a callable or type entity.

    This is the 'page table entry' — contains everything needed for graph
    traversal and basic context assembly, but NOT the summary (which lives
    in a separate SummaryStore for memory efficiency).

    ~20-25 tokens when serialized for LLM context (signature + decorators).
    """

    # ── Identity ───────────────────────────────────────────────────────────
    fqn: str
    """Fully qualified name. Format: 'path/to/file.py::Class.method'
    Used as hash key for O(1) lookup. Graph edges reference this."""

    file_path: str
    """Relative path from project root. Agent needs this to read/edit."""

    line_start: int
    """1-indexed start line. NOT sent to LLM — used by agent tool for
    page-fault extraction (like a disk sector address)."""

    line_end: int
    """1-indexed end line (inclusive)."""

    # ── Interface ──────────────────────────────────────────────────────────
    signature: str
    """Full declaration line. For Python: 'def validate_token(token: str) -> bool:'
    For TS: 'function validateToken(token: string): boolean'
    This IS the public contract — often sufficient without a summary."""

    # ── Classification ─────────────────────────────────────────────────────
    kind: NodeKind
    """Semantic kind. Determines retrieval behavior (e.g., CONSTRUCTOR
    gets auto-included when parent class is in context)."""

    decorators: Tuple[str, ...] = ()
    """Decorator/annotation names. ~3 tokens each, high information density.
    e.g., ('@staticmethod', '@login_required', '@pytest.fixture')"""

    is_exported: bool = False
    """True if part of the module's public API (__all__, export default, pub).
    Exported entities get higher priority in blast-radius analysis."""

    # ── Metrics ────────────────────────────────────────────────────────────
    complexity: int = 1
    """Cyclomatic complexity. High-complexity functions are more likely to
    contain bugs and need full-body expansion. Used by ranker."""

    body_token_estimate: int = 0
    """Approximate token count of the full function body. Used by the
    budget manager to decide if page-fault expansion fits the budget."""

    # ── Documentation ──────────────────────────────────────────────────────
    docstring: str = ""
    """First line of the function/class docstring. Used for search indexing
    (inverted index). NOT sent to LLM context — purely for retrieval."""

    # ── Dirty tracking ─────────────────────────────────────────────────────
    sha256: str = ""
    """Hash of the function body text. Only re-summarize if changed.
    Never sent to LLM — purely internal."""

    @property
    def file_display(self) -> str:
        """Short display: 'middleware.py:45'"""
        name = Path(self.file_path).name
        return f"{name}:{self.line_start}"

    @property
    def return_type(self) -> Optional[str]:
        """Extract return type from signature if present."""
        if "->" in self.signature:
            return self.signature.split("->")[-1].strip().rstrip(":")
        if ":" in self.signature and self.kind in (
            NodeKind.FUNCTION, NodeKind.METHOD, NodeKind.ASYNC_FUNCTION,
        ):
            # TypeScript: function name(): ReturnType
            parts = self.signature.rsplit(":", 1)
            if len(parts) == 2 and "(" in parts[0]:
                ret = parts[1].strip()
                if ret:  # Guard against empty string from trailing colon
                    return ret
        return None

    def to_tier2_str(self, summary: str = "") -> str:
        """Tier 2 serialization: signature + summary + decorators.
        Used in Zone 3 for 1-hop neighbors. ~25 tokens."""
        parts = []
        if self.decorators:
            parts.append(" ".join(self.decorators))
        parts.append(self.signature)
        if summary:
            parts.append(f'  # {summary}')
        return "\n".join(parts)

    def to_tier3_str(self) -> str:
        """Tier 3 serialization: FQN + return type only.
        Used in Zone 3 for 2-hop periphery. ~8 tokens."""
        ret = self.return_type or "?"
        # Take just the function name from FQN for brevity
        short_name = self.fqn.split("::")[-1] if "::" in self.fqn else self.fqn
        return f"{short_name} -> {ret}"

    def to_dict(self) -> dict:
        """Serialize for JSON storage."""
        d = {
            "fqn": self.fqn,
            "file_path": self.file_path,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "signature": self.signature,
            "kind": self.kind.value,
            "decorators": list(self.decorators),
            "is_exported": self.is_exported,
            "complexity": self.complexity,
            "body_token_estimate": self.body_token_estimate,
            "sha256": self.sha256,
        }
        if self.docstring:
            d["docstring"] = self.docstring
        return d

    @classmethod
    def from_dict(cls, data: dict) -> SkeletonCore:
        """Deserialize from JSON storage."""
        return cls(
            fqn=data["fqn"],
            file_path=data["file_path"],
            line_start=data["line_start"],
            line_end=data["line_end"],
            signature=data["signature"],
            kind=NodeKind(data["kind"]),
            decorators=tuple(data.get("decorators", ())),
            is_exported=data.get("is_exported", False),
            complexity=data.get("complexity", 1),
            body_token_estimate=data.get("body_token_estimate", 0),
            docstring=data.get("docstring", ""),
            sha256=data.get("sha256", ""),
        )


@dataclass
class ClassSkeleton:
    """Class-level container grouping methods with class metadata.

    Carries information about the class 'shape' — inheritance, constructor
    params, instance attributes — things callers need without reading
    the full constructor body.
    """

    name: str
    fqn: str                          # e.g., "auth/middleware.py::AuthMiddleware"
    file_path: str

    bases: List[str] = field(default_factory=list)
    """Base classes / interfaces. Inheritance = dependency edge.
    If a base class changes, all subclasses may be affected."""

    constructor_params: List[str] = field(default_factory=list)
    """From __init__: ['app: ASGIApp', 'secret: str'].
    This is the class's 'shape' — what's needed to instantiate it."""

    instance_attrs: List[str] = field(default_factory=list)
    """Instance attributes: ['self.secret', 'self.algorithm'].
    State that all methods may reference."""

    class_attrs: List[str] = field(default_factory=list)
    """Class-level constants/attributes."""

    methods: List[SkeletonCore] = field(default_factory=list)
    """Method skeletons within this class."""

    decorators: List[str] = field(default_factory=list)
    """Class-level decorators: ['@dataclass', '@final']."""

    line_start: int = 0
    line_end: int = 0

    kind: NodeKind = NodeKind.CLASS
    signature: str = ""  # Reconstructed on the fly if needed

    def to_core(self) -> SkeletonCore:
        """Convert to SkeletonCore for indexing and ranking."""
        if not self.signature:
            bases_str = f"({', '.join(self.bases)})" if self.bases else ""
            sig = f"class {self.name}{bases_str}:"
        else:
            sig = self.signature

        return SkeletonCore(
            fqn=self.fqn,
            file_path=self.file_path,
            line_start=self.line_start,
            line_end=self.line_end,
            signature=sig,
            kind=self.kind,
            decorators=tuple(self.decorators),
            is_exported=True,
            complexity=1,
            body_token_estimate=0, # Class header itself is small
        )

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "fqn": self.fqn,
            "file_path": self.file_path,
            "bases": self.bases,
            "constructor_params": self.constructor_params,
            "instance_attrs": self.instance_attrs,
            "class_attrs": self.class_attrs,
            "methods": [m.to_dict() for m in self.methods],
            "decorators": self.decorators,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "kind": self.kind.value,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ClassSkeleton:
        return cls(
            name=data["name"],
            fqn=data["fqn"],
            file_path=data["file_path"],
            bases=data.get("bases", []),
            constructor_params=data.get("constructor_params", []),
            instance_attrs=data.get("instance_attrs", []),
            class_attrs=data.get("class_attrs", []),
            methods=[SkeletonCore.from_dict(m) for m in data.get("methods", [])],
            decorators=data.get("decorators", []),
            line_start=data.get("line_start", 0),
            line_end=data.get("line_end", 0),
            kind=NodeKind(data.get("kind", "class")),
        )


@dataclass
class FileSkeleton:
    """File-level container grouping classes and top-level functions.

    This is the 'directory entry' — knowing what a file contains without
    reading it. Imports define file-level dependency edges.
    """

    path: str                         # Relative path from project root
    summary: str = ""                 # 1-line file purpose (from summarizer)
    module_docstring: str = ""        # First line of module docstring

    imports: List[str] = field(default_factory=list)
    """Raw import statements. Used for cross-file edge construction.
    e.g., ['from jwt import decode', 'from .models import User']"""

    exports: List[str] = field(default_factory=list)
    """Public API surface. For Python: __all__ entries. For JS: export statements.
    e.g., ['AuthMiddleware', 'validate_token']"""

    classes: List[ClassSkeleton] = field(default_factory=list)
    functions: List[SkeletonCore] = field(default_factory=list)
    """Top-level functions (not inside a class)."""

    constants: List[Tuple[str, str]] = field(default_factory=list)
    """Public constants: [('MAX_TOKEN_AGE', '3600'), ('AUTH_HEADER', 'str')].
    Only UPPER_CASE by convention. ~2-3 tokens each."""

    total_lines: int = 0
    sha256: str = ""                  # File-level dirty tracking

    @property
    def all_skeletons(self) -> List[SkeletonCore]:
        """All SkeletonCore entries in this file (top-level funcs + classes + methods)."""
        result = list(self.functions)
        for cls in self.classes:
            result.append(cls.to_core())
            result.extend(cls.methods)
        return result

    @property
    def all_fqns(self) -> List[str]:
        """All FQNs defined in this file."""
        return [s.fqn for s in self.all_skeletons]

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "summary": self.summary,
            "module_docstring": self.module_docstring,
            "imports": self.imports,
            "exports": self.exports,
            "classes": [c.to_dict() for c in self.classes],
            "functions": [f.to_dict() for f in self.functions],
            "constants": self.constants,
            "total_lines": self.total_lines,
            "sha256": self.sha256,
        }

    @classmethod
    def from_dict(cls, data: dict) -> FileSkeleton:
        return cls(
            path=data["path"],
            summary=data.get("summary", ""),
            module_docstring=data.get("module_docstring", ""),
            imports=data.get("imports", []),
            exports=data.get("exports", []),
            classes=[ClassSkeleton.from_dict(c) for c in data.get("classes", [])],
            functions=[SkeletonCore.from_dict(f) for f in data.get("functions", [])],
            constants=[tuple(c) for c in data.get("constants", [])],
            total_lines=data.get("total_lines", 0),
            sha256=data.get("sha256", ""),
        )


# ── Utility ────────────────────────────────────────────────────────────────

def make_fqn(file_path: str, *parts: str) -> str:
    """Construct a fully-qualified name.

    Examples:
        make_fqn("auth/middleware.py", "AuthMiddleware", "validate_token")
        → "auth/middleware.py::AuthMiddleware.validate_token"

        make_fqn("utils.py", "helper")
        → "utils.py::helper"
    """
    name = ".".join(parts)
    return f"{file_path}::{name}"


def make_lambda_fqn(file_path: str, parent_fqn: str, line: int) -> str:
    """Construct FQN for anonymous functions.

    Example:
        make_lambda_fqn("utils.py", "utils.py::process_data", 45)
        → "utils.py::process_data.<lambda:45>"
    """
    parent_name = parent_fqn.split("::")[-1] if "::" in parent_fqn else parent_fqn
    return f"{file_path}::{parent_name}.<lambda:{line}>"
