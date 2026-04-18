"""
Core AST extraction engine using Tree-sitter.

Parses source files into structured data:
  - SkeletonCore entries (functions, methods, classes)
  - Raw edge data (call sites, imports, inheritance)
  - File metadata (imports, exports, constants)

Language-specific logic is delegated to modules in parser/languages/.
This module handles the common tree-walking infrastructure.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from tree_sitter import Language, Parser, Node

from .node_kinds import NodeKind
from .skeleton import ClassSkeleton, FileSkeleton, SkeletonCore, make_fqn


# ── Language Registry ──────────────────────────────────────────────────────

_PARSERS: Dict[str, Parser] = {}
_LANGUAGES: Dict[str, Language] = {}


def _get_parser(lang: str) -> Parser:
    """Get or create a tree-sitter parser for the given language."""
    if lang not in _PARSERS:
        if lang == "python":
            import tree_sitter_python as tspy
            _LANGUAGES[lang] = Language(tspy.language())
        elif lang in ("javascript", "js"):
            import tree_sitter_javascript as tsjs
            _LANGUAGES[lang] = Language(tsjs.language())
        elif lang in ("typescript", "ts", "tsx"):
            import tree_sitter_typescript as tsts
            if lang == "tsx":
                _LANGUAGES[lang] = Language(tsts.language_tsx())
            else:
                _LANGUAGES[lang] = Language(tsts.language_typescript())
        else:
            raise ValueError(f"Unsupported language: {lang}")

        _PARSERS[lang] = Parser(_LANGUAGES[lang])

    return _PARSERS[lang]


def detect_language(file_path: str) -> Optional[str]:
    """Detect language from file extension."""
    ext = Path(file_path).suffix.lower()
    return {
        ".py": "python",
        ".pyw": "python",
        ".js": "javascript",
        ".mjs": "javascript",
        ".cjs": "javascript",
        ".jsx": "javascript",
        ".ts": "typescript",
        ".tsx": "tsx",
    }.get(ext)


# ── Raw Extraction Data ────────────────────────────────────────────────────

@dataclass
class RawFunction:
    """Raw function data extracted from AST before SkeletonCore construction."""
    name: str
    fqn: str
    file_path: str
    line_start: int
    line_end: int
    signature: str
    kind: NodeKind
    decorators: List[str] = field(default_factory=list)
    body_text: str = ""
    is_exported: bool = False
    parent_class: Optional[str] = None


@dataclass
class RawClass:
    """Raw class data extracted from AST."""
    name: str
    fqn: str
    file_path: str
    line_start: int
    line_end: int
    bases: List[str] = field(default_factory=list)
    decorators: List[str] = field(default_factory=list)
    methods: List[RawFunction] = field(default_factory=list)
    kind: NodeKind = NodeKind.CLASS


@dataclass
class RawCallSite:
    """A function call found in the source code."""
    caller_fqn: str
    callee_name: str        # May be simple name or dotted: "obj.method"
    line: int
    is_constructor: bool = False  # True if `ClassName(...)` or `new ClassName(...)`


@dataclass
class RawImport:
    """An import statement found in the source code."""
    module: str            # "jwt", "os.path", "./utils"
    names: List[str]       # ["decode", "encode"] or ["*"] or [] for `import X`
    aliases: Dict[str, str] = field(default_factory=dict)  # {"decode": "jwt_decode"}
    is_relative: bool = False
    line: int = 0


@dataclass
class FileExtractionResult:
    """Complete extraction result for a single file."""
    file_path: str
    language: str
    functions: List[RawFunction] = field(default_factory=list)
    classes: List[RawClass] = field(default_factory=list)
    imports: List[RawImport] = field(default_factory=list)
    call_sites: List[RawCallSite] = field(default_factory=list)
    exports: List[str] = field(default_factory=list)
    constants: List[Tuple[str, str]] = field(default_factory=list)
    top_level_code: bool = False
    module_docstring: str = ""
    total_lines: int = 0
    file_hash: str = ""


# ── Common Tree Utilities ──────────────────────────────────────────────────

def node_text(node: Node, source: bytes) -> str:
    """Extract text content of a node."""
    return source[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def count_branches(node: Node) -> int:
    """Count branching statements for cyclomatic complexity.

    Counts: if, elif, for, while, except, case, and, or, ternary.
    Base complexity is 1.
    """
    count = 0
    branch_types = {
        "if_statement", "elif_clause", "for_statement", "while_statement",
        "except_clause", "case_clause", "conditional_expression",
        "for_in_statement", "if_expression", "match_arm",
        # JS/TS
        "if_statement", "for_statement", "while_statement", "catch_clause",
        "switch_case", "ternary_expression",
    }
    boolean_ops = {"and", "or", "&&", "||"}

    def _walk(n: Node) -> None:
        nonlocal count
        if n.type in branch_types:
            count += 1
        if n.type in ("boolean_operator", "binary_expression"):
            op_text = ""
            for child in n.children:
                if child.type in ("and", "or"):
                    count += 1
                # JS/TS: && and || are in the operator
        for child in n.children:
            _walk(child)

    _walk(node)
    return count + 1  # Base complexity


def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token (GPT tokenizer average)."""
    return max(1, len(text) // 4)


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ── Core Extraction ────────────────────────────────────────────────────────

def extract_file(
    file_path: str,
    project_root: Path,
    source: Optional[str] = None,
) -> Optional[FileExtractionResult]:
    """Extract all structural information from a source file.

    Args:
        file_path: Relative path from project root.
        project_root: Absolute path to project root.
        source: File content (if None, reads from disk).

    Returns:
        FileExtractionResult or None if language is not supported.
    """
    lang = detect_language(file_path)
    if lang is None:
        return None

    full_path = project_root / file_path

    if source is None:
        if not full_path.exists():
            return None
        source = full_path.read_text(encoding="utf-8", errors="replace")

    source_bytes = source.encode("utf-8")
    parser = _get_parser(lang)
    tree = parser.parse(source_bytes)

    # Delegate to language-specific extractor
    if lang == "python":
        from .languages.python import extract_python
        return extract_python(file_path, source, source_bytes, tree)
    elif lang in ("javascript", "typescript", "tsx"):
        from .languages.typescript import extract_typescript
        return extract_typescript(file_path, source, source_bytes, tree, lang)
    else:
        return None


def result_to_file_skeleton(result: FileExtractionResult) -> FileSkeleton:
    """Convert a FileExtractionResult into a FileSkeleton with SkeletonCore entries."""

    # Build class skeletons
    class_skeletons = []
    for raw_cls in result.classes:
        methods = []
        for raw_fn in raw_cls.methods:
            sk = _raw_to_skeleton(raw_fn)
            methods.append(sk)

        cls_sk = ClassSkeleton(
            name=raw_cls.name,
            fqn=raw_cls.fqn,
            file_path=result.file_path,
            bases=raw_cls.bases,
            decorators=raw_cls.decorators,
            methods=methods,
            line_start=raw_cls.line_start,
            line_end=raw_cls.line_end,
            kind=raw_cls.kind,
        )

        # Extract constructor params and instance attrs
        for m in raw_cls.methods:
            if m.kind == NodeKind.CONSTRUCTOR:
                # Parse constructor params from signature
                match = re.search(r"\((.+?)\)", m.signature)
                if match:
                    params = [
                        p.strip() for p in match.group(1).split(",")
                        if p.strip() and p.strip() != "self" and p.strip() != "cls"
                    ]
                    cls_sk.constructor_params = params

                # Extract instance attrs from body (self.X = ...)
                attr_pattern = re.compile(r"self\.(\w+)\s*=")
                attrs = attr_pattern.findall(m.body_text)
                cls_sk.instance_attrs = [f"self.{a}" for a in dict.fromkeys(attrs)]

        class_skeletons.append(cls_sk)

    # Build top-level function skeletons
    func_skeletons = [_raw_to_skeleton(f) for f in result.functions]

    # Import strings
    import_strs = []
    for imp in result.imports:
        if imp.names:
            names = ", ".join(imp.names)
            import_strs.append(f"from {imp.module} import {names}")
        else:
            import_strs.append(f"import {imp.module}")

    return FileSkeleton(
        path=result.file_path,
        module_docstring=result.module_docstring,
        imports=import_strs,
        exports=result.exports,
        classes=class_skeletons,
        functions=func_skeletons,
        constants=result.constants,
        total_lines=result.total_lines,
        sha256=result.file_hash,
    )


def _raw_to_skeleton(raw: RawFunction) -> SkeletonCore:
    """Convert RawFunction to SkeletonCore."""
    return SkeletonCore(
        fqn=raw.fqn,
        file_path=raw.file_path,
        line_start=raw.line_start,
        line_end=raw.line_end,
        signature=raw.signature,
        kind=raw.kind,
        decorators=tuple(raw.decorators),
        is_exported=raw.is_exported,
        complexity=count_branches(_DUMMY_NODE) if not raw.body_text else 1,
        body_token_estimate=estimate_tokens(raw.body_text),
        sha256=_hash_text(raw.body_text) if raw.body_text else "",
    )


# Placeholder for when we don't have the AST node during conversion
class _DummyNode:
    type = ""
    children = []
    start_byte = 0
    end_byte = 0

_DUMMY_NODE = _DummyNode()
