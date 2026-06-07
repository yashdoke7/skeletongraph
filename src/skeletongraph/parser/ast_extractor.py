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
        elif lang == "java":
            import tree_sitter_java as tsjava
            _LANGUAGES[lang] = Language(tsjava.language())
        elif lang == "go":
            import tree_sitter_go as tsgo
            _LANGUAGES[lang] = Language(tsgo.language())
        elif lang == "rust":
            import tree_sitter_rust as tsrust
            _LANGUAGES[lang] = Language(tsrust.language())
        elif lang in ("cpp", "c++", "c"):
            import tree_sitter_cpp as tscpp
            _LANGUAGES[lang] = Language(tscpp.language())
        elif lang in ("csharp", "cs", "c#"):
            import tree_sitter_c_sharp as tscsharp
            _LANGUAGES[lang] = Language(tscsharp.language())
        elif lang == "ruby":
            import tree_sitter_ruby as tsruby
            _LANGUAGES[lang] = Language(tsruby.language())
        elif lang == "php":
            import tree_sitter_php as tsphp
            _LANGUAGES[lang] = Language(tsphp.language_php())
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
        ".java": "java",
        ".go": "go",
        ".rs": "rust",
        ".cpp": "cpp",
        ".c": "cpp",
        ".h": "cpp",
        ".hpp": "cpp",
        ".cs": "csharp",
        ".rb": "ruby",
        ".php": "php",
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
    docstring: str = ""  # First line of docstring (for search indexing)


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
    signature: str = ""


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

    # Iterative — see typescript.py::extract_call_sites_ts for rationale.
    # Deep TS/JS files blow Python's recursion limit on a pure-recursive walk.
    stack: list = [node]
    while stack:
        n = stack.pop()
        if n.type in branch_types:
            count += 1
        if n.type in ("boolean_operator", "binary_expression"):
            for child in n.children:
                if child.type in ("and", "or"):
                    count += 1
        stack.extend(n.children)
    return count + 1  # Base complexity


from ..eval.token_counter import measure_text_tokens

def estimate_tokens(text: str) -> int:
    """Precise token estimate using project's tiktoken implementation."""
    return measure_text_tokens(text)


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
    elif lang == "java":
        from .languages.java import extract_java
        return extract_java(file_path, source, source_bytes, tree)
    elif lang == "go":
        from .languages.go import extract_go
        return extract_go(file_path, source, source_bytes, tree)
    elif lang == "rust":
        from .languages.rust import extract_rust
        return extract_rust(file_path, source, source_bytes, tree)
    elif lang == "cpp":
        from .languages.cpp import extract_cpp
        return extract_cpp(file_path, source, source_bytes, tree)
    elif lang == "csharp":
        from .languages.csharp import extract_csharp
        return extract_csharp(file_path, source, source_bytes, tree)
    elif lang == "ruby":
        from .languages.ruby import extract_ruby
        return extract_ruby(file_path, source, source_bytes, tree)
    elif lang == "php":
        from .languages.php import extract_php
        return extract_php(file_path, source, source_bytes, tree)
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
            signature=raw_cls.signature,
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

                # Extract instance attrs from body (self.X = ... or this.X = ...)
                attr_pattern = re.compile(r"(?:self|this)\.(\w+)\s*=")
                attrs = attr_pattern.findall(m.body_text)
                cls_sk.instance_attrs = [f"self.{a}" if "self." in m.body_text else f"this.{a}" for a in dict.fromkeys(attrs)]

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
        complexity=_estimate_complexity_from_text(raw.body_text),
        body_token_estimate=estimate_tokens(raw.body_text),
        docstring=raw.docstring,
        sha256=_hash_text(raw.body_text) if raw.body_text else "",
    )


def _estimate_complexity_from_text(body_text: str) -> int:
    """Estimate cyclomatic complexity from raw body text.

    Since we don't retain the AST node at conversion time, we use a
    keyword-counting heuristic. Base complexity is 1; each branch
    keyword adds 1.
    """
    if not body_text:
        return 1  # No body → trivial complexity

    import re
    # Branch keywords across Python, JS/TS, Go, Rust, Java, C++, C#, Ruby, PHP
    branch_keywords = re.findall(
        r'\b(if|elif|else if|elseif|for|while|except|catch|case|match|when|'
        r'and|or|&&|\|\|)\b',
        body_text,
    )
    return 1 + len(branch_keywords)
