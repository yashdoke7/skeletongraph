"""
Python-specific AST extraction rules.

Handles:
  - function_definition, async function_definition (decorated_definition wrapper)
  - class_definition with inheritance
  - import_statement, import_from_statement
  - Call expressions for edge extraction
  - __all__ for exports, UPPER_CASE for constants
  - Decorators, properties, static/class methods
  - Module-level docstrings
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Set, Tuple

from tree_sitter import Node, Tree

from ..ast_extractor import (
    FileExtractionResult,
    RawCallSite,
    RawClass,
    RawFunction,
    RawImport,
    _hash_text,
    count_branches,
    estimate_tokens,
    node_text,
)
from ..node_kinds import NodeKind


def extract_python(
    file_path: str,
    source: str,
    source_bytes: bytes,
    tree: Tree,
) -> FileExtractionResult:
    """Extract all structural information from a Python file."""

    result = FileExtractionResult(
        file_path=file_path,
        language="python",
        total_lines=source.count("\n") + 1,
        file_hash=_hash_text(source),
    )

    root = tree.root_node

    # Extract module docstring
    result.module_docstring = _extract_module_docstring(root, source_bytes)

    # Walk top-level statements
    for child in root.children:
        _process_top_level(child, file_path, source_bytes, source, result)

    return result


def _extract_module_docstring(root: Node, source_bytes: bytes) -> str:
    """Extract the first string expression as module docstring."""
    for child in root.children:
        if child.type == "expression_statement":
            expr = child.children[0] if child.children else None
            if expr and expr.type == "string":
                text = node_text(expr, source_bytes)
                # Strip quotes and take first line
                text = text.strip("'\"").strip()
                first_line = text.split("\n")[0].strip()
                return first_line[:200]  # Cap length
        elif child.type == "comment":
            continue
        else:
            break  # Non-docstring statement encountered
    return ""


def _process_top_level(
    node: Node,
    file_path: str,
    source_bytes: bytes,
    source: str,
    result: FileExtractionResult,
) -> None:
    """Process a top-level AST node."""

    if node.type == "function_definition":
        fn = _extract_function(node, file_path, source_bytes, source)
        if fn:
            result.functions.append(fn)

    elif node.type == "decorated_definition":
        # Unwrap decorators
        decorators = _extract_decorators(node, source_bytes)
        inner = None
        for child in node.children:
            if child.type in ("function_definition", "class_definition"):
                inner = child
                break

        if inner and inner.type == "function_definition":
            fn = _extract_function(inner, file_path, source_bytes, source, decorators)
            if fn:
                # Override line_start to include decorators
                fn.line_start = node.start_point[0] + 1
                result.functions.append(fn)
        elif inner and inner.type == "class_definition":
            cls = _extract_class(inner, file_path, source_bytes, source, decorators)
            if cls:
                cls.line_start = node.start_point[0] + 1
                result.classes.append(cls)

    elif node.type == "class_definition":
        cls = _extract_class(node, file_path, source_bytes, source)
        if cls:
            result.classes.append(cls)

    elif node.type == "import_statement":
        imp = _extract_import(node, source_bytes)
        if imp:
            result.imports.append(imp)

    elif node.type == "import_from_statement":
        imp = _extract_import_from(node, source_bytes)
        if imp:
            result.imports.append(imp)

    elif node.type == "expression_statement":
        # Check for __all__ = [...] or CONSTANT = value
        _process_expression_statement(node, source_bytes, result)


def _extract_function(
    node: Node,
    file_path: str,
    source_bytes: bytes,
    source: str,
    decorators: Optional[List[str]] = None,
    parent_class: Optional[str] = None,
) -> Optional[RawFunction]:
    """Extract a function/method from a function_definition node."""

    name_node = node.child_by_field_name("name")
    if not name_node:
        return None

    name = node_text(name_node, source_bytes)
    params_node = node.child_by_field_name("parameters")
    return_type = node.child_by_field_name("return_type")

    # Build signature
    sig_parts = []

    # Check if async
    is_async = False
    for child in node.children:
        if child.type == "async":
            is_async = True
            break
    # Actually check parent or the node itself
    # In tree-sitter python, async functions are still "function_definition"
    # but have "async" keyword before "def"
    full_text = node_text(node, source_bytes)
    if full_text.strip().startswith("async"):
        is_async = True

    if is_async:
        sig_parts.append("async ")
    sig_parts.append("def ")
    sig_parts.append(name)

    if params_node:
        sig_parts.append(node_text(params_node, source_bytes))
    else:
        sig_parts.append("()")

    if return_type:
        sig_parts.append(f" -> {node_text(return_type, source_bytes)}")

    sig_parts.append(":")
    signature = "".join(sig_parts)

    # Determine kind
    kind = NodeKind.FUNCTION
    decs = decorators or []

    if parent_class:
        if name == "__init__":
            kind = NodeKind.CONSTRUCTOR
        elif any("staticmethod" in d for d in decs):
            kind = NodeKind.STATIC_METHOD
        elif any("classmethod" in d for d in decs):
            kind = NodeKind.CLASS_METHOD
        elif any("property" in d for d in decs):
            kind = NodeKind.PROPERTY
        else:
            kind = NodeKind.METHOD

    if is_async and kind == NodeKind.FUNCTION:
        kind = NodeKind.ASYNC_FUNCTION

    if any("fixture" in d for d in decs):
        kind = NodeKind.FIXTURE

    # Build FQN
    if parent_class:
        fqn = f"{file_path}::{parent_class}.{name}"
    else:
        fqn = f"{file_path}::{name}"

    # Extract body text
    body_node = node.child_by_field_name("body")
    body_text = node_text(body_node, source_bytes) if body_node else ""

    # Compute complexity
    complexity = count_branches(node)

    return RawFunction(
        name=name,
        fqn=fqn,
        file_path=file_path,
        line_start=node.start_point[0] + 1,
        line_end=node.end_point[0] + 1,
        signature=signature,
        kind=kind,
        decorators=decs,
        body_text=body_text,
        is_exported=not name.startswith("_"),
        parent_class=parent_class,
    )


def _extract_class(
    node: Node,
    file_path: str,
    source_bytes: bytes,
    source: str,
    decorators: Optional[List[str]] = None,
) -> Optional[RawClass]:
    """Extract a class from a class_definition node."""

    name_node = node.child_by_field_name("name")
    if not name_node:
        return None

    name = node_text(name_node, source_bytes)
    fqn = f"{file_path}::{name}"

    # Extract base classes
    bases = []
    superclasses = node.child_by_field_name("superclasses")
    if superclasses:
        # argument_list contains the base classes
        for child in superclasses.children:
            if child.type == "identifier":
                bases.append(node_text(child, source_bytes))
            elif child.type == "attribute":
                bases.append(node_text(child, source_bytes))
            elif child.type == "keyword_argument":
                # metaclass=ABCMeta, skip
                pass

    # Determine kind
    decs = decorators or []
    kind = NodeKind.CLASS
    if any("abstract" in d.lower() for d in decs) or "ABC" in bases:
        kind = NodeKind.ABSTRACT_CLASS

    raw_cls = RawClass(
        name=name,
        fqn=fqn,
        file_path=file_path,
        line_start=node.start_point[0] + 1,
        line_end=node.end_point[0] + 1,
        bases=bases,
        decorators=decs,
        kind=kind,
    )

    # Extract methods
    body = node.child_by_field_name("body")
    if body:
        for child in body.children:
            if child.type == "function_definition":
                fn = _extract_function(child, file_path, source_bytes, source,
                                       parent_class=name)
                if fn:
                    raw_cls.methods.append(fn)

            elif child.type == "decorated_definition":
                method_decs = _extract_decorators(child, source_bytes)
                inner = None
                for c in child.children:
                    if c.type == "function_definition":
                        inner = c
                        break
                if inner:
                    fn = _extract_function(inner, file_path, source_bytes, source,
                                           decorators=method_decs, parent_class=name)
                    if fn:
                        fn.line_start = child.start_point[0] + 1
                        raw_cls.methods.append(fn)

    return raw_cls


def _extract_decorators(node: Node, source_bytes: bytes) -> List[str]:
    """Extract decorator names from a decorated_definition."""
    decorators = []
    for child in node.children:
        if child.type == "decorator":
            text = node_text(child, source_bytes).strip()
            # Normalize: @decorator(args) → @decorator
            match = re.match(r"@([\w.]+)", text)
            if match:
                decorators.append(f"@{match.group(1)}")
    return decorators


def _extract_import(node: Node, source_bytes: bytes) -> Optional[RawImport]:
    """Extract from `import X` or `import X as Y`."""
    names = []
    aliases: Dict[str, str] = {}

    for child in node.children:
        if child.type == "dotted_name":
            module = node_text(child, source_bytes)
            names.append(module)
        elif child.type == "aliased_import":
            name_node = None
            alias_node = None
            for c in child.children:
                if c.type == "dotted_name":
                    if name_node is None:
                        name_node = c
                elif c.type == "identifier":
                    alias_node = c
            if name_node:
                mod = node_text(name_node, source_bytes)
                names.append(mod)
                if alias_node:
                    aliases[mod] = node_text(alias_node, source_bytes)

    if not names:
        return None

    return RawImport(
        module=names[0],
        names=[],  # Plain import, no specific names
        aliases=aliases,
        line=node.start_point[0] + 1,
    )


def _extract_import_from(node: Node, source_bytes: bytes) -> Optional[RawImport]:
    """Extract from `from X import Y, Z` or `from . import X`.

    Tree-sitter AST structure:
      from datetime import datetime:
        import_from_statement
          from, dotted_name("datetime"), import, dotted_name("datetime")

      from .models import User:
        import_from_statement
          from, relative_import(.models), import, dotted_name("User")
    """
    module = ""
    names: List[str] = []
    aliases: Dict[str, str] = {}
    is_relative = False
    found_import_keyword = False  # Track if we've passed "import" keyword

    for child in node.children:
        if child.type == "import":
            # Everything after "import" keyword is imported names
            found_import_keyword = True
            continue

        if child.type == "from":
            continue

        if not found_import_keyword:
            # Before "import" keyword — this is the module part
            if child.type == "dotted_name":
                module = node_text(child, source_bytes)
            elif child.type == "relative_import":
                is_relative = True
                module = node_text(child, source_bytes)
            elif child.type == "import_prefix":
                is_relative = True
                module = node_text(child, source_bytes)
        else:
            # After "import" keyword — these are imported names
            if child.type == "dotted_name":
                names.append(node_text(child, source_bytes))
            elif child.type == "wildcard_import":
                names.append("*")
            elif child.type == "identifier":
                name = node_text(child, source_bytes)
                if name not in ("import", "from", "as"):
                    names.append(name)
            elif child.type == "aliased_import":
                name_node = None
                alias_node = None
                for c in child.children:
                    if c.type == "identifier":
                        if name_node is None:
                            name_node = c
                        elif alias_node is None:
                            alias_node = c
                if name_node:
                    name = node_text(name_node, source_bytes)
                    names.append(name)
                    if alias_node:
                        aliases[name] = node_text(alias_node, source_bytes)
            elif child.type == "import_list":
                # from X import A, B, C
                for item in child.children:
                    if item.type == "identifier":
                        names.append(node_text(item, source_bytes))
                    elif item.type == "dotted_name":
                        names.append(node_text(item, source_bytes))
                    elif item.type == "aliased_import":
                        n = None
                        a = None
                        for c in item.children:
                            if c.type == "identifier":
                                if n is None:
                                    n = c
                                elif a is None:
                                    a = c
                        if n:
                            nm = node_text(n, source_bytes)
                            names.append(nm)
                            if a:
                                aliases[nm] = node_text(a, source_bytes)

    if not module and not names:
        return None

    return RawImport(
        module=module,
        names=names,
        aliases=aliases,
        is_relative=is_relative,
        line=node.start_point[0] + 1,
    )


def _process_expression_statement(
    node: Node,
    source_bytes: bytes,
    result: FileExtractionResult,
) -> None:
    """Process expression_statement for __all__ and constants."""
    if not node.children:
        return

    child = node.children[0]

    if child.type == "assignment":
        left = child.child_by_field_name("left")
        right = child.child_by_field_name("right")

        if left and right:
            name = node_text(left, source_bytes)

            # __all__ = [...]
            if name == "__all__" and right.type == "list":
                for item in right.children:
                    if item.type == "string":
                        val = node_text(item, source_bytes).strip("'\"")
                        result.exports.append(val)

            # UPPER_CASE constants
            elif re.match(r"^[A-Z][A-Z0-9_]+$", name):
                value = node_text(right, source_bytes)
                # Truncate long values
                if len(value) > 50:
                    value = value[:47] + "..."
                result.constants.append((name, value))


def extract_call_sites(
    file_path: str,
    source_bytes: bytes,
    tree: Tree,
    function_ranges: List[Tuple[str, int, int]],
) -> List[RawCallSite]:
    """Extract all function call sites from a Python file.

    Args:
        file_path: Relative file path.
        source_bytes: UTF-8 encoded source.
        tree: Parsed tree-sitter tree.
        function_ranges: List of (fqn, line_start, line_end) to determine caller.

    Returns:
        List of RawCallSite entries.
    """
    calls: List[RawCallSite] = []

    def _find_caller(line: int) -> str:
        """Find which function a line belongs to."""
        best_fqn = f"{file_path}::__top_level__"
        best_start = 0
        for fqn, start, end in function_ranges:
            if start <= line <= end and start > best_start:
                best_fqn = fqn
                best_start = start
        return best_fqn

    def _walk_calls(node: Node) -> None:
        if node.type == "call":
            func_node = node.child_by_field_name("function")
            if func_node:
                callee = node_text(func_node, source_bytes)
                line = node.start_point[0] + 1
                caller = _find_caller(line)

                # Detect constructor calls (PascalCase)
                is_ctor = bool(re.match(r"^[A-Z][a-zA-Z0-9]+$", callee.split(".")[-1]))

                calls.append(RawCallSite(
                    caller_fqn=caller,
                    callee_name=callee,
                    line=line,
                    is_constructor=is_ctor,
                ))

        for child in node.children:
            _walk_calls(child)

    _walk_calls(tree.root_node)
    return calls
