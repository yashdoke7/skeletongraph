"""
TypeScript/JavaScript-specific AST extraction rules.

Handles:
  - function_declaration, arrow_function, method_definition
  - class_declaration with extends/implements
  - interface_declaration, type_alias_declaration
  - import_statement (ESM), require() (CJS)
  - export_statement (default, named)
  - Call expressions for edge extraction
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


def extract_typescript(
    file_path: str,
    source: str,
    source_bytes: bytes,
    tree: Tree,
    lang: str = "typescript",
) -> FileExtractionResult:
    """Extract all structural information from a TypeScript/JavaScript file."""

    result = FileExtractionResult(
        file_path=file_path,
        language=lang,
        total_lines=source.count("\n") + 1,
        file_hash=_hash_text(source),
    )

    root = tree.root_node

    # Walk top-level statements
    for child in root.children:
        _process_top_level(child, file_path, source_bytes, source, result)

    return result


def _process_top_level(
    node: Node,
    file_path: str,
    source_bytes: bytes,
    source: str,
    result: FileExtractionResult,
) -> None:
    """Process a top-level AST node."""

    if node.type == "function_declaration":
        fn = _extract_function_decl(node, file_path, source_bytes, source)
        if fn:
            result.functions.append(fn)

    elif node.type == "class_declaration":
        cls = _extract_class(node, file_path, source_bytes, source)
        if cls:
            result.classes.append(cls)

    elif node.type == "interface_declaration":
        cls = _extract_interface(node, file_path, source_bytes, source)
        if cls:
            result.classes.append(cls)

    elif node.type in ("type_alias_declaration",):
        fn = _extract_type_alias(node, file_path, source_bytes)
        if fn:
            result.functions.append(fn)

    elif node.type == "import_statement":
        imp = _extract_import(node, source_bytes)
        if imp:
            result.imports.append(imp)

    elif node.type == "export_statement":
        _process_export(node, file_path, source_bytes, source, result)

    elif node.type == "lexical_declaration":
        # const X = ... or let X = ...
        _process_lexical_declaration(node, file_path, source_bytes, source, result)

    elif node.type == "expression_statement":
        # module.exports = ... (CJS)
        _process_cjs_export(node, source_bytes, result)


def _extract_function_decl(
    node: Node,
    file_path: str,
    source_bytes: bytes,
    source: str,
    parent_class: Optional[str] = None,
    is_exported: bool = False,
) -> Optional[RawFunction]:
    """Extract from function_declaration node."""

    name_node = node.child_by_field_name("name")
    if not name_node:
        return None

    name = node_text(name_node, source_bytes)
    params_node = node.child_by_field_name("parameters")
    return_type = node.child_by_field_name("return_type")

    # Build signature
    full_text = node_text(node, source_bytes)
    is_async = full_text.strip().startswith("async")

    sig_parts = []
    if is_async:
        sig_parts.append("async ")
    sig_parts.append("function ")
    sig_parts.append(name)

    if params_node:
        sig_parts.append(node_text(params_node, source_bytes))
    else:
        sig_parts.append("()")

    if return_type:
        sig_parts.append(f": {node_text(return_type, source_bytes)}")

    signature = "".join(sig_parts)

    # Kind
    kind = NodeKind.FUNCTION
    if is_async:
        kind = NodeKind.ASYNC_FUNCTION
    if parent_class:
        kind = NodeKind.METHOD

    # FQN
    if parent_class:
        fqn = f"{file_path}::{parent_class}.{name}"
    else:
        fqn = f"{file_path}::{name}"

    # Body
    body_node = node.child_by_field_name("body")
    body_text = node_text(body_node, source_bytes) if body_node else ""

    return RawFunction(
        name=name,
        fqn=fqn,
        file_path=file_path,
        line_start=node.start_point[0] + 1,
        line_end=node.end_point[0] + 1,
        signature=signature,
        kind=kind,
        body_text=body_text,
        is_exported=is_exported or not name.startswith("_"),
        parent_class=parent_class,
    )


def _extract_method_def(
    node: Node,
    file_path: str,
    source_bytes: bytes,
    source: str,
    parent_class: str,
) -> Optional[RawFunction]:
    """Extract from method_definition node inside a class."""

    name_node = node.child_by_field_name("name")
    if not name_node:
        return None

    name = node_text(name_node, source_bytes)
    params_node = node.child_by_field_name("parameters")
    return_type = node.child_by_field_name("return_type")

    # Build signature
    full_text = node_text(node, source_bytes)

    sig_parts = []

    # Check for static/async/get/set
    is_static = False
    is_async = False
    is_getter = False
    is_setter = False

    for child in node.children:
        t = node_text(child, source_bytes)
        if t == "static":
            is_static = True
        elif t == "async":
            is_async = True
        elif t == "get":
            is_getter = True
        elif t == "set":
            is_setter = True

    if is_static:
        sig_parts.append("static ")
    if is_async:
        sig_parts.append("async ")
    if is_getter:
        sig_parts.append("get ")
    elif is_setter:
        sig_parts.append("set ")

    sig_parts.append(name)
    if params_node:
        sig_parts.append(node_text(params_node, source_bytes))
    if return_type:
        sig_parts.append(f": {node_text(return_type, source_bytes)}")

    signature = "".join(sig_parts)

    # Kind
    if name == "constructor":
        kind = NodeKind.CONSTRUCTOR
    elif is_static:
        kind = NodeKind.STATIC_METHOD
    elif is_getter or is_setter:
        kind = NodeKind.PROPERTY
    elif is_async:
        kind = NodeKind.ASYNC_FUNCTION
    else:
        kind = NodeKind.METHOD

    fqn = f"{file_path}::{parent_class}.{name}"

    body_node = node.child_by_field_name("body")
    body_text = node_text(body_node, source_bytes) if body_node else ""

    return RawFunction(
        name=name,
        fqn=fqn,
        file_path=file_path,
        line_start=node.start_point[0] + 1,
        line_end=node.end_point[0] + 1,
        signature=signature,
        kind=kind,
        body_text=body_text,
        parent_class=parent_class,
    )


def _extract_class(
    node: Node,
    file_path: str,
    source_bytes: bytes,
    source: str,
    is_exported: bool = False,
) -> Optional[RawClass]:
    """Extract class_declaration."""

    name_node = node.child_by_field_name("name")
    if not name_node:
        return None

    name = node_text(name_node, source_bytes)
    fqn = f"{file_path}::{name}"

    # Base classes (extends / implements)
    bases = []
    for child in node.children:
        if child.type == "class_heritage":
            for c in child.children:
                if c.type == "extends_clause":
                    for cc in c.children:
                        if cc.type == "identifier" or cc.type == "member_expression":
                            bases.append(node_text(cc, source_bytes))
                elif c.type == "implements_clause":
                    for cc in c.children:
                        if cc.type == "identifier" or cc.type == "type_identifier":
                            bases.append(node_text(cc, source_bytes))

    raw_cls = RawClass(
        name=name,
        fqn=fqn,
        file_path=file_path,
        line_start=node.start_point[0] + 1,
        line_end=node.end_point[0] + 1,
        bases=bases,
        kind=NodeKind.CLASS,
    )

    # Extract methods from class_body
    body = node.child_by_field_name("body")
    if body:
        for child in body.children:
            if child.type == "method_definition":
                fn = _extract_method_def(child, file_path, source_bytes, source, name)
                if fn:
                    raw_cls.methods.append(fn)
            elif child.type == "public_field_definition":
                # Class fields — treat as class attrs
                pass

    return raw_cls


def _extract_interface(
    node: Node,
    file_path: str,
    source_bytes: bytes,
    source: str,
) -> Optional[RawClass]:
    """Extract interface_declaration as a RawClass with INTERFACE kind."""

    name_node = node.child_by_field_name("name")
    if not name_node:
        return None

    name = node_text(name_node, source_bytes)
    fqn = f"{file_path}::{name}"

    # Extract extends
    bases = []
    for child in node.children:
        if child.type == "extends_type_clause":
            for c in child.children:
                if c.type in ("type_identifier", "identifier"):
                    bases.append(node_text(c, source_bytes))

    raw_cls = RawClass(
        name=name,
        fqn=fqn,
        file_path=file_path,
        line_start=node.start_point[0] + 1,
        line_end=node.end_point[0] + 1,
        bases=bases,
        kind=NodeKind.INTERFACE,
    )

    # Extract method signatures from interface body
    body = node.child_by_field_name("body")
    if body:
        for child in body.children:
            if child.type in ("method_signature", "property_signature"):
                name_n = child.child_by_field_name("name")
                if name_n:
                    method_name = node_text(name_n, source_bytes)
                    sig = node_text(child, source_bytes).strip().rstrip(";")
                    raw_cls.methods.append(RawFunction(
                        name=method_name,
                        fqn=f"{file_path}::{name}.{method_name}",
                        file_path=file_path,
                        line_start=child.start_point[0] + 1,
                        line_end=child.end_point[0] + 1,
                        signature=sig,
                        kind=NodeKind.METHOD,
                        parent_class=name,
                    ))

    return raw_cls


def _extract_type_alias(
    node: Node,
    file_path: str,
    source_bytes: bytes,
) -> Optional[RawFunction]:
    """Extract type alias: `type X = ...`"""

    name_node = node.child_by_field_name("name")
    if not name_node:
        return None

    name = node_text(name_node, source_bytes)
    sig = node_text(node, source_bytes).strip().rstrip(";")
    # Truncate very long type definitions
    if len(sig) > 200:
        sig = sig[:197] + "..."

    return RawFunction(
        name=name,
        fqn=f"{file_path}::{name}",
        file_path=file_path,
        line_start=node.start_point[0] + 1,
        line_end=node.end_point[0] + 1,
        signature=sig,
        kind=NodeKind.TYPE_ALIAS,
    )


def _extract_import(node: Node, source_bytes: bytes) -> Optional[RawImport]:
    """Extract ESM import: import { X } from 'module'"""

    module = ""
    names: List[str] = []
    aliases: Dict[str, str] = {}
    default_import = ""

    for child in node.children:
        if child.type == "string" or child.type == "string_fragment":
            module = node_text(child, source_bytes).strip("'\"")
        elif child.type == "import_clause":
            for c in child.children:
                if c.type == "identifier":
                    # Default import
                    default_import = node_text(c, source_bytes)
                    names.append(default_import)
                elif c.type == "named_imports":
                    for spec in c.children:
                        if spec.type == "import_specifier":
                            local_name = None
                            imported_name = None
                            for s in spec.children:
                                if s.type == "identifier":
                                    if imported_name is None:
                                        imported_name = node_text(s, source_bytes)
                                    else:
                                        local_name = node_text(s, source_bytes)
                            if imported_name:
                                names.append(imported_name)
                                if local_name:
                                    aliases[imported_name] = local_name
                elif c.type == "namespace_import":
                    # import * as X from 'module'
                    for s in c.children:
                        if s.type == "identifier":
                            alias_name = node_text(s, source_bytes)
                            names.append("*")
                            aliases["*"] = alias_name

    if not module:
        # Try getting string from direct child
        full_text = node_text(node, source_bytes)
        match = re.search(r"""['"](.+?)['"]""", full_text)
        if match:
            module = match.group(1)

    if not module:
        return None

    is_relative = module.startswith(".")

    return RawImport(
        module=module,
        names=names,
        aliases=aliases,
        is_relative=is_relative,
        line=node.start_point[0] + 1,
    )


def _process_export(
    node: Node,
    file_path: str,
    source_bytes: bytes,
    source: str,
    result: FileExtractionResult,
) -> None:
    """Process export_statement. May contain declarations within."""
    full_text = node_text(node, source_bytes)
    is_default = "default" in full_text.split("{")[0]  # crude check

    for child in node.children:
        if child.type == "function_declaration":
            fn = _extract_function_decl(child, file_path, source_bytes, source,
                                        is_exported=True)
            if fn:
                result.functions.append(fn)
                result.exports.append(fn.name)

        elif child.type == "class_declaration":
            cls = _extract_class(child, file_path, source_bytes, source,
                                 is_exported=True)
            if cls:
                result.classes.append(cls)
                result.exports.append(cls.name)

        elif child.type == "lexical_declaration":
            _process_lexical_declaration(
                child, file_path, source_bytes, source, result,
                is_exported=True,
            )

        elif child.type == "export_clause":
            # export { A, B, C }
            for spec in child.children:
                if spec.type == "export_specifier":
                    for s in spec.children:
                        if s.type == "identifier":
                            result.exports.append(node_text(s, source_bytes))

        elif child.type == "identifier":
            name = node_text(child, source_bytes)
            if name != "default" and name != "export":
                result.exports.append(name)


def _process_lexical_declaration(
    node: Node,
    file_path: str,
    source_bytes: bytes,
    source: str,
    result: FileExtractionResult,
    is_exported: bool = False,
) -> None:
    """Process const/let/var declarations. Arrow functions become RawFunction."""

    for child in node.children:
        if child.type == "variable_declarator":
            name_node = child.child_by_field_name("name")
            value_node = child.child_by_field_name("value")

            if not name_node or not value_node:
                continue

            name = node_text(name_node, source_bytes)

            # Arrow function: const X = (...) => { ... }
            if value_node.type == "arrow_function":
                params = value_node.child_by_field_name("parameters")
                ret_type = value_node.child_by_field_name("return_type")
                body = value_node.child_by_field_name("body")

                sig = f"const {name} = "
                if params:
                    sig += node_text(params, source_bytes)
                else:
                    sig += "()"
                if ret_type:
                    sig += f": {node_text(ret_type, source_bytes)}"
                sig += " =>"

                body_text = node_text(body, source_bytes) if body else ""

                fn = RawFunction(
                    name=name,
                    fqn=f"{file_path}::{name}",
                    file_path=file_path,
                    line_start=node.start_point[0] + 1,
                    line_end=node.end_point[0] + 1,
                    signature=sig,
                    kind=NodeKind.FUNCTION,
                    body_text=body_text,
                    is_exported=is_exported,
                )
                result.functions.append(fn)

                if is_exported:
                    result.exports.append(name)

            # Check for UPPER_CASE constants
            elif re.match(r"^[A-Z][A-Z0-9_]+$", name):
                value = node_text(value_node, source_bytes)
                if len(value) > 50:
                    value = value[:47] + "..."
                result.constants.append((name, value))

                if is_exported:
                    result.exports.append(name)


def _process_cjs_export(
    node: Node,
    source_bytes: bytes,
    result: FileExtractionResult,
) -> None:
    """Handle module.exports = { ... } CommonJS pattern."""
    if not node.children:
        return

    child = node.children[0]
    if child.type == "assignment_expression":
        left = child.child_by_field_name("left")
        right = child.child_by_field_name("right")
        if left:
            left_text = node_text(left, source_bytes)
            if left_text == "module.exports" and right:
                if right.type == "object":
                    for prop in right.children:
                        if prop.type == "shorthand_property_identifier":
                            result.exports.append(node_text(prop, source_bytes))
                        elif prop.type == "pair":
                            key = prop.child_by_field_name("key")
                            if key:
                                result.exports.append(node_text(key, source_bytes))


def extract_call_sites_ts(
    file_path: str,
    source_bytes: bytes,
    tree: Tree,
    function_ranges: List[Tuple[str, int, int]],
) -> List[RawCallSite]:
    """Extract call sites from TypeScript/JavaScript."""
    calls: List[RawCallSite] = []

    def _find_caller(line: int) -> str:
        best_fqn = f"{file_path}::__top_level__"
        best_start = 0
        for fqn, start, end in function_ranges:
            if start <= line <= end and start > best_start:
                best_fqn = fqn
                best_start = start
        return best_fqn

    def _walk(node: Node) -> None:
        if node.type == "call_expression":
            func_node = node.child_by_field_name("function")
            if func_node:
                callee = node_text(func_node, source_bytes)
                line = node.start_point[0] + 1
                caller = _find_caller(line)

                calls.append(RawCallSite(
                    caller_fqn=caller,
                    callee_name=callee,
                    line=line,
                ))

        elif node.type == "new_expression":
            # new ClassName(...)
            constructor = node.child_by_field_name("constructor")
            if constructor:
                callee = node_text(constructor, source_bytes)
                line = node.start_point[0] + 1
                caller = _find_caller(line)

                calls.append(RawCallSite(
                    caller_fqn=caller,
                    callee_name=callee,
                    line=line,
                    is_constructor=True,
                ))

        for child in node.children:
            _walk(child)

    _walk(tree.root_node)
    return calls
