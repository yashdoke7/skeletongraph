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


def _extract_jsdoc_before(node: Node, source_bytes: bytes) -> str:
    """Extract the first line of a JSDoc comment preceding a node.

    Looks at the previous sibling for a /** ... */ comment block.
    Returns the first meaningful line, capped at 200 chars.
    """
    prev = node.prev_named_sibling
    if prev and prev.type == "comment":
        text = node_text(prev, source_bytes).strip()
        if text.startswith("/**"):
            # Strip /** and */ delimiters
            text = text.lstrip("/").lstrip("*").rstrip("*").rstrip("/").strip()
            # Take the first non-empty line
            for line in text.split("\n"):
                line = line.strip().lstrip("*").strip()
                if line and not line.startswith("@"):
                    return line[:200]
    return ""


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

    # CommonJS / prototype method assignments anywhere in the file — including
    # methods attached inside a `module.exports = function (App) { App.x = ... }`
    # factory (the dominant NodeBB pattern), which top-level walking can't reach.
    _walk_member_assign_fns(root, file_path, source_bytes, result)

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
        _process_cjs_export(node, file_path, source_bytes, result)


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

    # JSDoc docstring
    docstring = _extract_jsdoc_before(node, source_bytes)

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
        docstring=docstring,
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

    # JSDoc docstring
    docstring = _extract_jsdoc_before(node, source_bytes)

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
        docstring=docstring,
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


def _wrapped_function_arg(call_node: Node) -> Optional[Node]:
    """For an HOC call like forwardRef(fn) / memo(fn) / observer(fn) / React.memo(fn),
    return the arrow/function passed as a direct argument, so the bound name gets
    indexed as a component. Returns None for non-function calls (e.g.
    connect(...)(Comp), where the component is a reference defined elsewhere)."""
    args = call_node.child_by_field_name("arguments")
    if not args:
        return None
    for a in args.children:
        if a.type in ("arrow_function", "function_expression"):
            return a
    return None


def _fn_value(value_node: Node, source_bytes: bytes) -> Tuple[Optional[Node], str]:
    """If a value is a function/arrow or an HOC-wrapped function (forwardRef/memo),
    return (function_node, wrapper_text); else (None, "")."""
    if value_node.type in ("arrow_function", "function_expression"):
        return value_node, ""
    if value_node.type == "call_expression":
        inner = _wrapped_function_arg(value_node)
        if inner is not None:
            callee = value_node.child_by_field_name("function")
            return inner, (node_text(callee, source_bytes) if callee else "")
    return None, ""


def _raw_fn(name: str, display: str, fn_node: Node, decl_node: Node,
            file_path: str, source_bytes: bytes, wrapper: str = "",
            is_exported: bool = False) -> RawFunction:
    """Build a RawFunction from an arrow/function node bound to `name`."""
    params = fn_node.child_by_field_name("parameters")
    ret_type = fn_node.child_by_field_name("return_type")
    body = fn_node.child_by_field_name("body")
    sig = f"{display} = "
    if wrapper:
        sig += f"{wrapper}("
    sig += node_text(params, source_bytes) if params else "()"
    if ret_type:
        sig += node_text(ret_type, source_bytes)
    if fn_node.type == "arrow_function":
        sig += " =>"
    if wrapper:
        sig += ")"
    return RawFunction(
        name=name, fqn=f"{file_path}::{name}", file_path=file_path,
        line_start=decl_node.start_point[0] + 1,
        line_end=decl_node.end_point[0] + 1,
        signature=sig, kind=NodeKind.FUNCTION,
        body_text=node_text(body, source_bytes) if body else "",
        is_exported=is_exported,
    )


def _member_prop_name(left: Node, source_bytes: bytes) -> str:
    """Rightmost identifier of a member_expression: 'User.getAvatar' -> 'getAvatar',
    'module.exports.foo' -> 'foo'."""
    if left.type == "member_expression":
        prop = left.child_by_field_name("property")
        if prop:
            return node_text(prop, source_bytes)
    return node_text(left, source_bytes)


def _walk_member_assign_fns(root: Node, file_path: str, source_bytes: bytes,
                            result: FileExtractionResult) -> None:
    """Index `Obj.method = function(){}` / `exports.x = () => {}` assignments
    anywhere in the file — including methods attached inside a
    `module.exports = function (App) { App.x = ... }` factory, the dominant
    CommonJS pattern (NodeBB etc.). Without this they are never indexed and so
    never retrievable. JSX props / object literals are NOT assignment_expressions,
    so this stays free of inline-callback noise."""
    seen = {(f.line_start, f.name) for f in result.functions}
    stack = [root]
    while stack:
        n = stack.pop()
        if n.type == "assignment_expression":
            left = n.child_by_field_name("left")
            right = n.child_by_field_name("right")
            if left is not None and left.type == "member_expression" and right is not None:
                fn_node, wrapper = _fn_value(right, source_bytes)
                if fn_node is not None:
                    name = _member_prop_name(left, source_bytes)
                    display = node_text(left, source_bytes)
                    # bare `module.exports = fn` is handled by _process_cjs_export
                    # (named by file stem); indexing it here too would add a
                    # spurious "exports" function.
                    if display != "module.exports":
                        is_exp = display.startswith(("module.exports", "exports."))
                        key = (n.start_point[0] + 1, name)
                        if name and key not in seen:
                            seen.add(key)
                            result.functions.append(
                                _raw_fn(name, display, fn_node, n, file_path,
                                        source_bytes, wrapper, is_exp))
                            if is_exp:
                                result.exports.append(name)
        stack.extend(n.children)


def _process_lexical_declaration(
    node: Node,
    file_path: str,
    source_bytes: bytes,
    source: str,
    result: FileExtractionResult,
    is_exported: bool = False,
) -> None:
    """Process const/let/var declarations. Function-valued bindings (incl.
    HOC-wrapped React components) become RawFunction so they are retrievable."""

    for child in node.children:
        if child.type == "variable_declarator":
            name_node = child.child_by_field_name("name")
            value_node = child.child_by_field_name("value")

            if not name_node or not value_node:
                continue

            name = node_text(name_node, source_bytes)

            # A const binding holds a function/component when its value is a bare
            # arrow, a function expression, or an HOC-wrapped function
            # (forwardRef/memo/observer) — the wrapped case is essential for
            # modern TS/JS UIs (element-web etc.) where most components are
            # forwardRef/memo; missing it leaves them unindexed ("avatar" gap).
            fn_node, wrapper = _fn_value(value_node, source_bytes)
            if fn_node is not None:
                result.functions.append(
                    _raw_fn(name, f"const {name}", fn_node, node, file_path,
                            source_bytes, wrapper, is_exported))
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
    file_path: str,
    source_bytes: bytes,
    result: FileExtractionResult,
) -> None:
    """Handle CommonJS `module.exports = ...` (object, single function, or HOC)."""
    if not node.children:
        return
    child = node.children[0]
    if child.type != "assignment_expression":
        return
    left = child.child_by_field_name("left")
    right = child.child_by_field_name("right")
    if not left or not right:
        return
    if node_text(left, source_bytes) != "module.exports":
        return

    # module.exports = { ... }  → export names + index function-valued props.
    if right.type == "object":
        for prop in right.children:
            if prop.type == "shorthand_property_identifier":
                result.exports.append(node_text(prop, source_bytes))
            elif prop.type == "pair":
                key = prop.child_by_field_name("key")
                val = prop.child_by_field_name("value")
                if not key:
                    continue
                kname = node_text(key, source_bytes)
                result.exports.append(kname)
                if val is not None:
                    fn_node, wrapper = _fn_value(val, source_bytes)
                    if fn_node is not None and kname:
                        result.functions.append(
                            _raw_fn(kname, f"module.exports.{kname}", fn_node,
                                    prop, file_path, source_bytes, wrapper,
                                    is_exported=True))
        return

    # module.exports = function (...) { ... }  → index the whole module factory,
    # named by the file stem so it's retrievable (NodeBB-style modules).
    fn_node, wrapper = _fn_value(right, source_bytes)
    if fn_node is not None:
        stem = file_path.replace("\\", "/").rsplit("/", 1)[-1].rsplit(".", 1)[0] or "module"
        result.functions.append(
            _raw_fn(stem, "module.exports", fn_node, child, file_path,
                    source_bytes, wrapper, is_exported=True))
        result.exports.append(stem)


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

    # Iterative walk — TypeScript ASTs (especially bundled / generated files like
    # tutanota's flow→TS pipeline) can be deeper than Python's default 1000-frame
    # recursion limit. Stack-based avoids RecursionError without needing
    # setrecursionlimit (which would just shift the limit and burn memory).
    stack: list = [tree.root_node]
    while stack:
        node = stack.pop()
        if node.type == "call_expression":
            func_node = node.child_by_field_name("function")
            if func_node:
                callee = node_text(func_node, source_bytes)
                line = node.start_point[0] + 1
                caller = _find_caller(line)
                calls.append(RawCallSite(
                    caller_fqn=caller, callee_name=callee, line=line,
                ))
        elif node.type == "new_expression":
            constructor = node.child_by_field_name("constructor")
            if constructor:
                callee = node_text(constructor, source_bytes)
                line = node.start_point[0] + 1
                caller = _find_caller(line)
                calls.append(RawCallSite(
                    caller_fqn=caller, callee_name=callee, line=line,
                    is_constructor=True,
                ))
        # Push children in reverse so they're popped in source order (cosmetic,
        # but it makes the order match the previous recursive version).
        stack.extend(reversed(node.children))
    return calls
