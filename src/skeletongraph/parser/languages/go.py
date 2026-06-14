"""
Go tree-sitter AST extraction rules.
"""
from __future__ import annotations

from typing import List, Optional, Tuple, Dict
from tree_sitter import Node, Tree
from pathlib import Path

from ..ast_extractor import (
    FileExtractionResult,
    RawFunction,
    RawClass,
    RawImport,
    RawCallSite,
    node_text,
    count_branches,
    estimate_tokens,
    _hash_text
)
from ..node_kinds import NodeKind

def _get_child_by_type(node: Node, child_type: str) -> Optional[Node]:
    for child in node.children:
        if child.type == child_type:
            return child
    return None

def _get_name(node: Node, source_bytes: bytes) -> str:
    # tree-sitter-go names live under DIFFERENT node types per declaration:
    #   function_declaration -> `identifier`        (e.g. NewHTTPServer)
    #   type_spec (struct/iface) -> `type_identifier`  (e.g. HTTPServer)
    #   method_declaration -> `field_identifier`    (e.g. Start)
    # The old identifier-only lookup returned "" for the latter two, so EVERY Go
    # struct, interface, and receiver method was silently dropped from the index
    # — only free functions survived → ~0 recall on method-heavy Go (the bulk of
    # real Go code). Order matters: method_declaration ALSO has a direct
    # `type_identifier` child for its return type, so `field_identifier` must be
    # tried before `type_identifier` or methods would be named after their return.
    for nt in ("identifier", "field_identifier", "type_identifier"):
        name_node = _get_child_by_type(node, nt)
        if name_node:
            return node_text(name_node, source_bytes)
    return ""

def _extract_go_doc(node: Node, source_bytes: bytes) -> str:
    """Extract Go doc comment (// lines preceding a declaration)."""
    prev = node.prev_named_sibling
    if prev and prev.type == "comment":
        text = node_text(prev, source_bytes).strip()
        if text.startswith("//"):
            return text.lstrip("/").strip()[:200]
    return ""

def extract_go(file_path: str, source: str, source_bytes: bytes, tree: Tree) -> Optional[FileExtractionResult]:
    root_node = tree.root_node
    
    functions: List[RawFunction] = []
    classes: List[RawClass] = []
    imports: List[RawImport] = []
    call_sites: List[RawCallSite] = []
    constants: List[Tuple[str, str]] = []

    package_name = ""
    # Find package clause
    pkg_node = _get_child_by_type(root_node, "package_clause")
    if pkg_node:
        package_name = _get_name(pkg_node, source_bytes)
    
    # FQN prefix MUST be the file_path (canonical make_fqn format,
    # "<file_path>::<symbol>") so `fqn.split("::")[0]` yields the real file for
    # recall scoring, read_symbol(file::symbol), and the agent's file resolution.
    # Using the Go package name here (the old behavior) produced FQNs like
    # "server::Upload", whose split prefix "server" never matches gold file paths
    # like "internal/server/evaluation.go" → 0 recall on EVERY Go task. The
    # package name is kept as metadata on the result, not in the FQN.
    base_fqn = file_path

    def traverse(node: Node):
        if node.type == "import_declaration":
            # Imports can be single or block
            for child in node.children:
                if child.type == "import_spec":
                    path_node = _get_child_by_type(child, "interpreted_string_literal")
                    if path_node:
                        # strip quotes
                        path = node_text(path_node, source_bytes).strip('"')
                        name_node = _get_child_by_type(child, "package_identifier")
                        alias = node_text(name_node, source_bytes) if name_node else path.split('/')[-1]
                        
                        imports.append(RawImport(
                            module=path,
                            names=[],  # Go imports whole packages
                            aliases={alias: path},
                            line=node.start_point[0] + 1
                        ))

        elif node.type == "type_declaration":
            for spec in node.children:
                if spec.type == "type_spec":
                    struct_name = _get_name(spec, source_bytes)
                    if struct_name:
                        type_node = _get_child_by_type(spec, "struct_type") or _get_child_by_type(spec, "interface_type")
                        kind = NodeKind.STRUCT if type_node and type_node.type == "struct_type" else NodeKind.INTERFACE

                        cls = RawClass(
                            name=struct_name,
                            fqn=f"{base_fqn}::{struct_name}",
                            file_path=file_path,
                            line_start=spec.start_point[0] + 1,
                            line_end=spec.end_point[0] + 1,
                            kind=kind
                        )
                        # Interface method specs are the API contract — index them
                        # so a search for the method name finds the interface.
                        if type_node and type_node.type == "interface_type":
                            for m in type_node.children:
                                if m.type not in ("method_elem", "method_spec"):
                                    continue
                                mn = _get_child_by_type(m, "field_identifier")
                                if not mn:
                                    continue
                                mname = node_text(mn, source_bytes)
                                cls.methods.append(RawFunction(
                                    name=mname,
                                    fqn=f"{base_fqn}::{struct_name}.{mname}",
                                    file_path=file_path,
                                    line_start=m.start_point[0] + 1,
                                    line_end=m.end_point[0] + 1,
                                    signature=node_text(m, source_bytes)[:120],
                                    kind=NodeKind.METHOD,
                                    body_text="",
                                    is_exported=bool(mname) and mname[0].isupper(),
                                    parent_class=struct_name,
                                ))
                        classes.append(cls)
                        
        elif node.type == "function_declaration":
            func_name = _get_name(node, source_bytes)
            if func_name:
                sig_node = _get_child_by_type(node, "signature")
                sig = ""
                if sig_node:
                    sig = node_text(sig_node, source_bytes)
                
                body = ""
                body_node = _get_child_by_type(node, "block")
                if body_node:
                    body = node_text(body_node, source_bytes)
                    
                is_exported = func_name[0].isupper()
                
                functions.append(RawFunction(
                    name=func_name,
                    fqn=f"{base_fqn}::{func_name}",
                    file_path=file_path,
                    line_start=node.start_point[0] + 1,
                    line_end=node.end_point[0] + 1,
                    signature=f"func {func_name}{sig}",
                    kind=NodeKind.FUNCTION,
                    body_text=body,
                    is_exported=is_exported,
                    docstring=_extract_go_doc(node, source_bytes),
                ))
                
        elif node.type == "method_declaration":
            func_name = _get_name(node, source_bytes)
            if func_name:
                # Extract receiver FQN
                receiver_node = _get_child_by_type(node, "parameter_list")
                receiver_type = ""
                if receiver_node:
                    for p in receiver_node.children:
                        if p.type == "parameter_declaration":
                            type_id = _get_child_by_type(p, "type_identifier")
                            if not type_id: # pointer type
                                ptr_node = _get_child_by_type(p, "pointer_type")
                                if ptr_node:
                                    type_id = _get_child_by_type(ptr_node, "type_identifier")
                            if type_id:
                                receiver_type = node_text(type_id, source_bytes)
                                break
                
                sig_node = _get_child_by_type(node, "signature")
                sig = node_text(sig_node, source_bytes) if sig_node else ""
                
                body = ""
                body_node = _get_child_by_type(node, "block")
                if body_node:
                    body = node_text(body_node, source_bytes)
                    
                is_exported = func_name[0].isupper()
                parent_class = receiver_type if receiver_type else None
                fqn = f"{base_fqn}::{receiver_type}.{func_name}" if receiver_type else f"{base_fqn}::{func_name}"
                
                func = RawFunction(
                    name=func_name,
                    fqn=fqn,
                    file_path=file_path,
                    line_start=node.start_point[0] + 1,
                    line_end=node.end_point[0] + 1,
                    signature=f"func {(receiver_type)} {func_name}{sig}",
                    kind=NodeKind.METHOD,
                    body_text=body,
                    is_exported=is_exported,
                    parent_class=parent_class,
                    docstring=_extract_go_doc(node, source_bytes),
                )
                
                if parent_class:
                    found_class = False
                    for cls in classes:
                        if cls.name == parent_class:
                            cls.methods.append(func)
                            found_class = True
                            break
                    if not found_class:
                        # Append as loose function if struct isn't derived yet
                        functions.append(func)
                else:
                    functions.append(func)

        # Package-level const/var declarations (config values, ErrX sentinels,
        # defaults) — indexed as constants so the file is retrievable by name.
        # Guard on source_file parent so local vars inside functions are skipped.
        elif (node.type in ("const_declaration", "var_declaration")
              and node.parent is not None
              and node.parent.type == "source_file"):
            for spec in node.children:
                if spec.type in ("const_spec", "var_spec"):
                    for c in spec.children:
                        if c.type == "identifier":
                            cname = node_text(c, source_bytes)
                            if cname and cname != "_":
                                constants.append(
                                    (cname, node_text(spec, source_bytes)[:80]))

        # Call site extraction logic
        elif node.type == "call_expression":
            fn = _get_child_by_type(node, "identifier") or _get_child_by_type(node, "selector_expression")
            if fn:
                callee_name = node_text(fn, source_bytes)
                call_sites.append(RawCallSite(
                    caller_fqn="", # Edge extractor resolves this based on line number
                    callee_name=callee_name,
                    line=node.start_point[0] + 1
                ))

        for child in node.children:
            traverse(child)

    traverse(root_node)
    
    return FileExtractionResult(
        file_path=file_path,
        language="go",
        functions=functions,
        classes=classes,
        imports=imports,
        call_sites=call_sites,
        constants=constants,
        file_hash=_hash_text(source),
        total_lines=source.count("\n") + 1,
        exports=[f.name for f in functions if f.is_exported] + [c.name for c in classes if c.name and c.name[0].isupper()]
    )
