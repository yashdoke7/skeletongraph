"""
C++ tree-sitter AST extraction rules.
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
    name_node = _get_child_by_type(node, "identifier") or _get_child_by_type(node, "type_identifier")
    if name_node:
        return node_text(name_node, source_bytes)
    return ""

def _extract_cpp_doc(node: Node, source_bytes: bytes) -> str:
    """Extract C++ doc comment (/// or /** */) preceding a node."""
    prev = node.prev_named_sibling
    if prev and prev.type == "comment":
        text = node_text(prev, source_bytes).strip()
        if text.startswith("///"):
            return text.lstrip("/").strip()[:200]
        if text.startswith("/**"):
            text = text.lstrip("/").lstrip("*").rstrip("*").rstrip("/").strip()
            for line in text.split("\n"):
                line = line.strip().lstrip("*").strip()
                if line and not line.startswith("@"):
                    return line[:200]
    return ""

def extract_cpp(file_path: str, source: str, source_bytes: bytes, tree: Tree) -> Optional[FileExtractionResult]:
    root_node = tree.root_node
    
    functions: List[RawFunction] = []
    classes: List[RawClass] = []
    imports: List[RawImport] = []
    call_sites: List[RawCallSite] = []
    constants: List[Tuple[str, str]] = []

    base_namespace = []

    def current_namespace() -> str:
        if not base_namespace:
            return ""
        return "::".join(base_namespace)

    def traverse(node: Node, current_class: Optional[str] = None):
        if node.type == "preproc_include":
            path_node = _get_child_by_type(node, "string_literal") or _get_child_by_type(node, "system_lib_string")
            if path_node:
                path = node_text(path_node, source_bytes)
                imports.append(RawImport(
                    module=path,
                    names=[],
                    line=node.start_point[0] + 1
                ))

        elif node.type == "namespace_definition":
            ns_name = _get_name(node, source_bytes)
            if ns_name:
                base_namespace.append(ns_name)
            for child in node.children:
                traverse(child, current_class)
            if ns_name:
                base_namespace.pop()
            return
            
        elif node.type in ("class_specifier", "struct_specifier"):
            cls_name = _get_child_by_type(node, "type_identifier")
            cls_name_str = node_text(cls_name, source_bytes) if cls_name else ""
            if cls_name_str:
                kind = NodeKind.CLASS if node.type == "class_specifier" else NodeKind.STRUCT
                # file_path:: prefix (see go.py rationale); C++ namespace kept in
                # the symbol part, not as the split-prefix.
                ns = current_namespace()
                ns_part = f"{ns}." if ns else ""
                fqn = f"{file_path}::{ns_part}{cls_name_str}"

                classes.append(RawClass(
                    name=cls_name_str,
                    fqn=fqn,
                    file_path=file_path,
                    line_start=node.start_point[0] + 1,
                    line_end=node.end_point[0] + 1,
                    kind=kind
                ))
                for child in node.children:
                    traverse(child, current_class=cls_name_str)
            return

        elif node.type == "function_definition":
            # Extract function name, checking for Class::method syntax
            declarator = _get_child_by_type(node, "function_declarator")
            if not declarator:
                return
                
            name_node = _get_child_by_type(declarator, "identifier")
            qual_name = _get_child_by_type(declarator, "qualified_identifier")
            
            func_name = ""
            parent_cls = current_class
            
            if qual_name:
                # out-of-line definition: ClassName::methodName
                qual_str = node_text(qual_name, source_bytes)
                parts = qual_str.split("::")
                if len(parts) >= 2:
                    parent_cls = parts[-2]
                    func_name = parts[-1]
                else:
                    func_name = qual_str
            elif name_node:
                func_name = node_text(name_node, source_bytes)
                
            if func_name:
                ns = current_namespace()
                ns_part = f"{ns}." if ns else ""
                fqn = (f"{file_path}::{ns_part}{parent_cls}.{func_name}"
                       if parent_cls else f"{file_path}::{ns_part}{func_name}")
                
                block_node = _get_child_by_type(node, "compound_statement")
                sig_text = ""
                body_text = ""
                if block_node:
                    sig_bytes = source_bytes[node.start_byte:block_node.start_byte]
                    sig_text = sig_bytes.decode("utf-8", errors="replace").strip()
                    body_text = node_text(block_node, source_bytes)
                else:
                    sig_text = node_text(node, source_bytes)
                    
                kind = NodeKind.METHOD if parent_cls else NodeKind.FUNCTION
                
                func = RawFunction(
                    name=func_name,
                    fqn=fqn,
                    file_path=file_path,
                    line_start=node.start_point[0] + 1,
                    line_end=node.end_point[0] + 1,
                    signature=sig_text,
                    kind=kind,
                    body_text=body_text,
                    is_exported=True,
                    parent_class=parent_cls,
                    docstring=_extract_cpp_doc(node, source_bytes),
                )
                
                # Try to bind to local class if we are parsing the header
                if parent_cls:
                    for cls in classes:
                        if cls.name == parent_cls:
                            cls.methods.append(func)
                            break
                functions.append(func)

        # In-class method DECLARATIONS (no body) — `void baz();` in a header.
        # These are field_declaration nodes, not function_definition, so the
        # block above misses them — leaving most header-declared methods
        # unindexed. Also pick up const members here.
        elif node.type == "field_declaration" and current_class:
            fd = _get_child_by_type(node, "function_declarator")
            if fd:
                nm = (_get_child_by_type(fd, "field_identifier")
                      or _get_child_by_type(fd, "identifier"))
                if nm:
                    mname = node_text(nm, source_bytes)
                    ns = current_namespace()
                    ns_part = f"{ns}." if ns else ""
                    func = RawFunction(
                        name=mname,
                        fqn=f"{file_path}::{ns_part}{current_class}.{mname}",
                        file_path=file_path,
                        line_start=node.start_point[0] + 1,
                        line_end=node.end_point[0] + 1,
                        signature=node_text(node, source_bytes).strip()[:160],
                        kind=NodeKind.METHOD, body_text="",
                        is_exported=True, parent_class=current_class,
                    )
                    for cls in classes:
                        if cls.name == current_class:
                            cls.methods.append(func)
                            break
                    functions.append(func)
            elif _get_child_by_type(node, "type_qualifier"):  # const member
                fi = _get_child_by_type(node, "field_identifier")
                if fi:
                    constants.append((node_text(fi, source_bytes),
                                      node_text(node, source_bytes).strip()[:80]))

        # #define macros and file-level `const` declarations → constants.
        elif node.type == "preproc_def":
            nm = _get_child_by_type(node, "identifier")
            if nm:
                constants.append((node_text(nm, source_bytes),
                                  node_text(node, source_bytes).strip()[:80]))

        elif (node.type == "declaration" and node.parent is not None
              and node.parent.type == "translation_unit"
              and _get_child_by_type(node, "type_qualifier")):
            idecl = _get_child_by_type(node, "init_declarator")
            nm = _get_child_by_type(idecl, "identifier") if idecl else None
            if nm:
                constants.append((node_text(nm, source_bytes),
                                  node_text(node, source_bytes).strip()[:80]))

        elif node.type == "call_expression":
            fn = _get_child_by_type(node, "identifier") or _get_child_by_type(node, "field_expression") or _get_child_by_type(node, "qualified_identifier")
            if fn:
                callee_name = node_text(fn, source_bytes)
                call_sites.append(RawCallSite(
                    caller_fqn="", 
                    callee_name=callee_name,
                    line=node.start_point[0] + 1
                ))

        for child in node.children:
            traverse(child, current_class)

    traverse(root_node)
    
    return FileExtractionResult(
        file_path=file_path,
        language="cpp",
        functions=functions,
        classes=classes,
        imports=imports,
        call_sites=call_sites,
        constants=constants,
        file_hash=_hash_text(source),
        total_lines=source.count("\n") + 1,
        exports=[]
    )
