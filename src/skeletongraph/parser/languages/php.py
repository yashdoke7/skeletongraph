"""
PHP tree-sitter AST extraction rules.
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
    name_node = _get_child_by_type(node, "name") or _get_child_by_type(node, "identifier")
    if name_node:
        return node_text(name_node, source_bytes)
    return ""

def _extract_phpdoc(node: Node, source_bytes: bytes) -> str:
    """Extract PHPDoc /** */ comment preceding a node."""
    prev = node.prev_named_sibling
    if prev and prev.type == "comment":
        text = node_text(prev, source_bytes).strip()
        if text.startswith("/**"):
            text = text.lstrip("/").lstrip("*").rstrip("*").rstrip("/").strip()
            for line in text.split("\n"):
                line = line.strip().lstrip("*").strip()
                if line and not line.startswith("@"):
                    return line[:200]
    return ""

def extract_php(file_path: str, source: str, source_bytes: bytes, tree: Tree) -> Optional[FileExtractionResult]:
    root_node = tree.root_node
    
    functions: List[RawFunction] = []
    classes: List[RawClass] = []
    imports: List[RawImport] = []
    call_sites: List[RawCallSite] = []
    constants: List[Tuple[str, str]] = []

    base_fqn = Path(file_path).stem

    def traverse(node: Node, current_class: Optional[RawClass] = None, current_ns: str = ""):
        if node.type == "namespace_use_declaration":
            path_node = _get_child_by_type(node, "namespace_use_clause")
            if path_node:
                path = node_text(path_node, source_bytes)
                imports.append(RawImport(
                    module=path,
                    names=[],
                    line=node.start_point[0] + 1
                ))

        elif node.type == "namespace_definition":
            ns_name_node = _get_child_by_type(node, "namespace_name")
            if ns_name_node:
                current_ns = node_text(ns_name_node, source_bytes)
            
            for child in node.children:
                traverse(child, current_class, current_ns)
            return

        elif node.type in ("class_declaration", "interface_declaration", "trait_declaration"):
            cls_name = _get_name(node, source_bytes)
            if cls_name:
                kind_map = {
                    "class_declaration": NodeKind.CLASS,
                    "interface_declaration": NodeKind.INTERFACE,
                    "trait_declaration": NodeKind.TRAIT
                }
                
                # file_path:: prefix (see go.py rationale); PHP namespace kept in
                # the symbol part, not as the split-prefix.
                ns_part = f"{current_ns}\\" if current_ns else ""
                cls_fqn = f"{file_path}::{ns_part}{cls_name}"

                new_class = RawClass(
                    name=cls_name,
                    fqn=cls_fqn,
                    file_path=file_path,
                    line_start=node.start_point[0] + 1,
                    line_end=node.end_point[0] + 1,
                    kind=kind_map.get(node.type, NodeKind.CLASS)
                )
                classes.append(new_class)
                
                body = _get_child_by_type(node, "declaration_list")
                if body:
                    for child in body.children:
                        traverse(child, current_class=new_class, current_ns=current_ns)
            return
                        
        elif node.type == "method_declaration" or node.type == "function_definition":
            func_name = _get_name(node, source_bytes)
            if func_name:
                body_node = _get_child_by_type(node, "compound_statement")
                
                sig_text = ""
                body_text = ""
                if body_node:
                    sig_bytes = source_bytes[node.start_byte:body_node.start_byte]
                    sig_text = sig_bytes.decode("utf-8", errors="replace").strip()
                    body_text = node_text(body_node, source_bytes)
                else:
                    sig_text = node_text(node, source_bytes)
                
                is_exported = False
                mods = _get_child_by_type(node, "visibility_modifier")
                if mods and "public" in node_text(mods, source_bytes):
                    is_exported = True
                if not mods:  # Implicitly public in PHP
                    is_exported = True
                
                kind = NodeKind.METHOD if node.type == "method_declaration" else NodeKind.FUNCTION
                if func_name == "__construct":
                    kind = NodeKind.CONSTRUCTOR
                
                ns_part = f"{current_ns}\\" if current_ns else ""
                fqn = (f"{file_path}::{ns_part}{current_class.name}.{func_name}"
                       if current_class else f"{file_path}::{ns_part}{func_name}")
                
                func = RawFunction(
                    name=func_name,
                    fqn=fqn,
                    file_path=file_path,
                    line_start=node.start_point[0] + 1,
                    line_end=node.end_point[0] + 1,
                    signature=sig_text,
                    kind=kind,
                    body_text=body_text,
                    is_exported=is_exported,
                    parent_class=current_class.name if current_class else None,
                    docstring=_extract_phpdoc(node, source_bytes),
                )
                
                if current_class:
                    current_class.methods.append(func)
                else:
                    functions.append(func)

        # `const NAME = ...;` at file or class scope (PHP's constants).
        elif node.type == "const_declaration":
            for el in node.children:
                if el.type == "const_element":
                    nm = _get_child_by_type(el, "name")
                    if nm:
                        constants.append((node_text(nm, source_bytes),
                                          node_text(el, source_bytes).strip()[:80]))

        elif node.type == "member_call_expression" or node.type == "function_call_expression":
            fn = _get_child_by_type(node, "name") or _get_child_by_type(node, "identifier")
            if fn:
                callee_name = node_text(fn, source_bytes)
                call_sites.append(RawCallSite(
                    caller_fqn="", 
                    callee_name=callee_name,
                    line=node.start_point[0] + 1
                ))

        for child in node.children:
            traverse(child, current_class, current_ns)

    traverse(root_node)
    
    return FileExtractionResult(
        file_path=file_path,
        language="php",
        functions=functions,
        classes=classes,
        imports=imports,
        call_sites=call_sites,
        constants=constants,
        file_hash=_hash_text(source),
        total_lines=source.count("\n") + 1,
        exports=[]
    )
