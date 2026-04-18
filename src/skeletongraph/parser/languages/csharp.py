"""
C# tree-sitter AST extraction rules.
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
    name_node = _get_child_by_type(node, "identifier")
    if name_node:
        return node_text(name_node, source_bytes)
    return ""

def extract_csharp(file_path: str, source: str, source_bytes: bytes, tree: Tree) -> Optional[FileExtractionResult]:
    root_node = tree.root_node
    
    functions: List[RawFunction] = []
    classes: List[RawClass] = []
    imports: List[RawImport] = []
    call_sites: List[RawCallSite] = []
    
    base_fqn = Path(file_path).stem

    def traverse(node: Node, current_class: Optional[RawClass] = None, current_ns: str = ""):
        if node.type == "using_directive":
            path_node = _get_child_by_type(node, "identifier") or _get_child_by_type(node, "qualified_name")
            if path_node:
                path = node_text(path_node, source_bytes)
                imports.append(RawImport(
                    module=path,
                    names=[],
                    line=node.start_point[0] + 1
                ))

        elif node.type == "namespace_declaration":
            ns_name_node = _get_child_by_type(node, "qualified_name") or _get_child_by_type(node, "identifier")
            if ns_name_node:
                current_ns = node_text(ns_name_node, source_bytes)
            
            body = _get_child_by_type(node, "declaration_list")
            if body:
                for child in body.children:
                    traverse(child, current_class, current_ns)
            return

        elif node.type in ("class_declaration", "interface_declaration", "struct_declaration", "record_declaration"):
            cls_name = _get_name(node, source_bytes)
            if cls_name:
                kind_map = {
                    "class_declaration": NodeKind.CLASS,
                    "interface_declaration": NodeKind.INTERFACE,
                    "struct_declaration": NodeKind.STRUCT,
                    "record_declaration": NodeKind.CLASS
                }
                
                ns_prefix = f"{current_ns}::" if current_ns else ""
                cls_fqn = f"{ns_prefix}{cls_name}"
                
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
                        
        elif node.type in ("method_declaration", "constructor_declaration"):
            func_name = _get_name(node, source_bytes)
            if node.type == "constructor_declaration" and current_class:
                func_name = current_class.name
                
            if func_name:
                block_node = _get_child_by_type(node, "block")
                arrow_node = _get_child_by_type(node, "arrow_expression_clause")
                
                body_target = block_node or arrow_node
                
                sig_text = ""
                body_text = ""
                if body_target:
                    sig_bytes = source_bytes[node.start_byte:body_target.start_byte]
                    sig_text = sig_bytes.decode("utf-8", errors="replace").strip()
                    body_text = node_text(body_target, source_bytes)
                else:
                    sig_text = node_text(node, source_bytes)
                
                is_exported = False
                mods = _get_child_by_type(node, "modifier")
                if mods and "public" in node_text(mods, source_bytes):
                    is_exported = True
                
                kind = NodeKind.CONSTRUCTOR if node.type == "constructor_declaration" else NodeKind.METHOD
                
                ns_prefix = f"{current_ns}::" if current_ns else ""
                fqn = f"{ns_prefix}{current_class.name}.{func_name}" if current_class else f"{ns_prefix}{func_name}"
                
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
                    parent_class=current_class.name if current_class else None
                )
                
                if current_class:
                    current_class.methods.append(func)
                else:
                    functions.append(func)

        elif node.type == "invocation_expression":
            fn = _get_child_by_type(node, "identifier") or _get_child_by_type(node, "member_access_expression")
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
        language="csharp",
        functions=functions,
        classes=classes,
        imports=imports,
        call_sites=call_sites,
        file_hash=_hash_text(source),
        total_lines=source.count("\\n") + 1,
        exports=[c.name for c in classes]
    )
