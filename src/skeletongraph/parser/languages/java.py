"""
Java tree-sitter AST extraction rules.
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

def extract_java(file_path: str, source: str, source_bytes: bytes, tree: Tree) -> Optional[FileExtractionResult]:
    root_node = tree.root_node
    
    functions: List[RawFunction] = []
    classes: List[RawClass] = []
    imports: List[RawImport] = []
    call_sites: List[RawCallSite] = []
    
    package_name = ""
    # Find package declaration
    pkg_node = _get_child_by_type(root_node, "package_declaration")
    if pkg_node:
        ident = _get_child_by_type(pkg_node, "scoped_identifier") or _get_child_by_type(pkg_node, "identifier")
        if ident:
            package_name = node_text(ident, source_bytes)
            
    base_fqn = package_name if package_name else Path(file_path).stem

    def traverse(node: Node, current_class: Optional[RawClass] = None, current_fqn: str = base_fqn):
        if node.type == "import_declaration":
            path_node = _get_child_by_type(node, "scoped_identifier") or _get_child_by_type(node, "identifier")
            if path_node:
                path = node_text(path_node, source_bytes)
                alias = path.split('.')[-1]
                imports.append(RawImport(
                    module=path,
                    names=[],
                    aliases={alias: path},
                    line=node.start_point[0] + 1
                ))

        elif node.type in ("class_declaration", "interface_declaration", "record_declaration", "enum_declaration"):
            cls_name = _get_name(node, source_bytes)
            if cls_name:
                kind_map = {
                    "class_declaration": NodeKind.CLASS,
                    "interface_declaration": NodeKind.INTERFACE,
                    "record_declaration": NodeKind.CLASS,
                    "enum_declaration": NodeKind.ENUM
                }
                
                cls_fqn = f"{current_fqn}::{cls_name}" if current_fqn == base_fqn else f"{current_fqn}${cls_name}"
                
                new_class = RawClass(
                    name=cls_name,
                    fqn=cls_fqn,
                    file_path=file_path,
                    line_start=node.start_point[0] + 1,
                    line_end=node.end_point[0] + 1,
                    kind=kind_map.get(node.type, NodeKind.CLASS)
                )
                classes.append(new_class)
                
                # Recursively parse body, passing down the new scope
                body = _get_child_by_type(node, "class_body") or _get_child_by_type(node, "interface_body") or _get_child_by_type(node, "enum_body")
                if body:
                    for child in body.children:
                        traverse(child, current_class=new_class, current_fqn=cls_fqn)
            return # Block default recursion to avoid double processing
                        
        elif node.type in ("method_declaration", "constructor_declaration"):
            func_name = _get_name(node, source_bytes)
            if node.type == "constructor_declaration" and current_class:
                func_name = current_class.name
                
            if func_name:
                sig_text = ""
                # A crude signature extraction (Modifiers + Type + Name + Params)
                # To keep it simple, we just extract from start to the block.
                block_node = _get_child_by_type(node, "block")
                if block_node:
                    sig_bytes = source_bytes[node.start_byte:block_node.start_byte]
                    sig_text = sig_bytes.decode("utf-8", errors="replace").strip()
                else:
                    sig_text = node_text(node, source_bytes)
                
                body_text = node_text(block_node, source_bytes) if block_node else ""
                
                # Check modifiers for export status
                is_exported = False
                mods = _get_child_by_type(node, "modifiers")
                if mods and "public" in node_text(mods, source_bytes):
                    is_exported = True
                
                # For interfaces, everything is exported
                if current_class and current_class.kind == NodeKind.INTERFACE:
                    is_exported = True
                    
                kind = NodeKind.CONSTRUCTOR if node.type == "constructor_declaration" else NodeKind.METHOD
                
                fqn = f"{current_fqn}.{func_name}" if current_class else f"{current_fqn}::{func_name}"
                
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

        # Call site extraction logic
        elif node.type == "method_invocation":
            fn = _get_child_by_type(node, "identifier")
            if fn:
                callee_name = node_text(fn, source_bytes)
                call_sites.append(RawCallSite(
                    caller_fqn="", 
                    callee_name=callee_name,
                    line=node.start_point[0] + 1
                ))
        elif node.type == "object_creation_expression":
            type_id = _get_child_by_type(node, "type_identifier")
            if type_id:
                call_sites.append(RawCallSite(
                    caller_fqn="", 
                    callee_name=node_text(type_id, source_bytes),
                    line=node.start_point[0] + 1,
                    is_constructor=True
                ))

        for child in node.children:
            traverse(child, current_class, current_fqn)

    traverse(root_node)
    
    return FileExtractionResult(
        file_path=file_path,
        language="java",
        functions=functions,
        classes=classes,
        imports=imports,
        call_sites=call_sites,
        file_hash=_hash_text(source),
        total_lines=source.count("\\n") + 1,
        exports=[c.name for c in classes] # Java exports public classes
    )
