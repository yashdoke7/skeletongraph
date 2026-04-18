"""
Ruby tree-sitter AST extraction rules.
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
    name_node = _get_child_by_type(node, "identifier") or _get_child_by_type(node, "constant")
    if name_node:
        return node_text(name_node, source_bytes)
    return ""

def extract_ruby(file_path: str, source: str, source_bytes: bytes, tree: Tree) -> Optional[FileExtractionResult]:
    root_node = tree.root_node
    
    functions: List[RawFunction] = []
    classes: List[RawClass] = []
    imports: List[RawImport] = []
    call_sites: List[RawCallSite] = []
    
    base_fqn = Path(file_path).stem

    def traverse(node: Node, current_class: Optional[RawClass] = None, current_module: str = ""):
        if node.type == "call":
            method_node = _get_child_by_type(node, "identifier")
            if method_node:
                name = node_text(method_node, source_bytes)
                if name in ("require", "require_relative", "include", "extend"):
                    arg_list = _get_child_by_type(node, "argument_list")
                    if arg_list:
                        str_node = _get_child_by_type(arg_list, "string")
                        if str_node:
                            path = node_text(str_node, source_bytes).strip("\"'")
                            imports.append(RawImport(
                                module=path,
                                names=[],
                                line=node.start_point[0] + 1
                            ))
                else:
                    # General Method call
                    call_sites.append(RawCallSite(
                        caller_fqn="", 
                        callee_name=name,
                        line=node.start_point[0] + 1
                    ))

        elif node.type == "module":
            mod_name = _get_name(node, source_bytes)
            mod_fqn = f"{current_module}::{mod_name}" if current_module else mod_name
            for child in node.children:
                traverse(child, current_class, current_module=mod_fqn)
            return
            
        elif node.type == "class":
            cls_name = _get_name(node, source_bytes)
            if cls_name:
                ns_prefix = f"{current_module}::" if current_module else ""
                cls_fqn = f"{ns_prefix}{cls_name}"
                
                new_class = RawClass(
                    name=cls_name,
                    fqn=cls_fqn,
                    file_path=file_path,
                    line_start=node.start_point[0] + 1,
                    line_end=node.end_point[0] + 1,
                    kind=NodeKind.CLASS
                )
                classes.append(new_class)
                
                for child in node.children:
                    traverse(child, current_class=new_class, current_module=current_module)
            return
                        
        elif node.type == "method":
            func_name = _get_name(node, source_bytes)
            if func_name:
                sig_text = f"def {func_name}"
                params = _get_child_by_type(node, "method_parameters")
                if params:
                    sig_text += node_text(params, source_bytes)
                
                body_text = ""
                # Ruby bodies are just a sequence of statements inside the method
                # We can't strictly grab a block, so we'll store the whole method text as body_text minus def line
                body_text = node_text(node, source_bytes)
                
                ns_prefix = f"{current_module}::" if current_module else ""
                fqn = f"{ns_prefix}{current_class.name}.{func_name}" if current_class else f"{ns_prefix}{func_name}"
                
                func = RawFunction(
                    name=func_name,
                    fqn=fqn,
                    file_path=file_path,
                    line_start=node.start_point[0] + 1,
                    line_end=node.end_point[0] + 1,
                    signature=sig_text,
                    kind=NodeKind.METHOD if current_class else NodeKind.FUNCTION,
                    body_text=body_text,
                    is_exported=True, # public by default in Ruby
                    parent_class=current_class.name if current_class else None
                )
                
                if current_class:
                    current_class.methods.append(func)
                else:
                    functions.append(func)

        for child in node.children:
            traverse(child, current_class, current_module)

    traverse(root_node)
    
    return FileExtractionResult(
        file_path=file_path,
        language="ruby",
        functions=functions,
        classes=classes,
        imports=imports,
        call_sites=call_sites,
        file_hash=_hash_text(source),
        total_lines=source.count("\\n") + 1,
        exports=[]
    )
