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
    name_node = _get_child_by_type(node, "identifier")
    if name_node:
        return node_text(name_node, source_bytes)
    return ""

def extract_go(file_path: str, source: str, source_bytes: bytes, tree: Tree) -> Optional[FileExtractionResult]:
    root_node = tree.root_node
    
    functions: List[RawFunction] = []
    classes: List[RawClass] = []
    imports: List[RawImport] = []
    call_sites: List[RawCallSite] = []
    
    package_name = ""
    # Find package clause
    pkg_node = _get_child_by_type(root_node, "package_clause")
    if pkg_node:
        package_name = _get_name(pkg_node, source_bytes)
    
    # Prefix for all FQNs in this file will be based on the package
    base_fqn = package_name if package_name else Path(file_path).stem

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
                        
                        classes.append(RawClass(
                            name=struct_name,
                            fqn=f"{base_fqn}::{struct_name}",
                            file_path=file_path,
                            line_start=spec.start_point[0] + 1,
                            line_end=spec.end_point[0] + 1,
                            kind=kind
                        ))
                        
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
                    is_exported=is_exported
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
                    parent_class=parent_class
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
        file_hash=_hash_text(source),
        total_lines=source.count("\\n") + 1,
        exports=[f.name for f in functions if f.is_exported] + [c.name for c in classes if c.name and c.name[0].isupper()]
    )
