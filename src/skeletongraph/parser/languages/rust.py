"""
Rust tree-sitter AST extraction rules.
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

def _extract_rust_doc(node: Node, source_bytes: bytes) -> str:
    """Extract Rust /// doc comment preceding a node."""
    prev = node.prev_named_sibling
    if prev and prev.type == "line_comment":
        text = node_text(prev, source_bytes).strip()
        if text.startswith("///"):
            return text.lstrip("/").strip()[:200]
    return ""

def extract_rust(file_path: str, source: str, source_bytes: bytes, tree: Tree) -> Optional[FileExtractionResult]:
    root_node = tree.root_node
    
    functions: List[RawFunction] = []
    classes: List[RawClass] = []
    imports: List[RawImport] = []
    call_sites: List[RawCallSite] = []
    constants: List[Tuple[str, str]] = []

    # FQN prefix MUST be the file_path (see go.py rationale) — the old
    # Path(file_path).stem dropped the directory and extension, so the split
    # prefix never matched gold file paths → 0 recall on every Rust task.
    base_fqn = file_path

    def traverse(node: Node, current_impl: Optional[str] = None):
        if node.type == "use_declaration":
            # Very simplistic use parsing for now
            path_node = _get_child_by_type(node, "scoped_identifier") or _get_child_by_type(node, "identifier")
            if path_node:
                path = node_text(path_node, source_bytes)
                imports.append(RawImport(
                    module=path,
                    names=[],
                    line=node.start_point[0] + 1
                ))

        elif node.type in ("const_item", "static_item"):
            nm = _get_child_by_type(node, "identifier")
            if nm:
                cname = node_text(nm, source_bytes)
                if cname:
                    constants.append(
                        (cname, node_text(node, source_bytes).strip()[:80]))

        elif node.type in ("struct_item", "trait_item", "enum_item"):
            cls_name = _get_name(node, source_bytes)
            if cls_name:
                kind_map = {
                    "struct_item": NodeKind.STRUCT,
                    "trait_item": NodeKind.TRAIT,
                    "enum_item": NodeKind.ENUM
                }

                classes.append(RawClass(
                    name=cls_name,
                    fqn=f"{base_fqn}::{cls_name}",
                    file_path=file_path,
                    line_start=node.start_point[0] + 1,
                    line_end=node.end_point[0] + 1,
                    kind=kind_map.get(node.type, NodeKind.STRUCT)
                ))

                # enum variants → searchable beacons for the file
                if node.type == "enum_item":
                    body = _get_child_by_type(node, "enum_variant_list")
                    if body:
                        for v in body.children:
                            if v.type == "enum_variant":
                                vn = _get_child_by_type(v, "identifier")
                                if vn:
                                    constants.append(
                                        (node_text(vn, source_bytes),
                                         f"{cls_name} variant"))

                # trait method signatures are the API contract — recurse with the
                # trait as current_impl so required/default methods bind to it.
                if node.type == "trait_item":
                    for child in node.children:
                        traverse(child, current_impl=cls_name)
                    return

        elif node.type == "impl_item":
            # Extract the actual type from impl TypeName { ... }
            type_id = _get_child_by_type(node, "type_identifier") or _get_child_by_type(node, "generic_type")
             # generic_type has a type_identifier inside it
            if type_id and type_id.type == "generic_type":
                type_id = _get_child_by_type(type_id, "type_identifier")
                
            impl_name = node_text(type_id, source_bytes) if type_id else None
            
            # Recursive pass into the impl block to catch function_item mapping
            if impl_name:
                for child in node.children:
                    traverse(child, current_impl=impl_name)
            return
            
        elif node.type == "function_item":
            func_name = _get_name(node, source_bytes)
            if func_name:
                # Is it public?
                is_exported = False
                vis_node = _get_child_by_type(node, "visibility_modifier")
                if vis_node:
                    is_exported = True
                    
                # Is it a method? Check parameter list for 'self'
                kind = NodeKind.FUNCTION
                param_node = _get_child_by_type(node, "parameters")
                if param_node:
                    first_p = node_text(param_node, source_bytes)
                    if "self" in first_p:
                        kind = NodeKind.METHOD
                        
                # Extract signature and body
                block_node = _get_child_by_type(node, "block")
                sig_text = ""
                body_text = ""
                if block_node:
                    sig_bytes = source_bytes[node.start_byte:block_node.start_byte]
                    sig_text = sig_bytes.decode("utf-8", errors="replace").strip()
                    body_text = node_text(block_node, source_bytes)
                else:
                    sig_text = node_text(node, source_bytes)
                    
                # Bind to struct if we are inside an impl block
                fqn = f"{base_fqn}::{current_impl}.{func_name}" if current_impl else f"{base_fqn}::{func_name}"
                
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
                    parent_class=current_impl,
                    docstring=_extract_rust_doc(node, source_bytes),
                )
                
                if current_impl:
                    # Bind it to the existing class declaration if found
                    for cls in classes:
                        if cls.name == current_impl:
                            cls.methods.append(func)
                            break
                    # Regardless, we keep flattened functions list
                functions.append(func)

        # Trait required-method declarations (no body) — bind to current trait.
        elif node.type == "function_signature_item" and current_impl:
            func_name = _get_name(node, source_bytes)
            if func_name:
                func = RawFunction(
                    name=func_name,
                    fqn=f"{base_fqn}::{current_impl}.{func_name}",
                    file_path=file_path,
                    line_start=node.start_point[0] + 1,
                    line_end=node.end_point[0] + 1,
                    signature=node_text(node, source_bytes).strip()[:160],
                    kind=NodeKind.METHOD, body_text="",
                    is_exported=True, parent_class=current_impl,
                )
                for cls in classes:
                    if cls.name == current_impl:
                        cls.methods.append(func)
                        break
                functions.append(func)

        # Call extraction
        elif node.type == "call_expression":
            fn = _get_child_by_type(node, "identifier") or _get_child_by_type(node, "field_expression")
            if fn:
                callee_name = node_text(fn, source_bytes)
                call_sites.append(RawCallSite(
                    caller_fqn="", 
                    callee_name=callee_name,
                    line=node.start_point[0] + 1
                ))

        for child in node.children:
            traverse(child, current_impl)

    traverse(root_node)
    
    return FileExtractionResult(
        file_path=file_path,
        language="rust",
        functions=functions,
        classes=classes,
        imports=imports,
        call_sites=call_sites,
        constants=constants,
        file_hash=_hash_text(source),
        total_lines=source.count("\n") + 1,
        exports=[f.name for f in functions if f.is_exported] + [c.name for c in classes] # simplified export logic
    )
