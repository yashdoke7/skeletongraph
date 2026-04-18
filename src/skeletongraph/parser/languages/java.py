"""
Java tree-sitter AST extraction rules.
"""
from __future__ import annotations
from typing import Optional
from tree_sitter import Node, Tree
from ..ast_extractor import FileExtractionResult

def extract_java(file_path: str, source: str, source_bytes: bytes, tree: Tree) -> Optional[FileExtractionResult]:
    # Placeholder for Java AST extraction (class_declaration, method_declaration)
    # See languages/python.py for full implementation pattern
    return FileExtractionResult(file_path=file_path, language="java", file_hash="", total_lines=source.count("\\n") + 1)
