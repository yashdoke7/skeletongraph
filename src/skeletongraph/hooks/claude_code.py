"""
Claude Code hook integrations.

Claude Code allows configuring custom scripts for events like session start,
before prompt submission, and after tool use. These functions implement
the SkeletonGraph v4 pipeline for those hooks.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

from ..storage.local import load_index
from ..retrieval.resolver import resolve_context
from ..retrieval.classifier import classify_query, ContextMode
from ..assembly.prompt_builder import assemble
from ..session.memory import SessionMemory
from ..engine import SGEngine
from ..config import load_config

logger = logging.getLogger(__name__)


def hook_session_start(project_root: Path) -> str:
    """Run at session start.
    
    1. Triggers an sg build (which handles init if needed).
    2. Returns the L0 project.md constraints to inject into the system prompt.
    """
    sg_dir = project_root / ".skeletongraph"
    project_md = sg_dir / "project.md"
    
    # Run build if index doesn't exist
    if not sg_dir.exists() or not (sg_dir / "index.json").exists():
        from ..build import build_index
        try:
            build_index(project_root)
        except Exception as e:
            logger.error(f"Failed to build index on session start: {e}")
            return "SkeletonGraph: Failed to build index."
            
    if project_md.exists():
        content = project_md.read_text(encoding="utf-8", errors="replace")
        return f"=== Project Constraints (SkeletonGraph L0) ===\n{content}"
        
    return "SkeletonGraph: Project initialized."


def hook_pre_prompt(project_root: Path, prompt: str) -> str:
    """Run before user prompt is submitted to Claude.
    
    v4: Uses SGEngine for unified pipeline (classify → resolve → assemble).
    Writes context.md + shadows. Returns status string with model routing.
    """
    sg_dir = project_root / ".skeletongraph"
    
    if not sg_dir.exists():
        return "SkeletonGraph: Index not found. Run `sg build`."
        
    try:
        t0 = time.perf_counter()
        
        # v4: Use SGEngine
        engine = SGEngine(project_root=project_root)
        result = engine.query(prompt)
        
        if not result.success:
            return f"SkeletonGraph error: {result.error}"
        
        # Write context.md
        context_path = sg_dir / "context.md"
        context_path.write_text(result.context_text, encoding="utf-8")
        
        duration_ms = int((time.perf_counter() - t0) * 1000)
        
        # Model routing recommendation
        routing_hint = ""
        if result.model_tier.value == "llm":
            routing_hint = f"\nRecommended: /model {result.recommended_model} (complex task)"
        elif result.model_tier.value == "slm":
            routing_hint = "\nThis is a simple lookup — current model is sufficient."
        
        cost_info = ""
        if result.slm_used:
            cost_info = f" SLM: ${result.slm_cost_usd:.4f}"
        
        return (
            f"SkeletonGraph prepared context ({result.query_mode.value} mode, "
            f"{result.context_tokens} tokens, {result.confidence}) in {duration_ms}ms."
            f"{cost_info}\n"
            f"Please read `.skeletongraph/context.md` for target code and constraints "
            f"before taking further action."
            f"{routing_hint}"
        )
        
    except Exception as e:
        logger.error(f"Error in pre-prompt hook: {e}")
        return f"SkeletonGraph error: {e}"


def hook_post_tool_use(project_root: Path, tool_name: str, tool_args: str, tool_output: str) -> str:
    """Run after Claude uses a tool (e.g., file modification).
    
    1. Extracts modified files and deletes their stale shadow files.
    2. Runs session post-processing to extract decisions.
    """
    sg_dir = project_root / ".skeletongraph"
    shadow_dir = sg_dir / "shadows"
    
    modified_files = []
    
    # Basic heuristic to detect modified files from tool args/output
    # This would need to be customized based on exact Claude Code tool schemas
    if "write" in tool_name.lower() or "edit" in tool_name.lower() or "replace" in tool_name.lower():
        import re
        # Try to find file paths in arguments
        paths = re.findall(r'[\w./\\-]+\.(?:py|js|ts|tsx|jsx|java|go|rs|cpp|cs|rb|php)', tool_args)
        modified_files.extend(paths)
        
    if not modified_files:
        return ""
        
    # Delete stale shadows
    deleted_shadows = []
    if shadow_dir.exists():
        for file_path in modified_files:
            shadow_path = shadow_dir / file_path
            if shadow_path.exists():
                try:
                    shadow_path.unlink()
                    deleted_shadows.append(file_path)
                except Exception:
                    pass
                    
    # Update session memory
    try:
        mem = SessionMemory.load(sg_dir)
        # We use a placeholder prompt here since we only have the tool context
        mem.post_process(
            prompt="[Tool Use Turn]", 
            agent_response=tool_output,
            files_modified=modified_files
        )
    except Exception as e:
        logger.error(f"Failed to update session memory in post-tool hook: {e}")
        
    if deleted_shadows:
        return f"SkeletonGraph: Deleted stale shadow files for {len(deleted_shadows)} modified files."
    return ""


def hook_session_end(project_root: Path) -> str:
    """Run at session end.
    
    Compresses session memory (current → recent → project_log).
    """
    sg_dir = project_root / ".skeletongraph"
    try:
        mem = SessionMemory.load(sg_dir)
        mem.compress()
        return "SkeletonGraph: Session memory compressed."
    except Exception as e:
        return f"SkeletonGraph error compressing session: {e}"


def _write_shadow_files(assembled, store, project_root: Path, shadow_dir: Path):
    """Write pre-assembled file views as shadow files."""
    files_seen = set()
    for fqn in assembled.targets:
        sk = store.skeleton_table.get(fqn)
        if not sk:
            continue
        fp = sk.file_path
        if fp in files_seen:
            continue
        files_seen.add(fp)

        source = project_root / fp
        if not source.exists():
            continue

        shadow_path = shadow_dir / fp
        shadow_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            content = source.read_text(encoding="utf-8", errors="replace")
            shadow_path.write_text(content, encoding="utf-8")
        except Exception:
            pass
