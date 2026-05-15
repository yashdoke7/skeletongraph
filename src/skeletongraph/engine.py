"""
SGEngine — unified pipeline entry point for v4.

Single class that orchestrates the 3-phase pipeline:
  Phase 1: UNDERSTAND (regex + SLM)
  Phase 2: RETRIEVE (graph traversal, zero cost)
  Phase 3: ASSEMBLE (prompt builder for main LLM)

Used by:
  - MCP server (server/mcp.py)
  - Claude Code hooks (hooks/claude_code.py)
  - CLI commands (cli/)

Replaces duplicated pipeline logic that was in mcp.py and zone_assembler.py.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from .config import SGConfig, load_config
from .storage.local import IndexStore, load_index
from .retrieval.intent import Intent, analyze_intent
from .retrieval.resolver import (
    resolve_context, ResolverResult, RankedCandidate, Tier,
)
from .retrieval.classifier import (
    classify_query, ClassificationResult, QueryMode, ModelTier, MODE_SPECS,
)
from .retrieval.confidence import compute_confidence
from .retrieval.model_router import route_model_tier
from .retrieval.session import Session
from .retrieval.slm_extractor import (
    slm_extract, SLMResult, prefilter_for_slm,
)

logger = logging.getLogger(__name__)


# ── Pipeline Result ─────────────────────────────────────────────────────


@dataclass
class PipelineResult:
    """Complete output of the v4 pipeline."""

    # Assembled context for the main LLM
    context_text: str = ""

    # Classification
    query_mode: QueryMode = QueryMode.BUILD_GUIDED
    model_tier: ModelTier = ModelTier.MLM
    recommended_model: str = ""
    delivery: str = "ide"
    base_model_tier: ModelTier = ModelTier.MLM
    complexity_score: float = 0.0
    routing_reason: str = ""

    # Retrieval details
    candidates: List[RankedCandidate] = field(default_factory=list)
    confidence: str = "MEDIUM"
    confidence_reason: str = ""

    # SLM details
    slm_used: bool = False
    slm_reasoning: str = ""
    slm_entities_found: int = 0

    # Cost tracking
    slm_input_tokens: int = 0
    slm_output_tokens: int = 0
    slm_cost_usd: float = 0.0
    context_tokens: int = 0
    saved_vs_raw_tokens: int = 0

    # Timing
    phase1_ms: float = 0.0
    phase2_ms: float = 0.0
    phase3_ms: float = 0.0
    total_ms: float = 0.0

    # Status
    success: bool = True
    error: str = ""
    pipeline_path: str = ""   # "regex", "slm", "slm_retry", "passthrough"

    def cost_summary(self) -> Dict[str, Any]:
        """Cost transparency dict for MCP responses."""
        return {
            "slm_tokens": {
                "input": self.slm_input_tokens,
                "output": self.slm_output_tokens,
            },
            "slm_cost_usd": round(self.slm_cost_usd, 6),
            "context_tokens": self.context_tokens,
            "recommended_tier": self.model_tier.value,
            "recommended_model": self.recommended_model,
            "delivery": self.delivery,
            "base_tier": self.base_model_tier.value,
            "complexity_score": self.complexity_score,
            "routing_reason": self.routing_reason,
            "saved_vs_raw_tokens": self.saved_vs_raw_tokens,
            "pipeline_path": self.pipeline_path,
            "latency_ms": round(self.total_ms, 1),
        }


# ── SGEngine ────────────────────────────────────────────────────────────


class SGEngine:
    """Unified v4 pipeline engine.

    Usage:
        engine = SGEngine(project_root="/path/to/project")
        result = engine.query("fix the content-length bug")
        print(result.context_text)
        print(result.cost_summary())
    """

    def __init__(
        self,
        project_root: Optional[Path] = None,
        config: Optional[SGConfig] = None,
    ):
        self._root = Path(project_root) if project_root else Path.cwd()
        self._sg_dir = self._root / ".skeletongraph"
        self._config = config or load_config(self._root)
        self._store: Optional[IndexStore] = None
        self._session: Optional[Session] = None
        self._last_assembly_tokens: int = 0
        self._last_assembly_native_estimate: int = 0

    def _ensure_loaded(self) -> IndexStore:
        """Lazy-load the index store."""
        if self._store is None:
            if not self._sg_dir.exists():
                raise RuntimeError(
                    "No SkeletonGraph index found. Run `sg build` first."
                )
            store = load_index(self._root)
            if store is None:
                raise RuntimeError(
                    "No SkeletonGraph index metadata found. Run `sg build` first."
                )
            self._store = store
        return self._store

    def _ensure_session(self) -> Session:
        """Get or create session."""
        if self._session is None:
            self._session = Session(
                ttl_minutes=self._config.session_ttl_minutes,
                max_turns=self._config.session_max_turns,
            )
        return self._session

    # ── Main Entry Point ────────────────────────────────────────────────

    def query(
        self,
        prompt: str,
        entities: Optional[Set[str] | List[str]] = None,
        max_retries: int = 2,
        exclude_fqns: Optional[Set[str]] = None,
        delivery: str = "ide",
        force_slm: bool = False,
    ) -> PipelineResult:
        """Execute the full v4 pipeline: Understand → Retrieve → Assemble.

        Args:
            prompt: User's natural language request.
            max_retries: Max SLM retry attempts on MISS.
            exclude_fqns: FQNs to exclude (for supplementary queries).

        Returns:
            PipelineResult with assembled context and metadata.
        """
        total_start = time.time()
        result = PipelineResult()
        result.delivery = delivery

        try:
            store = self._ensure_loaded()
        except RuntimeError as e:
            result.success = False
            result.error = str(e)
            return result

        session = self._ensure_session() if self._config.enable_session else None

        for attempt in range(max_retries + 1):
            # ── Phase 1: UNDERSTAND ─────────────────────────────────────
            p1_start = time.time()
            intent, slm_result, regex_confidence = self._phase1_understand(
                prompt, store, session,
                retry_note="" if attempt == 0 else f"Attempt {attempt + 1}: Previous attempt found {result.slm_entities_found} entities. Broaden search.",
                force_slm=force_slm,
            )
            result.phase1_ms = (time.time() - p1_start) * 1000

            # Classify mode
            classification = classify_query(
                intent,
                confidence=None,  # Will be set after resolution
                target_fqns=set(),
                n_files_involved=0,
                slm_result=slm_result,
            )

            # Track SLM usage
            if slm_result and slm_result.success:
                result.slm_used = True
                result.slm_reasoning = slm_result.reasoning
                result.slm_entities_found = len(slm_result.entities)
                result.slm_input_tokens = slm_result.input_tokens
                result.slm_output_tokens = slm_result.output_tokens
                result.slm_cost_usd = slm_result.cost_usd
                result.pipeline_path = "slm"
            else:
                result.pipeline_path = "regex"

            # ── Phase 2: RETRIEVE ───────────────────────────────────────
            p2_start = time.time()
            resolver_result = self._phase2_retrieve(
                prompt, intent, slm_result, classification,
                store, session, exclude_fqns, set(entities or []),
            )
            result.phase2_ms = (time.time() - p2_start) * 1000

            # Check if we got anything useful
            result.candidates = resolver_result.candidates
            result.confidence = resolver_result.confidence
            result.confidence_reason = resolver_result.confidence_reason

            if not resolver_result.candidates and attempt < max_retries:
                # Retry with broadened search
                result.pipeline_path = "slm_retry"
                continue

            if resolver_result.confidence == "MISS" and attempt < max_retries:
                result.pipeline_path = "slm_retry"
                continue

            # Got results — proceed to assembly
            break

        # If still nothing after retries
        if not resolver_result.candidates and result.confidence in ("LOW", "MISS"):
            result.pipeline_path = "passthrough"

        # ── Phase 3: ASSEMBLE ───────────────────────────────────────────
        p3_start = time.time()
        result.context_text = self._phase3_assemble(
            prompt, resolver_result, classification,
            slm_result, store, session,
        )
        result.phase3_ms = (time.time() - p3_start) * 1000

        # Set final metadata
        result.query_mode = classification.query_mode
        result.base_model_tier = MODE_SPECS[classification.query_mode].tier

        # Token counting
        result.context_tokens = self._last_assembly_tokens or len(result.context_text) // 4
        result.saved_vs_raw_tokens = self._last_assembly_native_estimate

        if self._config.enable_dynamic_model_routing:
            routing = route_model_tier(
                mode=classification.query_mode,
                base_tier=result.base_model_tier,
                confidence=result.confidence,
                candidate_count=len(result.candidates),
                context_tokens=result.context_tokens,
                slm_used=result.slm_used,
            )
            result.model_tier = routing.tier
            result.complexity_score = routing.complexity_score
            result.routing_reason = routing.reason
            result.recommended_model = self._model_for_tier(routing.tier.value, delivery)
        else:
            result.model_tier = result.base_model_tier
            result.recommended_model = self._model_for_tier(result.model_tier.value, delivery)
            result.routing_reason = "static mode routing"

        result.total_ms = (time.time() - total_start) * 1000
        result.success = True

        logger.info(
            "Pipeline complete: mode=%s tier=%s confidence=%s "
            "context_tokens=%d latency=%.0fms path=%s",
            result.query_mode.value, result.model_tier.value,
            result.confidence, result.context_tokens,
            result.total_ms, result.pipeline_path,
        )

        return result

    # ── Phase 1: UNDERSTAND ─────────────────────────────────────────────

    def _phase1_understand(
        self,
        prompt: str,
        store: IndexStore,
        session: Optional[Session],
        retry_note: str = "",
        force_slm: bool = False,
    ):
        """Phase 1: Regex + (optionally) SLM entity extraction.

        Returns (intent, slm_result, regex_confidence).
        """
        # Step 1: Regex entity extraction (~1ms, always)
        known_files = set(store.file_skeletons.keys())
        known_fqns = set(store.skeleton_table.keys())
        intent = analyze_intent(prompt, known_files, known_fqns)

        # Step 2: Determine if SLM is needed
        regex_confidence = "HIGH" if intent.function_names else (
            "MEDIUM" if intent.file_paths else "LOW"
        )

        slm_result = None

        # Skip SLM conditions:
        # 1. Regex found exact FQN matches → HIGH confidence
        # 2. Tiny project (<10 functions) → graph handles everything
        # 3. SLM disabled in config
        should_skip_slm = (
            (not force_slm and regex_confidence == "HIGH")
            or len(store.skeleton_table) < 10
            or not self._config.enable_slm_fallback
        )

        if not should_skip_slm:
            # Get session FQNs for pre-filter
            session_fqns = None
            if session and session.turn_count > 0:
                session_fqns = set(session.get_last_target_fqns())

            slm_result = slm_extract(
                prompt=prompt,
                store=store,
                sg_dir=self._sg_dir,
                config=self._config,
                session_fqns=session_fqns,
                retry_note=retry_note,
            )

            # Merge SLM entities into intent
            if slm_result.success and slm_result.entities:
                from .retrieval.intent import Entity
                for e in slm_result.entities:
                    intent.entities.append(Entity(
                        value=e.fqn,
                        entity_type="slm_entity",
                        confidence=0.8,
                    ))
                    # Try to resolve to function_names
                    if e.fqn in known_fqns or any(
                        e.fqn.endswith(f"::{fn}") for fn in intent.function_names
                    ):
                        intent.function_names.append(e.fqn)

        return intent, slm_result, regex_confidence

    # ── Phase 2: RETRIEVE ───────────────────────────────────────────────

    def _phase2_retrieve(
        self,
        prompt: str,
        intent: Intent,
        slm_result: Optional[SLMResult],
        classification: ClassificationResult,
        store: IndexStore,
        session: Optional[Session],
        exclude_fqns: Optional[Set[str]] = None,
        seed_fqns: Optional[Set[str]] = None,
    ) -> ResolverResult:
        """Phase 2: Graph traversal and candidate ranking.

        Uses the existing resolver but feeds it SLM-enriched intent.
        Zero LLM cost — pure Python computation.
        """
        mode_spec = classification.mode_spec or MODE_SPECS[classification.query_mode]

        # Use mode-specific graph depth
        max_depth = max(mode_spec.blast_depth, mode_spec.dep_depth)
        if max_depth == 0:
            max_depth = 1  # At least 1-hop for basic context

        resolver_result = resolve_context(
            prompt=prompt,
            store=store,
            max_depth=max_depth,
            session=session,
            top_n=50,
            seed_fqns=seed_fqns,
            mode_spec=mode_spec,
            enable_keyword_fallback=self._config.enable_keyword_fallback,
            enable_bm25_fallback=self._config.enable_bm25_fallback,
        )

        # Filter excluded FQNs (for supplementary queries)
        if exclude_fqns:
            resolver_result.candidates = [
                c for c in resolver_result.candidates
                if c.skeleton.fqn not in exclude_fqns
            ]

        return resolver_result

    # ── Phase 3: ASSEMBLE ───────────────────────────────────────────────

    def _phase3_assemble(
        self,
        prompt: str,
        resolver_result: ResolverResult,
        classification: ClassificationResult,
        slm_result: Optional[SLMResult],
        store: IndexStore,
        session: Optional[Session],
    ) -> str:
        """Phase 3: Build structured prompt for main LLM.

        Delegates to prompt_builder with v4 mode-aware layer loading.
        """
        try:
            from .assembly.prompt_builder import assemble
            assembled = assemble(
                prompt=prompt,
                resolver_result=resolver_result,
                classification=classification,
                store=store,
                project_root=self._root,
                sg_dir=self._sg_dir,
                session=session,
                config=self._config,
            )
            context_text = assembled.text if hasattr(assembled, "text") else str(assembled)
            self._last_assembly_tokens = getattr(assembled, "token_count", 0) or len(context_text) // 4
            reduction_ratio = getattr(assembled, "reduction_ratio", 0.0) or 0.0
            self._last_assembly_native_estimate = (
                int(reduction_ratio * self._last_assembly_tokens)
                if reduction_ratio > 0
                else 0
            )
        except Exception as e:
            logger.warning("prompt_builder.assemble failed: %s, using minimal assembly", e)
            context_text = self._minimal_assemble(
                prompt, resolver_result, slm_result
            )
            self._last_assembly_tokens = len(context_text) // 4
            self._last_assembly_native_estimate = 0

        # Prepend SLM reasoning if available
        if slm_result and slm_result.reasoning:
            context_text = (
                f"## Retrieval Analysis\n{slm_result.reasoning}\n\n"
                + context_text
            )
            self._last_assembly_tokens = len(context_text) // 4

        return context_text

    def _minimal_assemble(
        self,
        prompt: str,
        resolver_result: ResolverResult,
        slm_result: Optional[SLMResult],
    ) -> str:
        """Minimal fallback assembly if prompt_builder fails."""
        parts = [f"## Task\n{prompt}\n"]

        if slm_result and slm_result.reasoning:
            parts.append(f"## Analysis\n{slm_result.reasoning}\n")

        for c in resolver_result.candidates[:10]:
            sk = c.skeleton
            if c.tier == Tier.TIER1:
                parts.append(f"## {sk.fqn}\n{sk.signature}\n")
            else:
                parts.append(f"  {sk.signature}  # {sk.fqn}")

        return "\n".join(parts)

    # ── Utility Methods ─────────────────────────────────────────────────

    def expand(
        self,
        target: str,
        expand_type: str = "auto",
        start_line: Optional[int] = None,
        end_line: Optional[int] = None,
        include_neighbors: bool = False,
        max_tokens: int = 4000,
    ) -> str:
        """Universal expand — any content type.

        Args:
            target: FQN, file path, or directory path.
            expand_type: "function" | "class" | "file" | "range" | "directory" | "auto"
            start_line: Start line for range expansion.
            end_line: End line for range expansion.
            include_neighbors: Include 1-hop neighbors.
            max_tokens: Max tokens to return.

        Returns:
            Expanded content as string.
        """
        store = self._ensure_loaded()

        # Auto-detect type
        if expand_type == "auto":
            if "::" in target:
                expand_type = "function"
            elif start_line is not None and end_line is not None:
                expand_type = "range"
            elif target.endswith("/") or not "." in target.split("/")[-1]:
                expand_type = "directory"
            else:
                expand_type = "file"

        if expand_type == "function":
            return self._expand_function(target, store, include_neighbors)
        elif expand_type == "class":
            return self._expand_class(target, store)
        elif expand_type == "range":
            return self._expand_range(target, start_line, end_line)
        elif expand_type == "file":
            return self._expand_file(target, max_tokens)
        elif expand_type == "directory":
            return self._expand_directory(target, store)
        else:
            return f"Unknown expand type: {expand_type}"

    def _expand_function(self, fqn: str, store: IndexStore, include_neighbors: bool) -> str:
        """Expand a single function by FQN."""
        sk = store.skeleton_table.get(fqn)
        if not sk:
            # Try partial match
            for k, v in store.skeleton_table.items():
                if k.endswith(fqn) or fqn in k:
                    sk = v
                    fqn = k
                    break
        if not sk:
            return f"Function not found: {fqn}"

        file_path = self._root / sk.file_path
        if not file_path.exists():
            return f"File not found: {sk.file_path}"

        lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
        start = max(0, sk.line_start - 1)
        end = min(len(lines), sk.line_end)
        body = "\n".join(lines[start:end])

        parts = [f"# {sk.file_path}:{sk.line_start} — {fqn}\n{body}"]

        if include_neighbors:
            # 1-hop callers
            callers = store.graph.blast_radius(fqn, max_depth=1)
            for caller_fqn in list(callers.keys())[:5]:
                caller_sk = store.skeleton_table.get(caller_fqn)
                if caller_sk:
                    summary = store.summaries.get(caller_fqn, "")
                    parts.append(f"  {caller_sk.signature}  # {summary[:60]}")

        return "\n".join(parts)

    def _expand_class(self, class_fqn: str, store: IndexStore) -> str:
        """Expand all methods of a class."""
        methods = [
            (fqn, sk) for fqn, sk in store.skeleton_table.items()
            if fqn.startswith(class_fqn) or ("::" in fqn and fqn.split("::")[1].startswith(class_fqn.split("::")[-1]))
        ]
        if not methods:
            return f"Class not found: {class_fqn}"

        parts = []
        for fqn, sk in sorted(methods, key=lambda x: x[1].line_start):
            parts.append(f"  {sk.signature}")
        return "\n".join(parts)

    def _expand_range(self, file_path: str, start: Optional[int], end: Optional[int]) -> str:
        """Expand a line range from a file."""
        full_path = self._root / file_path
        if not full_path.exists():
            return f"File not found: {file_path}"

        lines = full_path.read_text(encoding="utf-8", errors="replace").splitlines()
        s = max(0, (start or 1) - 1)
        e = min(len(lines), end or len(lines))
        return "\n".join(f"{i+1}: {lines[i]}" for i in range(s, e))

    def _expand_file(self, file_path: str, max_tokens: int) -> str:
        """Expand full file content (capped)."""
        full_path = self._root / file_path
        if not full_path.exists():
            return f"File not found: {file_path}"

        text = full_path.read_text(encoding="utf-8", errors="replace")
        max_chars = max_tokens * 4
        if len(text) > max_chars:
            text = text[:max_chars] + f"\n\n[... truncated at {max_tokens} tokens ...]"
        return text

    def _expand_directory(self, dir_path: str, store: IndexStore) -> str:
        """Expand directory outline: all files + function signatures."""
        lines = []
        for file_path, skeletons in sorted(store.file_skeletons.items()):
            if file_path.startswith(dir_path) or dir_path in file_path:
                lines.append(f"\n# {file_path}")
                for sk in skeletons:
                    lines.append(f"  {sk.signature}")
        if not lines:
            return f"No indexed files in: {dir_path}"
        return "\n".join(lines)

    def get_session(self) -> Optional[Session]:
        """Access the current session (for post-turn processing)."""
        return self._session

    def get_config(self) -> SGConfig:
        """Access the current configuration."""
        return self._config

    def get_store(self) -> IndexStore:
        """Access the loaded index store (lazy loaded)."""
        return self._ensure_loaded()

    def _model_for_tier(self, tier: str, delivery: str) -> str:
        """Resolve a tier to an IDE label or CLI provider model."""
        if delivery == "cli":
            return self._config.get_cli_model_for_tier(tier)
        return self._config.get_model_for_tier(tier)
