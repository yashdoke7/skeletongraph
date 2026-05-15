"""
Prompt builder: attention-optimal context assembly with 5 layers and 5 modes.

Replaces zone_assembler.py as the primary assembly engine for v3.
zone_assembler.py is kept for backward compatibility but bypassed.

Assembly order (optimized for LLM attention — "Lost in the Middle"):
  1. [TOP]    Task prompt              → high attention (primacy)
  2. [TOP]    Constraints (L0 + L2)    → high attention
  3. [TOP]    Modifier instructions    → shapes reasoning before code
  4.          Session memory (L3)      → recent context
  5.          Architecture map (L1)    → reference material
  6.          Domain context (L2)      → domain-specific decisions
  7.          Caller/structural code   → neighbors
  8. [BOTTOM] Target code              → high attention (recency)
  9. [BOTTOM] Blast radius + verify    → actionable close
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set

from ..retrieval.classifier import ClassificationResult, ContextMode, QueryType, QueryMode, MODE_SPECS
from ..retrieval.resolver import RankedCandidate, ResolverResult, Tier
from ..retrieval.session import Session
from ..assembly.modifier import render_modifiers, estimate_modifier_tokens
from ..storage.local import IndexStore
from ..parser.skeleton import SkeletonCore
from ..config import SGConfig

logger = logging.getLogger(__name__)

# Token budget caps per mode (imported from classifier but kept here for reference)
_MODE_BUDGETS = {
    ContextMode.FAST: 1200,
    ContextMode.STANDARD: 4000,
    ContextMode.DEEP: 8000,
    ContextMode.PLANNING: 3500,
    ContextMode.REVIEW: 2000,
    ContextMode.PASS_THROUGH: 100,
}


@dataclass
class LayerContent:
    """Content loaded for a single layer."""
    name: str
    text: str
    tokens: int


@dataclass
class AssembledPrompt:
    """Output of the prompt builder."""
    text: str
    token_count: int
    mode: ContextMode
    query_type: QueryType
    confidence_level: str
    modifiers: List[str]
    extended_thinking: bool
    layers_loaded: List[str]
    layer_breakdown: Dict[str, int]
    targets: List[str]          # FQN list of primary targets
    reduction_ratio: float = 0.0
    session_dedup_count: int = 0
    warning: str = ""


def assemble(
    classification: ClassificationResult = None,
    resolver_result: ResolverResult = None,
    store: IndexStore = None,
    project_root: Path = None,
    session: Optional[Session] = None,
    # v4 additional params (optional, used by SGEngine)
    prompt: str = "",
    sg_dir: Optional[Path] = None,
    config: Optional[SGConfig] = None,
) -> AssembledPrompt:
    """Assemble attention-optimal context from classification + resolver results.

    This is the main entry point. Accepts both v3 and v4 call signatures.
    v4 params (prompt, sg_dir, config) are optional and enhance layer loading.
    """
    legacy_mode = classification.mode
    mode_spec = classification.mode_spec or MODE_SPECS[classification.query_mode]
    prompt = resolver_result.intent.raw_prompt

    # ── PASS_THROUGH: get out of the way ─────────────────────────────
    if legacy_mode == ContextMode.PASS_THROUGH:
        return _pass_through(classification, resolver_result)

    sg_dir = project_root / ".skeletongraph"
    budget = mode_spec.budget
    layers_loaded: List[str] = []
    layer_breakdown: Dict[str, int] = {}
    tokens_used = 0

    # ── Load layers based on mode ────────────────────────────────────

    # L0: Project DNA
    l0 = None
    if mode_spec.load_project:
        l0 = _load_file_layer(sg_dir / "project.md", "project", budget=300)
        if l0:
            layers_loaded.append("project")
            layer_breakdown["L0_project"] = l0.tokens

    # L1: Architecture Map
    l1 = None
    if mode_spec.load_architecture:
        l1 = _load_file_layer(sg_dir / "architecture.md", "architecture", budget=800)
        if l1:
            layers_loaded.append("architecture")
            layer_breakdown["L1_architecture"] = l1.tokens

    # L2: Domain Context
    l2_parts: List[LayerContent] = []
    if mode_spec.load_domain:
        target_fqns = {c.skeleton.fqn for c in resolver_result.candidates if c.tier == Tier.TIER1}
        domain_files = _detect_relevant_domains(target_fqns, store, sg_dir / "domain")
        for df in domain_files:
            dl = _load_file_layer(df, f"domain/{df.stem}", budget=400)
            if dl:
                l2_parts.append(dl)
                layers_loaded.append(f"domain/{df.stem}")
                layer_breakdown[f"L2_{df.stem}"] = dl.tokens

    # L3: Session Memory
    l3_parts: List[LayerContent] = []
    if mode_spec.load_current_session:
        l3_current = _load_file_layer(sg_dir / "session" / "current.md", "current_session", budget=250)
        if l3_current:
            l3_parts.append(l3_current)
            layers_loaded.append("current_session")
            layer_breakdown["L3_current"] = l3_current.tokens

    if mode_spec.load_recent_session:
        l3_recent = _load_file_layer(sg_dir / "session" / "recent.md", "recent_decisions", budget=500)
        if l3_recent:
            l3_parts.append(l3_recent)
            layers_loaded.append("recent_decisions")
            layer_breakdown["L3_recent"] = l3_recent.tokens

    if mode_spec.load_project_log:
        l3_log = _load_file_layer(sg_dir / "session" / "project_log.md", "project_log", budget=200)
        if l3_log:
            l3_parts.append(l3_log)
            layers_loaded.append("project_log")
            layer_breakdown["L3_log"] = l3_log.tokens

    # L4: Task Code
    l4_target_parts: List[str] = []
    l4_caller_parts: List[str] = []
    l4_blast_parts: List[str] = []
    l4_test_parts: List[str] = []
    l4_targets: List[str] = []
    session_dedup_count = 0

    candidates = resolver_result.candidates
    tier1 = [c for c in candidates if c.tier == Tier.TIER1]
    tier2 = [c for c in candidates if c.tier == Tier.TIER2]
    tier3 = [c for c in candidates if c.tier == Tier.TIER3]

    test_candidates = [c for c in candidates if _is_test_candidate(c)]
    test_fqns = {c.skeleton.fqn for c in test_candidates}

    # Target bodies or signatures
    if mode_spec.load_target_bodies:
        for c in tier1:
            sk = c.skeleton
            l4_targets.append(sk.fqn)

            if session and c.session_cached:
                header = f"# {sk.file_display} — {_short_name(sk.fqn)}"
                l4_target_parts.append(f"{header}\n{sk.signature}  # [body already provided]")
                session_dedup_count += 1
                continue

            body = _read_function_body(sk, project_root)
            if body:
                header = f"# {sk.file_display}:{sk.line_start} — {_short_name(sk.fqn)}"
                l4_target_parts.append(f"{header}\n{body}")
    else:
        for c in tier1:
            sk = c.skeleton
            l4_targets.append(sk.fqn)
            summary = _pick_summary(sk, store)
            entry = f"  {sk.signature}"
            if summary:
                entry += f"  # {summary[:80]}"
            l4_target_parts.append(entry)

    # Neighbor context (Tier 2/3)
    if mode_spec.load_neighbor_sigs:
        for c in sorted(tier2, key=lambda x: -x.score):
            if c.skeleton.fqn in test_fqns:
                continue
            sk = c.skeleton
            summary = _pick_summary(sk, store)
            entry = f"  {sk.signature}"
            if summary:
                entry += f"  # {summary[:80]}"
            l4_caller_parts.append(entry)

        if mode_spec.blast_depth > 1 or mode_spec.dep_depth > 1:
            for c in sorted(tier3, key=lambda x: -x.score)[:15]:
                l4_caller_parts.append(f"  {c.skeleton.to_tier3_str()}")

    # Tests (Tier 2 flagged by resolver)
    if mode_spec.load_tests and test_candidates:
        for c in sorted(test_candidates, key=lambda x: -x.score)[:10]:
            sk = c.skeleton
            if mode_spec.test_detail == "bodies":
                body = _read_function_body(sk, project_root)
                if body:
                    header = f"# {sk.file_display}:{sk.line_start} — {_short_name(sk.fqn)}"
                    l4_test_parts.append(f"{header}\n{body}")
            else:
                entry = f"  {sk.signature}  # {sk.file_display}"
                l4_test_parts.append(entry)

    # Blast radius
    if l4_targets:
        for fqn in l4_targets[:3]:
            blast = store.graph.blast_radius(
                fqn,
                max_depth=2 if (mode_spec.blast_depth > 1 or mode_spec.dep_depth > 1) else 1,
            )
            if blast:
                lines = [f"Blast radius for {_short_name(fqn)}:"]
                for affected_fqn, dist in sorted(blast.items(), key=lambda x: x[1])[:10]:
                    affected_sk = store.skeleton_table.get(affected_fqn)
                    if affected_sk:
                        lines.append(
                            f"  {'>' * dist} {_short_name(affected_fqn)} ({affected_sk.file_display})"
                        )
                l4_blast_parts.append("\n".join(lines))

    if l4_test_parts:
        layers_loaded.append("tests")

    layers_loaded.append("task_code")

    # ── Compute L4 tokens ────────────────────────────────────────────
    l4_target_text = "\n\n".join(l4_target_parts)
    l4_caller_text = "\n".join(l4_caller_parts)
    l4_test_text = "\n\n".join(l4_test_parts)
    l4_blast_text = "\n\n".join(l4_blast_parts)
    layer_breakdown["L4_target"] = _est_tokens(l4_target_text)
    layer_breakdown["L4_callers"] = _est_tokens(l4_caller_text)
    layer_breakdown["L4_tests"] = _est_tokens(l4_test_text)
    layer_breakdown["L4_blast"] = _est_tokens(l4_blast_text)

    # ── Modifiers ────────────────────────────────────────────────────
    modifier_text = render_modifiers(classification.modifiers)
    layer_breakdown["modifiers"] = _est_tokens(modifier_text)

    # ── Budget enforcement (truncation if over) ──────────────────────
    total_est = sum(layer_breakdown.values()) + _est_tokens(prompt) + 20  # overhead
    warning = ""
    if total_est > budget:
        warning = _truncate_to_budget(
            l4_target_parts,
            l4_caller_parts,
            l4_test_parts,
            l4_blast_parts,
            l1,
            l2_parts,
            layer_breakdown,
            budget,
            total_est,
        )
        # Recompute texts after truncation
        l4_target_text = "\n\n".join(l4_target_parts)
        l4_caller_text = "\n".join(l4_caller_parts)
        l4_test_text = "\n\n".join(l4_test_parts)
        l4_blast_text = "\n\n".join(l4_blast_parts)

    # ── Constraints text (L0 constraints + L2 domain constraints) ────
    constraints_parts: List[str] = []
    if l0 and l0.text:
        constraints_parts.append(l0.text)

    # ── Assembly (attention-optimal order) ────────────────────────────
    sections: List[str] = []

    # 1. Task (TOP — primacy attention)
    sections.append(f"## Task\n{prompt}")

    # 2. Constraints (TOP)
    if constraints_parts:
        sections.append("## Constraints\n" + "\n".join(constraints_parts))

    # 3. Modifiers (TOP — must come before code to shape reasoning)
    if modifier_text:
        sections.append(modifier_text)

    # 4. Session memory (L3)
    if l3_parts:
        session_text = "\n\n".join(f"### {p.name}\n{p.text}" for p in l3_parts)
        sections.append(f"## Session\n{session_text}")

    # 5. Architecture (L1 — middle, reference)
    if l1 and l1.text:
        sections.append(f"## Architecture\n{l1.text}")

    # 6. Domain context (L2 — middle)
    if l2_parts:
        domain_text = "\n\n".join(f"### {p.name}\n{p.text}" for p in l2_parts)
        sections.append(f"## Domain\n{domain_text}")

    # 7. Callers/structural (L4 neighbors)
    if l4_caller_text:
        header = "## Context" if legacy_mode != ContextMode.PLANNING else "## Module Signatures"
        sections.append(f"{header}\n{l4_caller_text}")

    # 7b. Tests (if requested by mode)
    if l4_test_text:
        sections.append(f"## Tests\n{l4_test_text}")

    # 8. Target code (BOTTOM — recency attention)
    if l4_target_text:
        sections.append(f"## Target Code\n{l4_target_text}")

    # 9. Blast radius + verification (BOTTOM)
    if l4_blast_text:
        sections.append(f"## Blast Radius\n{l4_blast_text}")

    assembled = "\n\n---\n\n".join(sections)
    total_tokens = _est_tokens(assembled)

    # Estimate reduction ratio
    raw_tokens = _estimate_raw_tokens(resolver_result.candidates, project_root)
    reduction = raw_tokens / max(total_tokens, 1) if raw_tokens > 0 else 0

    # Record to session
    if session and legacy_mode not in (ContextMode.PLANNING, ContextMode.REVIEW):
        fqns_returned = {c.skeleton.fqn for c in resolver_result.candidates}
        zone2_fqns = {c.skeleton.fqn for c in resolver_result.candidates if c.tier == Tier.TIER1}
        session.record_turn(
            prompt=prompt,
            fqns_returned=fqns_returned,
            zone2_fqns=zone2_fqns,
            token_count=total_tokens,
            estimated_native_tokens=raw_tokens,
            confidence=resolver_result.confidence,
        )

    return AssembledPrompt(
        text=assembled,
        token_count=total_tokens,
        mode=legacy_mode,
        query_type=classification.query_type,
        confidence_level=resolver_result.confidence,
        modifiers=classification.modifiers,
        extended_thinking=classification.extended_thinking,
        layers_loaded=layers_loaded,
        layer_breakdown=layer_breakdown,
        targets=l4_targets,
        reduction_ratio=round(reduction, 1),
        session_dedup_count=session_dedup_count,
        warning=warning,
    )


# ── 4-zone assembler (P2 canonical) ─────────────────────────────────────


@dataclass
class Zone4Result:
    """Output of the 4-zone assembler."""
    text: str
    token_count: int
    zones: Dict[str, int]          # zone_name → token count
    targets: List[str]
    warning: str = ""


def assemble_4zone(
    prompt: str,
    resolver_result: ResolverResult,
    store: IndexStore,
    project_root: Path,
    sg_dir: Optional[Path] = None,
    budget: int = 8000,
    session_digest: str = "",
) -> Zone4Result:
    """4-zone context assembly per the canonical SG plan v3.

    Zones (primacy → recency):
      Zone 1 — Constraints: project rules, never trimmed
      Zone 2 — Curated retrieval: Tier-1 target bodies + summaries
      Zone 3 — Peripheral: Tier-2/3 signatures, neighbors
      Zone 4 — Current request: the user's prompt

    Trim order when over budget: Zone 3 first, then Zone 2 signatures,
    then Zone 2 bodies → signatures. Zone 1 and Zone 4 never trimmed.
    """
    if sg_dir is None:
        sg_dir = project_root / ".skeletongraph"

    zones: Dict[str, int] = {}
    targets: List[str] = []

    # ── Zone 1: Constraints ───────────────────────────────────────────
    z1_parts: List[str] = []
    cs_file = sg_dir / "constraints.md"
    if cs_file.exists():
        cs_text = cs_file.read_text(encoding="utf-8", errors="replace").strip()
        if cs_text:
            z1_parts.append(f"## Constraints\n{cs_text}")
    # Session digest (compact 5-turn) also goes in Zone 1 for visibility
    if session_digest:
        z1_parts.append(session_digest)

    z1_text = "\n\n".join(z1_parts)
    zones["z1_constraints"] = _est_tokens(z1_text)

    # ── Zone 4: Current request (built early to reserve budget) ───────
    z4_text = f"## Task\n{prompt}"
    zones["z4_request"] = _est_tokens(z4_text)

    # ── Remaining budget for Zones 2 + 3 ─────────────────────────────
    reserved = zones["z1_constraints"] + zones["z4_request"] + 20
    inner_budget = max(budget - reserved, 500)

    # ── Zone 2: Curated retrieval (Tier-1 targets) ────────────────────
    tier1 = [c for c in resolver_result.candidates if c.tier == Tier.TIER1]
    tier2_3 = [c for c in resolver_result.candidates if c.tier != Tier.TIER1]

    z2_parts: List[str] = []
    for c in tier1:
        sk = c.skeleton
        targets.append(sk.fqn)
        body = _read_function_body(sk, project_root)
        if body:
            header = f"# {sk.file_path}:{sk.line_start} — {_short_name(sk.fqn)}"
            z2_parts.append(f"{header}\n{body}")
        else:
            summary = _pick_summary(sk, store)
            entry = f"  {sk.signature}"
            if summary:
                entry += f"  # {summary[:80]}"
            z2_parts.append(entry)

    z2_text = "\n\n".join(z2_parts)
    zones["z2_curated"] = _est_tokens(z2_text)

    # ── Zone 3: Peripheral (Tier-2/3 signatures) ──────────────────────
    z3_parts: List[str] = []
    for c in sorted(tier2_3, key=lambda x: -x.score):
        sk = c.skeleton
        summary = _pick_summary(sk, store)
        entry = f"  {sk.signature}"
        if summary:
            entry += f"  # {summary[:80]}"
        z3_parts.append(entry)

    z3_text = "\n".join(z3_parts)
    zones["z3_peripheral"] = _est_tokens(z3_text)

    # ── Budget enforcement: trim Zone 3, then Zone 2 ──────────────────
    current = zones["z2_curated"] + zones["z3_peripheral"]
    warning = ""

    if current > inner_budget:
        # 1. Drop peripheral from back
        while _est_tokens("\n".join(z3_parts)) > (inner_budget - zones["z2_curated"]) and z3_parts:
            z3_parts.pop()
        z3_text = "\n".join(z3_parts)
        zones["z3_peripheral"] = _est_tokens(z3_text)

    current = zones["z2_curated"] + zones["z3_peripheral"]
    if current > inner_budget and z2_parts:
        # 2. Replace Zone 2 bodies with signatures
        new_z2: List[str] = []
        for c in tier1:
            sk = c.skeleton
            summary = _pick_summary(sk, store)
            entry = f"  {sk.signature}"
            if summary:
                entry += f"  # {summary[:80]}"
            new_z2.append(entry)
        z2_parts = new_z2
        z2_text = "\n".join(z2_parts)
        zones["z2_curated"] = _est_tokens(z2_text)
        warning = "Over budget: Zone 2 bodies replaced with signatures"

    # ── Assemble in zone order ─────────────────────────────────────────
    sections: List[str] = []

    if z1_text:
        sections.append(z1_text)

    if z2_text:
        sections.append(f"## Context\n{z2_text}")

    if z3_text:
        sections.append(f"## Related\n{z3_text}")

    sections.append(z4_text)

    assembled = "\n\n---\n\n".join(sections)
    total = _est_tokens(assembled)

    return Zone4Result(
        text=assembled,
        token_count=total,
        zones=zones,
        targets=targets,
        warning=warning,
    )


# ── Pass-through (MISS confidence) ──────────────────────────────────────

def _pass_through(
    classification: ClassificationResult,
    resolver_result: ResolverResult,
) -> AssembledPrompt:
    """Emit minimal context when graph found nothing useful."""
    text = (
        "SkeletonGraph: No relevant context found for this task.\n"
        "The graph doesn't have a confident match for your query.\n"
        "Explore freely using normal file reading and search.\n"
        "The graph is available for specific function lookups via query_context "
        "if you identify a target function name."
    )
    return AssembledPrompt(
        text=text,
        token_count=_est_tokens(text),
        mode=ContextMode.PASS_THROUGH,
        query_type=classification.query_type,
        confidence_level="MISS",
        modifiers=[],
        extended_thinking=False,
        layers_loaded=[],
        layer_breakdown={},
        targets=[],
        warning="PASS_THROUGH: graph found nothing relevant",
    )


# ── Layer loading helpers ────────────────────────────────────────────────

def _load_file_layer(path: Path, name: str, budget: int = 500) -> Optional[LayerContent]:
    """Load a markdown file as a layer. Returns None if missing or empty."""
    if not path.exists():
        return None
    try:
        text = path.read_text(encoding="utf-8", errors="replace").strip()
        if not text:
            return None
        tokens = _est_tokens(text)
        if tokens > budget:
            # Truncate to budget
            chars = budget * 4  # ~4 chars per token
            text = text[:chars] + "\n[...truncated]"
            tokens = budget
        return LayerContent(name=name, text=text, tokens=tokens)
    except Exception as e:
        logger.warning(f"Failed to load layer {name} from {path}: {e}")
        return None


def _detect_relevant_domains(
    target_fqns: Set[str],
    store: IndexStore,
    domain_dir: Path,
) -> List[Path]:
    """Match target FQN file paths to domain note filenames."""
    if not domain_dir.exists():
        return []
    available = {p.stem: p for p in domain_dir.glob("*.md")}
    if not available:
        return []

    matched: List[Path] = []
    for fqn in target_fqns:
        sk = store.skeleton_table.get(fqn)
        if not sk:
            continue
        parts = sk.file_path.replace("\\", "/").split("/")
        for part in parts[:-1]:  # skip filename
            if part in available and available[part] not in matched:
                matched.append(available[part])
    return matched


def _read_function_body(sk: SkeletonCore, project_root: Path) -> str:
    """Read the full function body from disk using line numbers."""
    file_path = project_root / sk.file_path
    if not file_path.exists():
        return ""
    try:
        lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
        body_lines = lines[sk.line_start - 1:sk.line_end]
        return "\n".join(body_lines)
    except Exception:
        return ""


def _short_name(fqn: str) -> str:
    """Extract short name from FQN: 'file.py::Class.method' → 'Class.method'"""
    return fqn.split("::")[-1] if "::" in fqn else fqn


def _pick_summary(sk: SkeletonCore, store: IndexStore) -> str:
    """Pick the best available summary (LLM summary or docstring fallback)."""
    if sk.docstring:
        return sk.docstring.strip().splitlines()[0]
    summary = store.summaries.get(sk.fqn) or ""
    if summary:
        return summary
    return ""


def _is_test_candidate(candidate: RankedCandidate) -> bool:
    """Return True when a candidate looks like test code."""
    sk = candidate.skeleton
    path = sk.file_path.replace("\\", "/").lower()
    name = sk.fqn.split("::")[-1].lower()
    return (
        "/test" in path
        or path.startswith("test")
        or "_test." in path
        or path.endswith("_test.py")
        or path.endswith("test.py")
        or name.startswith("test_")
        or ".test_" in name
    )


# ── Token estimation ─────────────────────────────────────────────────────

def _est_tokens(text: str) -> int:
    """Estimate token count. ~4 chars per token for mixed code/English."""
    if not text:
        return 0
    return max(1, len(text) // 4)


def _estimate_raw_tokens(candidates: List[RankedCandidate], project_root: Path) -> int:
    """Estimate how many tokens native file reading would cost."""
    files_seen: Set[str] = set()
    total = 0
    for c in candidates:
        fp = c.skeleton.file_path
        if fp not in files_seen:
            files_seen.add(fp)
            full_path = project_root / fp
            if full_path.exists():
                try:
                    content = full_path.read_text(encoding="utf-8", errors="replace")
                    total += _est_tokens(content)
                except Exception:
                    pass
    return total


# ── Budget enforcement ───────────────────────────────────────────────────

def _truncate_to_budget(
    target_parts: List[str],
    caller_parts: List[str],
    test_parts: List[str],
    blast_parts: List[str],
    l1: Optional[LayerContent],
    l2_parts: List[LayerContent],
    breakdown: Dict[str, int],
    budget: int,
    current: int,
) -> str:
    """Truncate content to fit within budget. Returns warning string.

    Truncation priority (least important first):
    1. Drop 2-hop neighbors (bottom of caller_parts)
    2. Trim blast radius → direct callers only
    3. Trim L1 architecture → top-level only
    4. Trim L2 domain → summary only
    5. Trim test bodies → signatures only
    Keep: L0, target full body (always preserved)
    """
    trimmed: List[str] = []

    # 1. Trim caller_parts from the bottom
    while current > budget and len(caller_parts) > 3:
        removed = caller_parts.pop()
        current -= _est_tokens(removed)
        trimmed.append("periphery")

    # 2. Trim blast radius
    while current > budget and len(blast_parts) > 1:
        removed = blast_parts.pop()
        current -= _est_tokens(removed)
        trimmed.append("blast_radius")

    # 3. Trim test bodies to signatures
    if current > budget and test_parts:
        for i, entry in enumerate(list(test_parts)):
            if current <= budget:
                break
            if "\n" in entry:
                head = entry.split("\n", 1)[0]
                new_entry = f"{head}  # [truncated]"
                current -= _est_tokens(entry)
                current += _est_tokens(new_entry)
                test_parts[i] = new_entry
                trimmed.append("tests")

    # 4. Trim L1
    if current > budget and l1 and l1.text:
        # Keep only first 200 chars
        old_tokens = l1.tokens
        l1.text = l1.text[:200] + "\n[...truncated]"
        l1.tokens = _est_tokens(l1.text)
        current -= (old_tokens - l1.tokens)
        if "L1_architecture" in breakdown:
            breakdown["L1_architecture"] = l1.tokens
        trimmed.append("architecture")

    # 5. Trim L2
    if current > budget and l2_parts:
        for dl in l2_parts:
            old_tokens = dl.tokens
            dl.text = dl.text[:100] + "\n[...truncated]"
            dl.tokens = _est_tokens(dl.text)
            current -= (old_tokens - dl.tokens)
        trimmed.append("domain")

    if trimmed:
        return f"Over budget by {current - budget} tokens. Trimmed: {', '.join(set(trimmed))}"
    return ""
