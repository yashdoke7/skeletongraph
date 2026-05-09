"""Deterministic model-tier routing for SG CLI and IDE recommendations.

The router is intentionally small: it turns retrieval metadata into a tier
recommendation without calling a model. IDE integrations surface this as advice;
future SG CLI execution can use the same decision to choose an actual provider.
"""

from __future__ import annotations

from dataclasses import dataclass

from .classifier import ModelTier, QueryMode


@dataclass(frozen=True)
class RoutingDecision:
    """Model-tier decision with enough detail for CLI/UI display and eval."""

    tier: ModelTier
    complexity_score: float
    reason: str


_MODE_COMPLEXITY = {
    QueryMode.RETRIEVAL_FAST: 0.10,
    QueryMode.DOCUMENT: 0.22,
    QueryMode.DEBUG_TARGETED: 0.35,
    QueryMode.TEST: 0.42,
    QueryMode.EXPLAIN: 0.45,
    QueryMode.BUILD_GUIDED: 0.52,
    QueryMode.REVIEW: 0.58,
    QueryMode.DEBUG_INVESTIGATE: 0.66,
    QueryMode.REFACTOR: 0.68,
    QueryMode.MIGRATE: 0.72,
    QueryMode.BUILD_GREENFIELD: 0.82,
    QueryMode.ARCHITECTURE: 0.88,
}


def route_model_tier(
    mode: QueryMode,
    base_tier: ModelTier,
    confidence: str,
    candidate_count: int,
    context_tokens: int,
    slm_used: bool = False,
) -> RoutingDecision:
    """Choose the cheapest likely-sufficient model tier.

    Inputs are all deterministic outputs of SG retrieval/assembly. This keeps
    routing auditable and cheap enough to run for every query.
    """

    confidence = (confidence or "MEDIUM").upper()
    score = _MODE_COMPLEXITY.get(mode, 0.50)
    reasons = [f"mode={mode.value}"]

    if confidence == "LOW":
        score += 0.10
        reasons.append("low confidence")
    elif confidence == "MISS":
        score = max(score - 0.15, 0.05)
        reasons.append("miss/pass-through")
    elif confidence == "HIGH":
        score -= 0.05
        reasons.append("high confidence")

    if candidate_count >= 25:
        score += 0.12
        reasons.append("wide candidate set")
    elif candidate_count <= 3:
        score -= 0.05
        reasons.append("narrow candidate set")

    if context_tokens >= 6000:
        score += 0.10
        reasons.append("large packet")
    elif context_tokens <= 900:
        score -= 0.08
        reasons.append("small packet")

    if slm_used:
        score += 0.03
        reasons.append("semantic fallback used")

    # Respect obviously hard modes even if the packet is short.
    if base_tier == ModelTier.LLM:
        score = max(score, 0.74)

    score = max(0.0, min(1.0, score))

    if confidence == "MISS":
        tier = ModelTier.MLM
    elif score < 0.28:
        tier = ModelTier.SLM
    elif score < 0.74:
        tier = ModelTier.MLM
    else:
        tier = ModelTier.LLM

    if base_tier == ModelTier.MLM and tier == ModelTier.SLM:
        tier = ModelTier.MLM
        reasons.append("mlm floor for code-changing mode")

    return RoutingDecision(
        tier=tier,
        complexity_score=round(score, 2),
        reason=", ".join(reasons),
    )
