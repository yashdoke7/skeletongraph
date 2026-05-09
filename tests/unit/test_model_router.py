from skeletongraph.retrieval.classifier import ModelTier, QueryMode
from skeletongraph.retrieval.model_router import route_model_tier


def test_routes_small_navigation_to_slm():
    decision = route_model_tier(
        mode=QueryMode.RETRIEVAL_FAST,
        base_tier=ModelTier.SLM,
        confidence="HIGH",
        candidate_count=1,
        context_tokens=300,
    )

    assert decision.tier == ModelTier.SLM
    assert decision.complexity_score < 0.28


def test_routes_targeted_fix_to_mlm():
    decision = route_model_tier(
        mode=QueryMode.DEBUG_TARGETED,
        base_tier=ModelTier.MLM,
        confidence="HIGH",
        candidate_count=4,
        context_tokens=1800,
    )

    assert decision.tier == ModelTier.MLM


def test_keeps_architecture_on_llm():
    decision = route_model_tier(
        mode=QueryMode.ARCHITECTURE,
        base_tier=ModelTier.LLM,
        confidence="HIGH",
        candidate_count=5,
        context_tokens=1400,
    )

    assert decision.tier == ModelTier.LLM
    assert decision.complexity_score >= 0.74


def test_wide_large_investigation_routes_to_llm():
    decision = route_model_tier(
        mode=QueryMode.DEBUG_INVESTIGATE,
        base_tier=ModelTier.MLM,
        confidence="LOW",
        candidate_count=30,
        context_tokens=7000,
        slm_used=True,
    )

    assert decision.tier == ModelTier.LLM
