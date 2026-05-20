"""Tests for SG ablation toggles.

Verifies that each ablation flag changes the pipeline's behaviour in a
measurable, observable way. Tests are unit-level — they don't need a full
built index on disk.

Covered:
  1. enable_centrality_rerank=False → hub score excluded from Ranker.score()
  2. enable_graph_expansion=False   → graph expansion skipped in resolver
  3. SGConfig defaults + round-trip  → all three flags present and True
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from skeletongraph.config import SGConfig, save_config, load_config
from skeletongraph.retrieval.ranker import Ranker, RankWeights
from skeletongraph.graph.dependency import DependencyGraph


# ── helpers ──────────────────────────────────────────────────────────────────


def _dummy_skeleton(fqn: str, file_path: str = "a.py"):
    """Minimal SkeletonCore-like object for ranker tests."""
    sk = MagicMock()
    sk.fqn = fqn
    sk.file_path = file_path
    sk.complexity = 5
    sk.is_exported = True
    sk.signature = f"def {fqn.split('::')[-1]}(): ..."  # real string, not MagicMock
    sk.docstring = ""
    sk.sha256 = "deadbeef"
    sk.kind.auto_include_constructor = False
    return sk


# ── 1. Centrality rerank flag ─────────────────────────────────────────────────


class TestCentralityRerank:
    """Ranker with centrality disabled scores differently from the full ranker."""

    def setup_method(self):
        self.graph = DependencyGraph()
        self.sk = _dummy_skeleton("mod.py::my_func")

    def _score_with_hub(self, enabled: bool, hub_value: float) -> float:
        ranker = Ranker(self.graph, centrality_enabled=enabled)
        # Manually inject a hub score so we don't need a full graph
        ranker._hub_scores["mod.py::my_func"] = hub_value
        return ranker.score("mod.py::my_func", self.sk, 0, "Direct target")

    def test_hub_score_included_when_enabled(self):
        """Full ranker adds hub signal to the score."""
        score_no_hub = self._score_with_hub(enabled=True, hub_value=0.0)
        score_high_hub = self._score_with_hub(enabled=True, hub_value=1.0)
        w = RankWeights()
        assert score_high_hub == pytest.approx(score_no_hub + w.connectivity)

    def test_hub_score_excluded_when_disabled(self):
        """sg-norerank ablation: hub signal ignored regardless of hub value."""
        score_low = self._score_with_hub(enabled=False, hub_value=0.0)
        score_high = self._score_with_hub(enabled=False, hub_value=1.0)
        assert score_low == pytest.approx(score_high), (
            "With centrality disabled, different hub values must produce same score"
        )

    def test_disabled_score_equals_enabled_zero_hub(self):
        """With hub=0 and enabled=True the score matches disabled (correct zero-hub boundary)."""
        score_enabled_zero = self._score_with_hub(enabled=True, hub_value=0.0)
        score_disabled = self._score_with_hub(enabled=False, hub_value=0.5)
        assert score_enabled_zero == pytest.approx(score_disabled), (
            "Disabled centrality with any hub == enabled centrality with hub=0"
        )


# ── 2. Graph expansion flag ────────────────────────────────────────────────────


class TestGraphExpansion:
    """resolve_context with enable_graph_expansion=False skips traversal calls."""

    def test_blast_radius_not_called_when_disabled(self):
        """When enable_graph_expansion=False, store.graph.blast_radius is never called."""
        from skeletongraph.retrieval.resolver import resolve_context

        # Build a minimal mock IndexStore
        store = MagicMock()
        store.file_skeletons = {}
        store.skeleton_table = {}
        store.inverted_index.entry_count = 0
        store.inverted_index.search.return_value = []
        store.graph = MagicMock()
        store.graph.reverse = {}
        store.bm25_model = None
        store.embeddings = None
        store.summaries = MagicMock()
        store.summaries.get.return_value = None

        result = resolve_context(
            prompt="fix the validation bug",
            store=store,
            enable_graph_expansion=False,
        )

        store.graph.blast_radius.assert_not_called()
        store.graph.dependency_chain.assert_not_called()
        assert result.candidates == []

    def test_blast_radius_called_when_enabled(self):
        """When enable_graph_expansion=True and there are targets, traversal is attempted."""
        from skeletongraph.retrieval.resolver import resolve_context

        store = MagicMock()
        store.file_skeletons = {}
        # Seed one skeleton so there's a direct hit
        sk = _dummy_skeleton("a.py::validate")
        sk.sha256 = "abc"
        store.skeleton_table = {"a.py::validate": sk}
        store.inverted_index.entry_count = 0
        store.inverted_index.search.return_value = []
        store.graph = MagicMock()
        store.graph.reverse = {}
        store.graph.blast_radius.return_value = {}
        store.graph.dependency_chain.return_value = {}
        store.graph.test_coverage.return_value = []
        store.bm25_model = None
        store.embeddings = None
        store.summaries = MagicMock()
        store.summaries.get.return_value = None

        # Patch intent to match the skeleton's short name
        from skeletongraph.retrieval.intent import Intent, Entity, TaskType
        mock_intent = Intent(
            entities=[Entity(value="a.py::validate", entity_type="function", confidence=1.0)],
            function_names=["validate"],
            file_paths=[],
            task_type=TaskType.EDIT,
            raw_prompt="fix validate",
        )
        with patch(
            "skeletongraph.retrieval.resolver.analyze_intent",
            return_value=mock_intent,
        ):
            resolve_context(
                prompt="fix validate",
                store=store,
                enable_graph_expansion=True,
            )

        # blast_radius should have been called for the direct target
        store.graph.blast_radius.assert_called()


# ── 3. SGConfig defaults and round-trip ────────────────────────────────────────


class TestSGConfigAblationFlags:
    """All three ablation flags must exist on SGConfig with True defaults."""

    def test_defaults_are_true(self):
        cfg = SGConfig()
        assert cfg.enable_graph_expansion is True
        assert cfg.enable_centrality_rerank is True
        assert cfg.enable_summaries is True

    def test_can_disable_individually(self):
        cfg = SGConfig()
        cfg.enable_graph_expansion = False
        assert cfg.enable_graph_expansion is False
        assert cfg.enable_centrality_rerank is True  # unchanged

        cfg2 = SGConfig()
        cfg2.enable_centrality_rerank = False
        assert cfg2.enable_centrality_rerank is False
        assert cfg2.enable_graph_expansion is True  # unchanged

    def test_round_trip_via_save_load(self, tmp_path):
        """Flags survive save_config → load_config."""
        cfg = SGConfig()
        cfg.enable_graph_expansion = False
        cfg.enable_centrality_rerank = False
        cfg.enable_summaries = False
        save_config(cfg, tmp_path)

        loaded = load_config(tmp_path)
        assert loaded.enable_graph_expansion is False
        assert loaded.enable_centrality_rerank is False
        assert loaded.enable_summaries is False

    def test_round_trip_defaults_preserved(self, tmp_path):
        """Default True values are preserved through save/load."""
        cfg = SGConfig()
        save_config(cfg, tmp_path)
        loaded = load_config(tmp_path)
        assert loaded.enable_graph_expansion is True
        assert loaded.enable_centrality_rerank is True
        assert loaded.enable_summaries is True
