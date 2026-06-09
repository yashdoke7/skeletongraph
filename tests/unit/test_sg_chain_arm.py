"""Unit coverage for the path-aware sg-chain eval arm."""

from __future__ import annotations

from collections import Counter

from eval.agent.config import ARMS, STAGES
from eval.agent.tools import (
    _SG_BACKENDS,
    _diverse_top_k,
    _path_bridge_counts,
    _score_chain_candidates,
)
from skeletongraph.graph.dependency import (
    DependencyEdge,
    DependencyGraph,
    EdgeType,
)


def test_sg_chain_is_registered_as_trial_arm():
    assert "sg-chain" in ARMS
    assert "sg-chain" in _SG_BACKENDS
    assert "sg-chain" in STAGES["trial"].arms


def test_path_bridge_counts_short_graph_connectors():
    graph = DependencyGraph()
    graph.add_edge(DependencyEdge("a.py::source", "b.py::bridge", EdgeType.CALLS))
    graph.add_edge(DependencyEdge("b.py::bridge", "c.py::target", EdgeType.CALLS))

    counts = _path_bridge_counts(
        graph,
        ["a.py::source"],
        ["c.py::target"],
        {"a.py::source", "b.py::bridge", "c.py::target"},
    )

    assert counts["a.py::source"] == 1
    assert counts["b.py::bridge"] == 1
    assert counts["c.py::target"] == 1


def test_chain_scorer_rewards_consensus_and_graph_paths():
    ranked, reasons = _score_chain_candidates(
        ["pkg/a.py::target", "pkg/x.py::helper"],
        ["pkg/b.py::lexical", "pkg/a.py::target"],
        Counter({"pkg/bridge.py::join": 2, "pkg/a.py::target": 1}),
    )

    assert ranked[0] == "pkg/a.py::target"
    assert "consensus" in reasons["pkg/a.py::target"]
    assert "graph-path" in reasons["pkg/a.py::target"]
    assert "graph-path" in reasons["pkg/bridge.py::join"]


def test_diverse_top_k_caps_repeated_files():
    ranked = [
        "pkg/a.py::one",
        "pkg/a.py::two",
        "pkg/a.py::three",
        "pkg/b.py::one",
    ]

    assert _diverse_top_k(ranked, k=4, per_file=2) == [
        "pkg/a.py::one",
        "pkg/a.py::two",
        "pkg/b.py::one",
    ]
