"""Tests for DependencyGraph traversal algorithms."""

import pytest

from skeletongraph.graph.dependency import (
    DependencyEdge,
    DependencyGraph,
    EdgeType,
)


@pytest.fixture
def auth_graph():
    """Build a realistic auth-module dependency graph.

    Graph structure:
        validate_token → decode_jwt → TokenError (RAISES)
        validate_token → get_user
        authenticate → validate_token
        authenticate → check_permissions
        auth_required (decorator) → authenticate
        test_validate → validate_token (TESTS)
        test_auth → authenticate (TESTS)
        AuthMiddleware → BaseMiddleware (INHERITS)
    """
    g = DependencyGraph()

    edges = [
        DependencyEdge("m.py::validate_token", "m.py::decode_jwt", EdgeType.CALLS, 48),
        DependencyEdge("m.py::decode_jwt", "m.py::TokenError", EdgeType.RAISES, 30),
        DependencyEdge("m.py::validate_token", "m.py::get_user", EdgeType.CALLS, 52),
        DependencyEdge("m.py::authenticate", "m.py::validate_token", EdgeType.CALLS, 65),
        DependencyEdge("m.py::authenticate", "m.py::check_permissions", EdgeType.CALLS, 68),
        DependencyEdge("m.py::auth_required", "m.py::authenticate", EdgeType.DECORATES, 80),
        DependencyEdge("t.py::test_validate", "m.py::validate_token", EdgeType.TESTS, 10),
        DependencyEdge("t.py::test_auth", "m.py::authenticate", EdgeType.TESTS, 20),
        DependencyEdge("m.py::AuthMiddleware", "base.py::BaseMiddleware", EdgeType.INHERITS, 5),
    ]
    g.add_edges(edges)
    return g


class TestGraphConstruction:
    def test_node_count(self, auth_graph):
        # 8 source FQNs + 3 target-only FQNs (TokenError, get_user, BaseMiddleware)
        # but get_user also appears as target of validate_token, not as source
        # Actually: validate_token, decode_jwt, get_user, authenticate,
        # check_permissions, auth_required, test_validate, test_auth,
        # AuthMiddleware, TokenError, BaseMiddleware = 11
        assert auth_graph.node_count == 11

    def test_edge_count(self, auth_graph):
        assert auth_graph.edge_count == 9

    def test_has_node(self, auth_graph):
        assert auth_graph.has_node("m.py::validate_token")
        assert not auth_graph.has_node("m.py::nonexistent")

    def test_forward_edges(self, auth_graph):
        edges = auth_graph.get_forward_edges("m.py::validate_token")
        targets = {e.target_fqn for e in edges}
        assert targets == {"m.py::decode_jwt", "m.py::get_user"}

    def test_reverse_edges(self, auth_graph):
        edges = auth_graph.get_reverse_edges("m.py::validate_token")
        sources = {e.source_fqn for e in edges}
        assert "m.py::authenticate" in sources
        assert "t.py::test_validate" in sources

    def test_filtered_edges(self, auth_graph):
        edges = auth_graph.get_reverse_edges(
            "m.py::validate_token",
            edge_types={EdgeType.CALLS}
        )
        assert len(edges) == 1
        assert edges[0].source_fqn == "m.py::authenticate"


class TestBlastRadius:
    def test_blast_radius_depth1(self, auth_graph):
        affected = auth_graph.blast_radius("m.py::validate_token", max_depth=1)
        assert "m.py::authenticate" in affected
        assert "t.py::test_validate" in affected
        assert affected["m.py::authenticate"] == 1

    def test_blast_radius_depth2(self, auth_graph):
        affected = auth_graph.blast_radius("m.py::validate_token", max_depth=2)
        assert "m.py::authenticate" in affected
        assert "m.py::auth_required" in affected
        assert "t.py::test_auth" in affected
        assert affected["m.py::auth_required"] == 2

    def test_blast_radius_leaf_node(self, auth_graph):
        # decode_jwt has no reverse CALLS edges (only a RAISES from itself)
        affected = auth_graph.blast_radius("m.py::decode_jwt", max_depth=2)
        assert "m.py::validate_token" in affected

    def test_blast_radius_nonexistent(self, auth_graph):
        affected = auth_graph.blast_radius("m.py::nonexistent")
        assert affected == {}

    def test_blast_radius_filtered(self, auth_graph):
        # Only follow CALLS edges (not TESTS)
        affected = auth_graph.blast_radius(
            "m.py::validate_token",
            max_depth=2,
            edge_types={EdgeType.CALLS},
        )
        assert "m.py::authenticate" in affected
        assert "t.py::test_validate" not in affected


class TestDependencyChain:
    def test_dependency_chain_depth1(self, auth_graph):
        deps = auth_graph.dependency_chain("m.py::authenticate", max_depth=1)
        assert "m.py::validate_token" in deps
        assert "m.py::check_permissions" in deps
        assert deps["m.py::validate_token"] == 1

    def test_dependency_chain_depth2(self, auth_graph):
        deps = auth_graph.dependency_chain("m.py::authenticate", max_depth=2)
        assert "m.py::decode_jwt" in deps
        assert "m.py::get_user" in deps
        assert deps["m.py::decode_jwt"] == 2


class TestErrorTrace:
    def test_finds_raiser(self, auth_graph):
        paths = auth_graph.error_trace("m.py::validate_token")
        # validate_token → decode_jwt (which RAISES TokenError)
        assert len(paths) >= 1
        found_path = False
        for path in paths:
            if "m.py::decode_jwt" in path:
                found_path = True
        assert found_path

    def test_no_errors(self, auth_graph):
        paths = auth_graph.error_trace("m.py::check_permissions")
        assert paths == []


class TestTestCoverage:
    def test_finds_tests(self, auth_graph):
        tests = auth_graph.test_coverage("m.py::validate_token")
        assert "t.py::test_validate" in tests

    def test_no_tests(self, auth_graph):
        tests = auth_graph.test_coverage("m.py::decode_jwt")
        assert tests == []


class TestShortestPath:
    def test_direct_path(self, auth_graph):
        path = auth_graph.shortest_path("m.py::authenticate", "m.py::validate_token")
        assert path is not None
        assert path[0] == "m.py::authenticate"
        assert path[-1] == "m.py::validate_token"
        assert len(path) == 2

    def test_multi_hop_path(self, auth_graph):
        path = auth_graph.shortest_path("m.py::authenticate", "m.py::decode_jwt")
        assert path is not None
        assert len(path) == 3

    def test_no_path(self, auth_graph):
        g = DependencyGraph()
        g.add_node("a")
        g.add_node("b")
        assert g.shortest_path("a", "b") is None

    def test_self_path(self, auth_graph):
        path = auth_graph.shortest_path("m.py::authenticate", "m.py::authenticate")
        assert path == ["m.py::authenticate"]


class TestRemoveNode:
    def test_remove_cleans_edges(self, auth_graph):
        auth_graph.remove_node("m.py::validate_token")
        assert not auth_graph.has_node("m.py::validate_token")

        # Forward edges from authenticate should no longer reference validate_token
        edges = auth_graph.get_forward_edges("m.py::authenticate")
        targets = {e.target_fqn for e in edges}
        assert "m.py::validate_token" not in targets


class TestSubgraph:
    def test_subgraph_extraction(self, auth_graph):
        fqns = {"m.py::authenticate", "m.py::validate_token", "m.py::decode_jwt"}
        sub = auth_graph.subgraph(fqns)

        assert sub.node_count == 3
        # authenticate → validate_token, validate_token → decode_jwt
        assert sub.edge_count == 2


class TestEdgeSummary:
    def test_edge_summary(self, auth_graph):
        fqns = {"m.py::authenticate", "m.py::validate_token", "m.py::check_permissions"}
        summary = auth_graph.edge_summary(fqns)
        assert "calls" in summary
        assert "validate_token" in summary


class TestSerialization:
    def test_round_trip(self, auth_graph):
        data = auth_graph.to_dict()
        restored = DependencyGraph.from_dict(data)

        assert restored.node_count == auth_graph.node_count
        assert restored.edge_count == auth_graph.edge_count

        # Verify traversal still works
        affected = restored.blast_radius("m.py::validate_token", max_depth=1)
        assert "m.py::authenticate" in affected
