"""
Dependency graph: adjacency lists, edge types, and traversal algorithms.

Sparse representation using forward (outgoing) and reverse (incoming) edge lists.
Supports blast-radius, dependency-chain, error-trace, and shortest-path queries
with zero LLM cost — pure data structure traversal.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, FrozenSet, List, Optional, Set, Tuple


class EdgeType(Enum):
    """Classification of dependency relationships between code entities."""

    CALLS = "calls"                  # function A calls function B
    IMPORTS = "imports"              # file A imports from file B
    INHERITS = "inherits"            # class A extends class B
    IMPLEMENTS = "implements"        # class A implements interface B
    TESTS = "tests"                  # test_X tests function X
    CONSTRUCTS = "constructs"        # function A creates instance of class B
    RAISES = "raises"                # function A raises exception B
    DECORATES = "decorates"          # decorator D wraps function F
    TYPE_DEPENDS = "type_depends"    # function A uses type B as param/return
    OVERRIDES = "overrides"          # method A overrides method B in parent class

    @property
    def is_strong(self) -> bool:
        """Strong edges represent direct functional dependency."""
        return self in _STRONG_EDGES

    @property
    def is_test_edge(self) -> bool:
        return self == EdgeType.TESTS


_STRONG_EDGES = frozenset({
    EdgeType.CALLS, EdgeType.INHERITS, EdgeType.IMPLEMENTS,
    EdgeType.CONSTRUCTS, EdgeType.OVERRIDES,
})


@dataclass(frozen=True, slots=True)
class DependencyEdge:
    """A directed edge in the dependency graph.

    source_fqn → target_fqn with a typed relationship.
    """

    source_fqn: str
    target_fqn: str
    edge_type: EdgeType

    source_line: int = 0
    """Line in source file where this relationship appears.
    For agent to jump to exact location. NOT sent to LLM context."""

    confidence: float = 1.0
    """1.0 for AST-extracted (definite), 0.7-0.9 for heuristic-inferred.
    Lower confidence edges are deprioritized in ranking."""

    def to_dict(self) -> dict:
        return {
            "source": self.source_fqn,
            "target": self.target_fqn,
            "type": self.edge_type.value,
            "line": self.source_line,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: dict) -> DependencyEdge:
        return cls(
            source_fqn=data["source"],
            target_fqn=data["target"],
            edge_type=EdgeType(data["type"]),
            source_line=data.get("line", 0),
            confidence=data.get("confidence", 1.0),
        )


class DependencyGraph:
    """Sparse adjacency list with forward and reverse edge maps.

    O(V + E) space. All traversal methods are zero-LLM-cost.
    """

    def __init__(self) -> None:
        # Core adjacency lists: fqn → list of edges
        self.forward: Dict[str, List[DependencyEdge]] = defaultdict(list)
        self.reverse: Dict[str, List[DependencyEdge]] = defaultdict(list)

        # All known nodes (including those with no edges)
        self._nodes: Set[str] = set()

    # ── Construction ───────────────────────────────────────────────────────

    def add_node(self, fqn: str) -> None:
        """Register a node (even if it has no edges yet)."""
        self._nodes.add(fqn)

    def add_edge(self, edge: DependencyEdge) -> None:
        """Add a directed edge. Automatically registers both endpoints as nodes."""
        self.forward[edge.source_fqn].append(edge)
        self.reverse[edge.target_fqn].append(edge)
        self._nodes.add(edge.source_fqn)
        self._nodes.add(edge.target_fqn)

    def add_edges(self, edges: List[DependencyEdge]) -> None:
        """Batch add edges."""
        for edge in edges:
            self.add_edge(edge)

    def remove_node(self, fqn: str) -> None:
        """Remove a node and all its edges. Used during incremental updates."""
        # Remove forward edges from this node
        if fqn in self.forward:
            for edge in self.forward[fqn]:
                if edge.target_fqn in self.reverse:
                    self.reverse[edge.target_fqn] = [
                        e for e in self.reverse[edge.target_fqn]
                        if e.source_fqn != fqn
                    ]
            del self.forward[fqn]

        # Remove reverse edges pointing to this node
        if fqn in self.reverse:
            for edge in self.reverse[fqn]:
                if edge.source_fqn in self.forward:
                    self.forward[edge.source_fqn] = [
                        e for e in self.forward[edge.source_fqn]
                        if e.target_fqn != fqn
                    ]
            del self.reverse[fqn]

        self._nodes.discard(fqn)

    # ── Properties ─────────────────────────────────────────────────────────

    @property
    def node_count(self) -> int:
        return len(self._nodes)

    @property
    def edge_count(self) -> int:
        return sum(len(edges) for edges in self.forward.values())

    @property
    def nodes(self) -> FrozenSet[str]:
        return frozenset(self._nodes)

    def has_node(self, fqn: str) -> bool:
        return fqn in self._nodes

    def get_forward_edges(
        self, fqn: str, edge_types: Optional[Set[EdgeType]] = None
    ) -> List[DependencyEdge]:
        """Get outgoing edges from a node, optionally filtered by type."""
        edges = self.forward.get(fqn, [])
        if edge_types is not None:
            edges = [e for e in edges if e.edge_type in edge_types]
        return edges

    def get_reverse_edges(
        self, fqn: str, edge_types: Optional[Set[EdgeType]] = None
    ) -> List[DependencyEdge]:
        """Get incoming edges to a node, optionally filtered by type."""
        edges = self.reverse.get(fqn, [])
        if edge_types is not None:
            edges = [e for e in edges if e.edge_type in edge_types]
        return edges

    # ── Traversal Algorithms ───────────────────────────────────────────────

    def blast_radius(
        self,
        fqn: str,
        max_depth: int = 2,
        edge_types: Optional[Set[EdgeType]] = None,
    ) -> Dict[str, int]:
        """BFS through REVERSE edges. Returns {fqn: distance} for all affected.

        'If I change this function, what else might break?'
        Traverses: who CALLS this, who INHERITS this, who TESTS this.

        Args:
            fqn: The function being changed.
            max_depth: Maximum hop distance to traverse.
            edge_types: Filter to specific edge types. None = all.

        Returns:
            Dict mapping affected FQN → distance from the changed function.
        """
        return self._bfs(fqn, max_depth, direction="reverse", edge_types=edge_types)

    def dependency_chain(
        self,
        fqn: str,
        max_depth: int = 3,
        edge_types: Optional[Set[EdgeType]] = None,
    ) -> Dict[str, int]:
        """BFS through FORWARD edges. Returns {fqn: distance} for all dependencies.

        'What does this function depend on? What do I need to understand it?'
        Traverses: what it CALLS, what it IMPORTS, what it INHERITS.

        Args:
            fqn: The function being examined.
            max_depth: Maximum hop distance.
            edge_types: Filter to specific edge types. None = all.

        Returns:
            Dict mapping dependency FQN → distance from the source function.
        """
        return self._bfs(fqn, max_depth, direction="forward", edge_types=edge_types)

    def error_trace(self, fqn: str, max_depth: int = 5) -> List[List[str]]:
        """Find all paths from this function to functions that RAISE exceptions.

        For debugging: 'Where could the error have originated?'
        Traverses forward CALLS edges, looking for RAISES edges.

        Returns:
            List of paths, each path is a list of FQNs from source to raiser.
        """
        paths: List[List[str]] = []
        call_edges = {EdgeType.CALLS}
        raise_targets: Set[str] = set()

        # First, find all nodes that raise
        for node_fqn in self._nodes:
            for edge in self.forward.get(node_fqn, []):
                if edge.edge_type == EdgeType.RAISES:
                    raise_targets.add(node_fqn)

        # DFS to find paths from fqn to any raiser
        def _dfs(current: str, path: List[str], visited: Set[str]) -> None:
            if len(path) > max_depth:
                return
            if current in raise_targets and current != fqn:
                paths.append(list(path))
                # Don't return — there might be deeper raisers
            for edge in self.forward.get(current, []):
                if edge.edge_type in call_edges and edge.target_fqn not in visited:
                    visited.add(edge.target_fqn)
                    path.append(edge.target_fqn)
                    _dfs(edge.target_fqn, path, visited)
                    path.pop()
                    visited.discard(edge.target_fqn)

        _dfs(fqn, [fqn], {fqn})
        return paths

    def test_coverage(self, fqn: str) -> List[str]:
        """Find all test functions that test this function.

        Reverse lookup through TESTS edges.

        Returns:
            List of test function FQNs.
        """
        test_fqns = []
        for edge in self.reverse.get(fqn, []):
            if edge.edge_type == EdgeType.TESTS:
                test_fqns.append(edge.source_fqn)
        return test_fqns

    def shortest_path(
        self,
        source: str,
        target: str,
        max_depth: int = 10,
    ) -> Optional[List[str]]:
        """BFS shortest path between any two nodes (bidirectional edges).

        Returns None if no path exists within max_depth.
        """
        if source == target:
            return [source]
        if source not in self._nodes or target not in self._nodes:
            return None

        # BFS with parent tracking
        visited: Dict[str, Optional[str]] = {source: None}
        queue: deque[Tuple[str, int]] = deque([(source, 0)])

        while queue:
            current, depth = queue.popleft()
            if depth >= max_depth:
                continue

            # Check both forward and reverse neighbors
            neighbors: Set[str] = set()
            for edge in self.forward.get(current, []):
                neighbors.add(edge.target_fqn)
            for edge in self.reverse.get(current, []):
                neighbors.add(edge.source_fqn)

            for neighbor in neighbors:
                if neighbor not in visited:
                    visited[neighbor] = current
                    if neighbor == target:
                        # Reconstruct path
                        path = [target]
                        node = target
                        while visited[node] is not None:
                            node = visited[node]  # type: ignore[assignment]
                            path.append(node)
                        return list(reversed(path))
                    queue.append((neighbor, depth + 1))

        return None

    def subgraph(self, fqns: Set[str]) -> DependencyGraph:
        """Extract a subgraph containing only the specified nodes and edges between them."""
        sub = DependencyGraph()
        for fqn in fqns:
            if fqn in self._nodes:
                sub.add_node(fqn)
        for fqn in fqns:
            for edge in self.forward.get(fqn, []):
                if edge.target_fqn in fqns:
                    sub.add_edge(edge)
        return sub

    def edge_summary(self, fqns: Set[str], max_edges: int = 20) -> str:
        """Generate a compact text summary of edges between the given FQNs.

        Used in Zone 3 to show structural relationships.
        e.g., 'validate_token calls decode_jwt, AuthMiddleware inherits BaseMiddleware'
        """
        lines = []
        seen = set()
        for fqn in fqns:
            for edge in self.forward.get(fqn, []):
                if edge.target_fqn in fqns:
                    key = (edge.source_fqn, edge.target_fqn, edge.edge_type)
                    if key not in seen:
                        seen.add(key)
                        src = edge.source_fqn.split("::")[-1]
                        tgt = edge.target_fqn.split("::")[-1]
                        lines.append(f"{src} {edge.edge_type.value} {tgt}")
                        if len(lines) >= max_edges:
                            break
            if len(lines) >= max_edges:
                break
        return "; ".join(lines)

    # ── Private ────────────────────────────────────────────────────────────

    def _bfs(
        self,
        start: str,
        max_depth: int,
        direction: str,
        edge_types: Optional[Set[EdgeType]] = None,
    ) -> Dict[str, int]:
        """Generic BFS through forward or reverse edges.

        Returns {fqn: distance} for all reachable nodes within max_depth.
        The start node is NOT included in results.
        """
        if start not in self._nodes:
            return {}

        edge_map = self.forward if direction == "forward" else self.reverse
        result: Dict[str, int] = {}
        visited: Set[str] = {start}
        queue: deque[Tuple[str, int]] = deque([(start, 0)])

        while queue:
            current, depth = queue.popleft()
            if depth >= max_depth:
                continue

            for edge in edge_map.get(current, []):
                neighbor = (
                    edge.target_fqn if direction == "forward" else edge.source_fqn
                )
                if edge_types is not None and edge.edge_type not in edge_types:
                    continue
                if neighbor not in visited:
                    visited.add(neighbor)
                    result[neighbor] = depth + 1
                    queue.append((neighbor, depth + 1))

        return result

    # ── Serialization ──────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """Serialize the graph to a JSON-compatible dict."""
        all_edges = []
        for edges in self.forward.values():
            all_edges.extend(e.to_dict() for e in edges)
        return {
            "nodes": sorted(self._nodes),
            "edges": all_edges,
        }

    @classmethod
    def from_dict(cls, data: dict) -> DependencyGraph:
        """Deserialize from JSON dict."""
        graph = cls()
        for node in data.get("nodes", []):
            graph.add_node(node)
        for edge_data in data.get("edges", []):
            graph.add_edge(DependencyEdge.from_dict(edge_data))
        return graph
