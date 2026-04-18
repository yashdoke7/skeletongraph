"""
.skeletongraph/ directory management and persistence.

Handles reading/writing all index files with atomic writes
(write to temp, then rename) to prevent corruption.

Directory structure:
    .skeletongraph/
    ├── meta.json         # Build metadata, version, timestamp
    ├── skeletons.json    # SkeletonCore entries (no summaries)
    ├── summaries.json    # FQN → summary (separate layer)
    ├── graph.json        # DependencyGraph edges
    ├── index.json        # Inverted index
    ├── bloom.bin         # Bloom filter binary
    ├── hashes.json       # SHA256 dirty tracking cache
    └── session/          # Per-user session state (gitignored)
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from ..graph.bloom import BloomFilter
from ..graph.dependency import DependencyGraph
from ..graph.inverted_index import InvertedIndex
from ..parser.skeleton import FileSkeleton, SkeletonCore
from ..storage.dirty import DirtyTracker
from ..summary.summary_store import SummaryStore


SKELETONGRAPH_DIR = ".skeletongraph"
VERSION = "0.1.0"


@dataclass
class BuildMeta:
    """Build metadata stored in meta.json."""
    version: str = VERSION
    build_timestamp: float = 0.0
    total_files: int = 0
    total_functions: int = 0
    total_edges: int = 0
    languages: List[str] = field(default_factory=list)
    build_duration_seconds: float = 0.0

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "build_timestamp": self.build_timestamp,
            "total_files": self.total_files,
            "total_functions": self.total_functions,
            "total_edges": self.total_edges,
            "languages": self.languages,
            "build_duration_seconds": self.build_duration_seconds,
        }

    @classmethod
    def from_dict(cls, data: dict) -> BuildMeta:
        return cls(
            version=data.get("version", VERSION),
            build_timestamp=data.get("build_timestamp", 0.0),
            total_files=data.get("total_files", 0),
            total_functions=data.get("total_functions", 0),
            total_edges=data.get("total_edges", 0),
            languages=data.get("languages", []),
            build_duration_seconds=data.get("build_duration_seconds", 0.0),
        )


@dataclass
class IndexStore:
    """Complete in-memory representation of the .skeletongraph/ index.

    Load once, query many times. Persist after builds/updates.
    """

    meta: BuildMeta
    file_skeletons: Dict[str, FileSkeleton]    # file_path → FileSkeleton
    skeleton_table: Dict[str, SkeletonCore]    # fqn → SkeletonCore
    graph: DependencyGraph
    summaries: SummaryStore
    inverted_index: InvertedIndex
    bloom: BloomFilter
    dirty_tracker: DirtyTracker

    @property
    def function_count(self) -> int:
        return len(self.skeleton_table)

    @property
    def file_count(self) -> int:
        return len(self.file_skeletons)

    def get_skeleton(self, fqn: str) -> Optional[SkeletonCore]:
        return self.skeleton_table.get(fqn)

    def get_file_skeleton(self, file_path: str) -> Optional[FileSkeleton]:
        return self.file_skeletons.get(file_path)

    def fqn_exists(self, fqn: str) -> bool:
        """Fast existence check via bloom filter, confirmed against table."""
        if fqn not in self.bloom:
            return False
        return fqn in self.skeleton_table

    def search(self, query: str, top_k: int = 10) -> List[str]:
        """Search for functions matching a query. Returns ranked FQN list."""
        results = self.inverted_index.search(query, top_k=top_k)
        return [fqn for fqn, _score in results]

    def status_summary(self) -> str:
        """Human-readable status string for CLI output."""
        age = ""
        if self.meta.build_timestamp > 0:
            elapsed = time.time() - self.meta.build_timestamp
            if elapsed < 60:
                age = f"{elapsed:.0f}s ago"
            elif elapsed < 3600:
                age = f"{elapsed / 60:.0f}m ago"
            else:
                age = f"{elapsed / 3600:.1f}h ago"

        langs = ", ".join(self.meta.languages) if self.meta.languages else "none"
        return (
            f"{self.function_count} functions, "
            f"{self.meta.total_edges} edges, "
            f"{self.file_count} files indexed. "
            f"Languages: {langs}. "
            f"Last built: {age or 'never'}."
        )


def _atomic_write_json(path: Path, data: dict) -> None:
    """Write JSON atomically: write to .tmp, then rename."""
    tmp_path = path.with_suffix(".tmp")
    tmp_path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    tmp_path.replace(path)


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write bytes atomically."""
    tmp_path = path.with_suffix(".tmp")
    tmp_path.write_bytes(data)
    tmp_path.replace(path)


def save_index(store: IndexStore, project_root: Path) -> None:
    """Persist the full index to .skeletongraph/ directory."""
    sg_dir = project_root / SKELETONGRAPH_DIR
    sg_dir.mkdir(exist_ok=True)

    # Ensure session directory exists (gitignored)
    (sg_dir / "session").mkdir(exist_ok=True)

    # meta.json
    _atomic_write_json(sg_dir / "meta.json", store.meta.to_dict())

    # skeletons.json (FileSkeleton list, includes SkeletonCore inside)
    skeletons_data = {
        path: fs.to_dict() for path, fs in store.file_skeletons.items()
    }
    _atomic_write_json(sg_dir / "skeletons.json", skeletons_data)

    # graph.json
    _atomic_write_json(sg_dir / "graph.json", store.graph.to_dict())

    # summaries.json
    store.summaries.save(sg_dir)

    # index.json (inverted index)
    _atomic_write_json(sg_dir / "index.json", store.inverted_index.to_dict())

    # bloom.bin
    _atomic_write_bytes(sg_dir / "bloom.bin", store.bloom.to_bytes())

    # hashes.json
    store.dirty_tracker.save(sg_dir)


def load_index(project_root: Path) -> Optional[IndexStore]:
    """Load the full index from .skeletongraph/ directory.

    Returns None if no index exists.
    """
    sg_dir = project_root / SKELETONGRAPH_DIR
    if not sg_dir.exists():
        return None

    meta_path = sg_dir / "meta.json"
    if not meta_path.exists():
        return None

    # Load meta
    meta = BuildMeta.from_dict(
        json.loads(meta_path.read_text(encoding="utf-8"))
    )

    # Load file skeletons
    file_skeletons: Dict[str, FileSkeleton] = {}
    skeleton_table: Dict[str, SkeletonCore] = {}
    skel_path = sg_dir / "skeletons.json"
    if skel_path.exists():
        skel_data = json.loads(skel_path.read_text(encoding="utf-8"))
        for path, fs_dict in skel_data.items():
            fs = FileSkeleton.from_dict(fs_dict)
            file_skeletons[path] = fs
            for sk in fs.all_skeletons:
                skeleton_table[sk.fqn] = sk

    # Load graph
    graph = DependencyGraph()
    graph_path = sg_dir / "graph.json"
    if graph_path.exists():
        graph = DependencyGraph.from_dict(
            json.loads(graph_path.read_text(encoding="utf-8"))
        )

    # Load summaries
    summaries = SummaryStore.load(sg_dir)

    # Load inverted index
    inverted_index = InvertedIndex()
    idx_path = sg_dir / "index.json"
    if idx_path.exists():
        inverted_index = InvertedIndex.from_dict(
            json.loads(idx_path.read_text(encoding="utf-8"))
        )

    # Load bloom filter
    bloom = BloomFilter(expected_items=max(len(skeleton_table), 100))
    bloom_path = sg_dir / "bloom.bin"
    if bloom_path.exists():
        bloom = BloomFilter.from_bytes(bloom_path.read_bytes())

    # Load dirty tracker
    dirty_tracker = DirtyTracker.load(sg_dir)

    return IndexStore(
        meta=meta,
        file_skeletons=file_skeletons,
        skeleton_table=skeleton_table,
        graph=graph,
        summaries=summaries,
        inverted_index=inverted_index,
        bloom=bloom,
        dirty_tracker=dirty_tracker,
    )


def create_empty_index() -> IndexStore:
    """Create a fresh, empty index store."""
    return IndexStore(
        meta=BuildMeta(),
        file_skeletons={},
        skeleton_table={},
        graph=DependencyGraph(),
        summaries=SummaryStore(),
        inverted_index=InvertedIndex(),
        bloom=BloomFilter(),
        dirty_tracker=DirtyTracker(),
    )
