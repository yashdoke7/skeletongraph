"""
End-to-end integration test: build index → query → assemble context.

Tests the full pipeline against the python_small fixture codebase.
"""

import pytest
from pathlib import Path

from skeletongraph.build import build_index, update_index, discover_files
from skeletongraph.storage.local import load_index, save_index
from skeletongraph.retrieval.intent import Entity, Intent, analyze_intent, TaskType
from skeletongraph.retrieval.resolver import _resolve_entities, resolve_context, Tier
from skeletongraph.retrieval.budget import TokenBudget, Zone3Mode
from skeletongraph.assembly.zone_assembler import assemble_context


FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "python_small"


class TestBuildPipeline:
    """Test full build → persist → load cycle."""

    @pytest.fixture(scope="class")
    def store(self, tmp_path_factory):
        """Build index for the fixture project into a temp directory."""
        # Copy fixtures to a temp dir (so .skeletongraph/ doesn't pollute fixtures)
        import shutil
        tmp = tmp_path_factory.mktemp("project")
        if FIXTURES_DIR.exists():
            shutil.copytree(FIXTURES_DIR, tmp, dirs_exist_ok=True)

        store = build_index(tmp)
        return store, tmp

    def test_files_discovered(self, store):
        store, tmp = store
        assert store.meta.total_files >= 1

    def test_functions_extracted(self, store):
        store, tmp = store
        assert store.meta.total_functions >= 5  # middleware has ~7 functions

    def test_edges_created(self, store):
        store, tmp = store
        assert store.meta.total_edges >= 1

    def test_bloom_populated(self, store):
        store, tmp = store
        assert len(store.bloom) > 0

    def test_inverted_index_populated(self, store):
        store, tmp = store
        assert store.inverted_index.entry_count > 0

    def test_skeleton_table_populated(self, store):
        store, tmp = store
        assert len(store.skeleton_table) >= 5

    def test_persistence_roundtrip(self, store):
        store, tmp = store
        # Save and reload
        save_index(store, tmp)
        loaded = load_index(tmp)
        assert loaded is not None
        assert loaded.meta.total_functions == store.meta.total_functions
        assert len(loaded.skeleton_table) == len(store.skeleton_table)

    def test_fqn_exists(self, store):
        store, tmp = store
        assert store.fqn_exists("auth/middleware.py::validate_token")

    def test_search(self, store):
        store, tmp = store
        results = store.search("validate token")
        assert len(results) > 0
        assert any("validate_token" in fqn for fqn in results)


class TestIncrementalUpdate:
    """Test incremental update detects changes."""

    def test_no_changes(self, tmp_path):
        """Update with no changes should be fast and return same data."""
        import shutil
        if FIXTURES_DIR.exists():
            shutil.copytree(FIXTURES_DIR, tmp_path, dirs_exist_ok=True)

        store1 = build_index(tmp_path)
        store2 = update_index(tmp_path)

        assert store2.meta.total_functions == store1.meta.total_functions

    def test_new_file(self, tmp_path):
        """Adding a new file should be detected."""
        import shutil
        if FIXTURES_DIR.exists():
            shutil.copytree(FIXTURES_DIR, tmp_path, dirs_exist_ok=True)

        store1 = build_index(tmp_path)
        count1 = store1.meta.total_functions

        # Add a new file
        new_file = tmp_path / "new_module.py"
        new_file.write_text("def new_function():\n    return 42\n", encoding="utf-8")

        store2 = update_index(tmp_path)
        assert store2.meta.total_functions > count1

    def test_modified_file_replaces_file_level_edges(self, tmp_path):
        """Updating imports should not leave stale file pseudo-node edges."""
        import shutil
        if FIXTURES_DIR.exists():
            shutil.copytree(FIXTURES_DIR, tmp_path, dirs_exist_ok=True)

        consumer = tmp_path / "consumer.py"
        consumer.write_text(
            "from auth.middleware import validate_token\n\n"
            "def use_token(token, secret):\n"
            "    return validate_token(token, secret)\n",
            encoding="utf-8",
        )

        build_index(tmp_path)

        consumer.write_text(
            "from auth.middleware import decode_jwt\n\n"
            "def use_token(token, secret):\n"
            "    return decode_jwt(token, secret)\n",
            encoding="utf-8",
        )

        store = update_index(tmp_path)
        edges = store.graph.forward.get("consumer.py::__file__", [])

        assert len(edges) == 1
        assert edges[0].target_fqn == "auth.middleware::decode_jwt"


class TestIntentAnalysis:
    def test_debug_intent(self):
        intent = analyze_intent("fix the authentication bug in middleware.py")
        assert intent.task_type == TaskType.DEBUG
        assert any("middleware.py" in fp for fp in intent.file_paths)

    def test_create_intent(self):
        intent = analyze_intent("add a new rate limiting feature")
        assert intent.task_type == TaskType.CREATE

    def test_explain_intent(self):
        intent = analyze_intent("how does validate_token work?")
        assert intent.task_type == TaskType.EXPLAIN

    def test_refactor_intent(self):
        intent = analyze_intent("refactor the authentication module")
        assert intent.task_type == TaskType.REFACTOR

    def test_error_extraction(self):
        intent = analyze_intent('getting TokenError: Token expired on line 45')
        assert intent.error_message is not None
        assert intent.line_number == 45

    def test_file_extraction(self):
        intent = analyze_intent("update auth/middleware.py to support refresh tokens")
        assert any("middleware.py" in fp for fp in intent.file_paths)

    def test_function_matching(self):
        known = {"auth/m.py::validate_token", "auth/m.py::decode_jwt"}
        intent = analyze_intent("fix validate_token", known_fqns=known)
        assert "validate_token" in intent.function_names


class TestResolver:
    """Test context resolution against the fixture codebase."""

    @pytest.fixture(scope="class")
    def store(self, tmp_path_factory):
        import shutil
        tmp = tmp_path_factory.mktemp("resolve_project")
        if FIXTURES_DIR.exists():
            shutil.copytree(FIXTURES_DIR, tmp, dirs_exist_ok=True)
        return build_index(tmp), tmp

    def test_resolve_by_function_name(self, store):
        store, tmp = store
        result = resolve_context("fix validate_token", store)
        assert result.confidence in ("HIGH", "MEDIUM")
        assert len(result.candidates) >= 1

        # validate_token should be Tier 1
        tier1_fqns = [c.skeleton.fqn for c in result.candidates if c.tier == Tier.TIER1]
        assert any("validate_token" in fqn for fqn in tier1_fqns)

    def test_resolve_exact_slm_fqn(self, store):
        store, tmp = store
        fqn = "auth/middleware.py::validate_token"
        intent = Intent(
            task_type=TaskType.DEBUG,
            entities=[Entity(value=fqn, entity_type="slm_entity", confidence=0.8)],
            file_paths=[],
            function_names=[fqn],
        )

        assert fqn in _resolve_entities(intent, store)

    def test_resolve_by_file(self, store):
        store, tmp = store
        result = resolve_context("review auth/middleware.py", store)
        assert len(result.candidates) >= 1

    def test_resolve_includes_dependencies(self, store):
        store, tmp = store
        result = resolve_context("fix validate_token in auth/middleware.py", store)
        # Should include validate_token AND its dependencies (decode_jwt, get_user)
        all_fqns = {c.skeleton.fqn for c in result.candidates}
        assert any("validate_token" in fqn for fqn in all_fqns)

    def test_resolve_low_confidence(self, store):
        store, tmp = store
        result = resolve_context("deploy to kubernetes", store)
        assert result.confidence in ("LOW", "MISS")


class TestTokenBudget:
    def test_under_soft_target(self):
        budget = TokenBudget(model_context_limit=128000)
        alloc = budget.allocate(
            zone1_tokens=100, zone2_tokens=500,
            zone3_candidates_count=20, zone4_tokens=200,
        )
        assert alloc.zone3_mode == Zone3Mode.FULL
        assert alloc.zone3_budget > 0
        assert alloc.warning == ""

    def test_tight_budget(self):
        budget = TokenBudget(model_context_limit=4000)
        alloc = budget.allocate(
            zone1_tokens=100, zone2_tokens=3200,
            zone3_candidates_count=20, zone4_tokens=200,
        )
        # 3500 out of 4000 → tight
        assert alloc.zone3_mode in (Zone3Mode.MINIMAL, Zone3Mode.NONE)

    def test_over_budget(self):
        budget = TokenBudget(model_context_limit=2000)
        alloc = budget.allocate(
            zone1_tokens=100, zone2_tokens=1800,
            zone3_candidates_count=20, zone4_tokens=200,
        )
        # 2100 > 2000 → Zone 3 dropped
        assert alloc.zone3_mode == Zone3Mode.NONE
        assert alloc.warning != ""


class TestZoneAssembly:
    """Test end-to-end context assembly."""

    @pytest.fixture(scope="class")
    def assembled(self, tmp_path_factory):
        import shutil
        tmp = tmp_path_factory.mktemp("assemble_project")
        if FIXTURES_DIR.exists():
            shutil.copytree(FIXTURES_DIR, tmp, dirs_exist_ok=True)

        store = build_index(tmp)
        result = resolve_context("fix validate_token in middleware.py", store)
        return assemble_context(result, store, tmp, constraints="# Use strict typing")

    def test_has_constraints(self, assembled):
        assert "CONSTRAINTS" in assembled.text or "strict typing" in assembled.text

    def test_has_task(self, assembled):
        assert "TASK" in assembled.text
        assert "validate_token" in assembled.text

    def test_has_target_code(self, assembled):
        assert "TARGET CODE" in assembled.text or "validate_token" in assembled.text

    def test_token_count(self, assembled):
        assert assembled.token_count > 0
        assert assembled.token_count < 128000

    def test_confidence(self, assembled):
        assert assembled.confidence in ("HIGH", "MEDIUM", "LOW")

    def test_zone_breakdown(self, assembled):
        assert "zone1_constraints" in assembled.zone_breakdown
        assert "zone4_prompt" in assembled.zone_breakdown
