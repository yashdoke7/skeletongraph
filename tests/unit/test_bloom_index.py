"""Tests for Bloom filter and inverted index."""

import pytest

from skeletongraph.graph.bloom import BloomFilter
from skeletongraph.graph.inverted_index import (
    InvertedIndex,
    tokenize_identifier,
    tokenize_text,
)


class TestBloomFilter:
    def test_basic_membership(self):
        bf = BloomFilter(expected_items=100, fp_rate=0.01)
        bf.add("validate_token")
        bf.add("decode_jwt")

        assert bf.contains("validate_token")
        assert bf.contains("decode_jwt")
        assert not bf.contains("nonexistent_function")

    def test_in_operator(self):
        bf = BloomFilter(expected_items=100)
        bf.add("test")
        assert "test" in bf
        assert "missing" not in bf

    def test_no_false_negatives(self):
        bf = BloomFilter(expected_items=1000, fp_rate=0.001)
        items = [f"function_{i}" for i in range(1000)]
        bf.add_all(items)

        for item in items:
            assert bf.contains(item), f"False negative for {item}"

    def test_size_bytes(self):
        bf = BloomFilter(expected_items=10000, fp_rate=0.01)
        # Should be ~12KB for 10K items at 1% FP
        assert bf.size_bytes < 20000  # Generous upper bound

    def test_serialization(self):
        bf = BloomFilter(expected_items=100)
        bf.add("alpha")
        bf.add("beta")

        data = bf.to_bytes()
        restored = BloomFilter.from_bytes(data)

        assert restored.contains("alpha")
        assert restored.contains("beta")
        assert not restored.contains("gamma")

    def test_empty_filter(self):
        bf = BloomFilter(expected_items=10)
        assert len(bf) == 0
        assert not bf.contains("anything")

    def test_estimated_fp_rate(self):
        bf = BloomFilter(expected_items=100, fp_rate=0.01)
        for i in range(100):
            bf.add(f"item_{i}")
        # Should be around 1%
        assert bf.estimated_fp_rate < 0.05


class TestTokenizeIdentifier:
    def test_snake_case(self):
        assert tokenize_identifier("validate_token") == ["validate", "token"]

    def test_camel_case(self):
        tokens = tokenize_identifier("getUserById")
        assert "user" in tokens
        assert "by" in tokens

    def test_pascal_case(self):
        tokens = tokenize_identifier("AuthMiddleware")
        assert "auth" in tokens
        assert "middleware" in tokens

    def test_single_word(self):
        assert tokenize_identifier("authenticate") == ["authenticate"]

    def test_filters_stop_words(self):
        tokens = tokenize_identifier("get_self_value")
        assert "self" not in tokens

    def test_filters_short(self):
        # Single char parts should be filtered
        tokens = tokenize_identifier("a_b_validate")
        assert "validate" in tokens


class TestTokenizeText:
    def test_basic(self):
        tokens = tokenize_text("Validates JWT token and returns boolean")
        assert "validates" in tokens
        assert "jwt" in tokens
        assert "token" in tokens

    def test_filters_stop_words(self):
        tokens = tokenize_text("the function returns a value")
        assert "the" not in tokens


class TestInvertedIndex:
    @pytest.fixture
    def auth_index(self):
        idx = InvertedIndex()
        idx.add(
            "m.py::validate_token", "validate_token",
            "def validate_token(token: str) -> bool:",
            "Validates JWT token. Returns False if expired."
        )
        idx.add(
            "m.py::decode_jwt", "decode_jwt",
            "def decode_jwt(token: str) -> dict:",
            "Decodes and verifies JWT token payload."
        )
        idx.add(
            "m.py::authenticate", "authenticate",
            "def authenticate(request: Request) -> User:",
            "Authenticates a request using JWT middleware."
        )
        idx.add(
            "db.py::connect_db", "connect_db",
            "def connect_db(url: str) -> Connection:",
            "Connects to the PostgreSQL database."
        )
        return idx

    def test_search_keyword(self, auth_index):
        results = auth_index.search("jwt token validation")
        fqns = [fqn for fqn, _ in results]
        # validate_token and decode_jwt both relate to JWT
        assert "m.py::validate_token" in fqns
        assert "m.py::decode_jwt" in fqns

    def test_search_database(self, auth_index):
        results = auth_index.search("database connection")
        fqns = [fqn for fqn, _ in results]
        assert "db.py::connect_db" in fqns

    def test_search_no_match(self, auth_index):
        results = auth_index.search("kubernetes deployment")
        assert len(results) == 0

    def test_direct_lookup(self, auth_index):
        fqns = auth_index.lookup("jwt")
        assert "m.py::validate_token" in fqns
        assert "m.py::decode_jwt" in fqns
        assert "db.py::connect_db" not in fqns

    def test_remove(self, auth_index):
        auth_index.remove("m.py::validate_token")
        fqns = auth_index.lookup("validate")
        assert "m.py::validate_token" not in fqns

    def test_entry_count(self, auth_index):
        assert auth_index.entry_count == 4

    def test_serialization(self, auth_index):
        data = auth_index.to_dict()
        restored = InvertedIndex.from_dict(data)

        results = restored.search("jwt token")
        fqns = [fqn for fqn, _ in results]
        assert "m.py::validate_token" in fqns
        assert restored.entry_count == 4
