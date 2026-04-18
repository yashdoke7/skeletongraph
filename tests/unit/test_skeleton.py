"""Tests for core data structures: SkeletonCore, ClassSkeleton, FileSkeleton."""

import pytest

from skeletongraph.parser.node_kinds import NodeKind
from skeletongraph.parser.skeleton import (
    ClassSkeleton,
    FileSkeleton,
    SkeletonCore,
    make_fqn,
    make_lambda_fqn,
)


class TestNodeKind:
    def test_callable_kinds(self):
        assert NodeKind.FUNCTION.is_callable
        assert NodeKind.METHOD.is_callable
        assert NodeKind.CONSTRUCTOR.is_callable
        assert NodeKind.ASYNC_FUNCTION.is_callable
        assert not NodeKind.CLASS.is_callable
        assert not NodeKind.INTERFACE.is_callable

    def test_type_def_kinds(self):
        assert NodeKind.CLASS.is_type_definition
        assert NodeKind.STRUCT.is_type_definition
        assert NodeKind.INTERFACE.is_type_definition
        assert NodeKind.TRAIT.is_type_definition
        assert not NodeKind.FUNCTION.is_type_definition
        assert not NodeKind.METHOD.is_type_definition

    def test_container_kinds(self):
        assert NodeKind.CLASS.is_container
        assert NodeKind.MODULE.is_container
        assert NodeKind.IMPL_BLOCK.is_container
        assert not NodeKind.FUNCTION.is_container
        assert not NodeKind.LAMBDA.is_container

    def test_auto_constructor(self):
        assert NodeKind.CLASS.auto_include_constructor
        assert NodeKind.STRUCT.auto_include_constructor
        assert not NodeKind.INTERFACE.auto_include_constructor
        assert not NodeKind.FUNCTION.auto_include_constructor


class TestSkeletonCore:
    @pytest.fixture
    def sample_skeleton(self):
        return SkeletonCore(
            fqn="auth/middleware.py::AuthMiddleware.validate_token",
            file_path="auth/middleware.py",
            line_start=45,
            line_end=62,
            signature="def validate_token(self, token: str, expiry: int) -> bool:",
            kind=NodeKind.METHOD,
            decorators=("@login_required",),
            is_exported=True,
            complexity=8,
            body_token_estimate=180,
            sha256="abc123",
        )

    def test_creation(self, sample_skeleton):
        assert sample_skeleton.fqn == "auth/middleware.py::AuthMiddleware.validate_token"
        assert sample_skeleton.kind == NodeKind.METHOD
        assert sample_skeleton.complexity == 8

    def test_frozen(self, sample_skeleton):
        with pytest.raises(AttributeError):
            sample_skeleton.fqn = "new_fqn"  # type: ignore

    def test_file_display(self, sample_skeleton):
        assert sample_skeleton.file_display == "middleware.py:45"

    def test_return_type_python(self, sample_skeleton):
        assert sample_skeleton.return_type == "bool"

    def test_return_type_none(self):
        sk = SkeletonCore(
            fqn="test.py::func",
            file_path="test.py",
            line_start=1,
            line_end=5,
            signature="def func(x):",
            kind=NodeKind.FUNCTION,
        )
        assert sk.return_type is None

    def test_tier2_str(self, sample_skeleton):
        s = sample_skeleton.to_tier2_str("Validates JWT token. False if expired.")
        assert "@login_required" in s
        assert "def validate_token" in s
        assert "Validates JWT" in s

    def test_tier3_str(self, sample_skeleton):
        s = sample_skeleton.to_tier3_str()
        assert "validate_token" in s
        assert "bool" in s

    def test_serialization(self, sample_skeleton):
        d = sample_skeleton.to_dict()
        restored = SkeletonCore.from_dict(d)
        assert restored.fqn == sample_skeleton.fqn
        assert restored.kind == sample_skeleton.kind
        assert restored.decorators == sample_skeleton.decorators
        assert restored.complexity == sample_skeleton.complexity


class TestClassSkeleton:
    def test_creation_and_serialization(self):
        method = SkeletonCore(
            fqn="auth.py::Auth.login",
            file_path="auth.py",
            line_start=10,
            line_end=20,
            signature="def login(self, user: str) -> bool:",
            kind=NodeKind.METHOD,
        )
        cls = ClassSkeleton(
            name="Auth",
            fqn="auth.py::Auth",
            file_path="auth.py",
            bases=["BaseAuth"],
            constructor_params=["secret: str"],
            instance_attrs=["self.secret"],
            methods=[method],
            line_start=5,
            line_end=25,
        )

        d = cls.to_dict()
        restored = ClassSkeleton.from_dict(d)
        assert restored.name == "Auth"
        assert restored.bases == ["BaseAuth"]
        assert len(restored.methods) == 1
        assert restored.methods[0].fqn == "auth.py::Auth.login"


class TestFileSkeleton:
    def test_all_skeletons(self):
        func = SkeletonCore(
            fqn="test.py::helper",
            file_path="test.py",
            line_start=1,
            line_end=5,
            signature="def helper():",
            kind=NodeKind.FUNCTION,
        )
        method = SkeletonCore(
            fqn="test.py::MyClass.do_thing",
            file_path="test.py",
            line_start=10,
            line_end=15,
            signature="def do_thing(self):",
            kind=NodeKind.METHOD,
        )
        cls = ClassSkeleton(
            name="MyClass",
            fqn="test.py::MyClass",
            file_path="test.py",
            methods=[method],
        )
        fs = FileSkeleton(
            path="test.py",
            functions=[func],
            classes=[cls],
            total_lines=20,
        )

        all_sk = fs.all_skeletons
        assert len(all_sk) == 2
        assert set(fs.all_fqns) == {"test.py::helper", "test.py::MyClass.do_thing"}

    def test_serialization(self):
        fs = FileSkeleton(
            path="utils.py",
            summary="Utility functions",
            imports=["import os", "from pathlib import Path"],
            exports=["helper"],
            constants=[("MAX_SIZE", "1024")],
            total_lines=100,
            sha256="def456",
        )
        d = fs.to_dict()
        restored = FileSkeleton.from_dict(d)
        assert restored.path == "utils.py"
        assert restored.imports == ["import os", "from pathlib import Path"]
        assert restored.constants == [("MAX_SIZE", "1024")]


class TestFQNConstruction:
    def test_make_fqn_simple(self):
        assert make_fqn("utils.py", "helper") == "utils.py::helper"

    def test_make_fqn_method(self):
        result = make_fqn("auth/middleware.py", "AuthMiddleware", "validate_token")
        assert result == "auth/middleware.py::AuthMiddleware.validate_token"

    def test_make_lambda_fqn(self):
        result = make_lambda_fqn("utils.py", "utils.py::process_data", 45)
        assert result == "utils.py::process_data.<lambda:45>"
