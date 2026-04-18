"""Tests for Tree-sitter AST extraction — Python language."""

import pytest
from pathlib import Path

from skeletongraph.parser.ast_extractor import extract_file, result_to_file_skeleton
from skeletongraph.parser.node_kinds import NodeKind


FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "python_small"


class TestPythonExtraction:
    """Test extraction from the auth/middleware.py fixture."""

    @pytest.fixture(scope="class")
    def result(self):
        """Extract the middleware fixture."""
        return extract_file(
            "auth/middleware.py",
            FIXTURES_DIR,
        )

    def test_basic_metadata(self, result):
        assert result is not None
        assert result.language == "python"
        assert result.file_path == "auth/middleware.py"
        assert result.total_lines > 0

    def test_module_docstring(self, result):
        assert "Authentication middleware" in result.module_docstring

    def test_imports(self, result):
        assert len(result.imports) >= 3
        modules = [imp.module for imp in result.imports]
        assert "jwt" in modules

        # Check relative from-import
        model_imp = [i for i in result.imports if ".models" in i.module]
        assert len(model_imp) >= 1
        assert "User" in model_imp[0].names

    def test_exports(self, result):
        assert "AuthMiddleware" in result.exports
        assert "validate_token" in result.exports

    def test_constants(self, result):
        const_names = [c[0] for c in result.constants]
        assert "MAX_TOKEN_AGE" in const_names
        assert "DEFAULT_ALGORITHM" in const_names

    def test_class_extraction(self, result):
        assert len(result.classes) == 1
        cls = result.classes[0]
        assert cls.name == "AuthMiddleware"
        assert cls.fqn == "auth/middleware.py::AuthMiddleware"
        assert len(cls.methods) >= 2  # __init__, __call__, _extract_token

    def test_class_methods(self, result):
        cls = result.classes[0]
        method_names = {m.name for m in cls.methods}
        assert "__init__" in method_names
        assert "__call__" in method_names
        assert "_extract_token" in method_names

    def test_constructor_detection(self, result):
        cls = result.classes[0]
        init = [m for m in cls.methods if m.name == "__init__"][0]
        assert init.kind == NodeKind.CONSTRUCTOR

    def test_async_method(self, result):
        cls = result.classes[0]
        call_method = [m for m in cls.methods if m.name == "__call__"][0]
        # __call__ is async but inside a class, so it's a METHOD
        assert call_method.kind == NodeKind.METHOD
        assert "async" in call_method.signature

    def test_top_level_functions(self, result):
        func_names = {f.name for f in result.functions}
        assert "validate_token" in func_names
        assert "decode_jwt" in func_names
        assert "get_user" in func_names

    def test_function_signatures(self, result):
        validate = [f for f in result.functions if f.name == "validate_token"][0]
        assert "token: str" in validate.signature
        assert "secret: str" in validate.signature

    def test_function_kind(self, result):
        validate = [f for f in result.functions if f.name == "validate_token"][0]
        assert validate.kind == NodeKind.FUNCTION

    def test_function_line_range(self, result):
        validate = [f for f in result.functions if f.name == "validate_token"][0]
        assert validate.line_start > 0
        assert validate.line_end > validate.line_start

    def test_function_body_text(self, result):
        validate = [f for f in result.functions if f.name == "validate_token"][0]
        assert "decode_jwt" in validate.body_text
        assert "get_user" in validate.body_text


class TestFileSkeletonConversion:
    """Test converting FileExtractionResult to FileSkeleton."""

    @pytest.fixture(scope="class")
    def file_skeleton(self):
        result = extract_file("auth/middleware.py", FIXTURES_DIR)
        return result_to_file_skeleton(result)

    def test_file_path(self, file_skeleton):
        assert file_skeleton.path == "auth/middleware.py"

    def test_all_skeletons(self, file_skeleton):
        all_sk = file_skeleton.all_skeletons
        # Should have top-level functions + class methods
        assert len(all_sk) >= 5

    def test_skeleton_fqns(self, file_skeleton):
        fqns = file_skeleton.all_fqns
        assert "auth/middleware.py::validate_token" in fqns
        assert "auth/middleware.py::AuthMiddleware.__init__" in fqns

    def test_class_skeleton(self, file_skeleton):
        assert len(file_skeleton.classes) == 1
        cls = file_skeleton.classes[0]
        assert cls.name == "AuthMiddleware"
        assert len(cls.methods) >= 2

    def test_constructor_params(self, file_skeleton):
        cls = file_skeleton.classes[0]
        # Should have extracted params from __init__
        # (secret: str, algorithm: str = "HS256")
        # 'self' should be excluded
        assert len(cls.constructor_params) >= 1

    def test_instance_attrs(self, file_skeleton):
        cls = file_skeleton.classes[0]
        # self.app, self.secret, self.algorithm
        assert any("self.secret" in attr for attr in cls.instance_attrs)

    def test_imports_as_strings(self, file_skeleton):
        assert any("jwt" in imp for imp in file_skeleton.imports)

    def test_serialization_roundtrip(self, file_skeleton):
        d = file_skeleton.to_dict()
        from skeletongraph.parser.skeleton import FileSkeleton
        restored = FileSkeleton.from_dict(d)
        assert restored.path == file_skeleton.path
        assert len(restored.all_skeletons) == len(file_skeleton.all_skeletons)
