"""
Built-in evaluation cases for the python_small fixture.

Run with: python -m skeletongraph.eval.run_eval
"""

from __future__ import annotations

from pathlib import Path

from eval.metrics import EvalCase, evaluate, format_report


# Evaluation cases for the auth middleware fixture
PYTHON_SMALL_CASES = [
    EvalCase(
        prompt="fix validate_token in middleware.py",
        expected_fqns=[
            "auth/middleware.py::validate_token",
            "auth/middleware.py::decode_jwt",
        ],
        constraints="# Use strict typing",
        description="Debug: fix a specific function",
    ),
    EvalCase(
        prompt="how does AuthMiddleware.__call__ handle authentication?",
        expected_fqns=[
            "auth/middleware.py::AuthMiddleware.__init__",
            "auth/middleware.py::AuthMiddleware.__call__",
        ],
        description="Explain: understand a class method",
    ),
    EvalCase(
        prompt="refactor decode_jwt and validate_token in auth/middleware.py",
        expected_fqns=[
            "auth/middleware.py::validate_token",
            "auth/middleware.py::decode_jwt",
        ],
        description="Refactor: restructure auth functions",
    ),
    EvalCase(
        prompt="update AuthMiddleware.__call__ to add rate limiting",
        expected_fqns=[
            "auth/middleware.py::AuthMiddleware.__call__",
        ],
        description="Edit: add feature to existing method",
    ),
    EvalCase(
        prompt="review auth/middleware.py for security issues",
        expected_fqns=[
            "auth/middleware.py::validate_token",
            "auth/middleware.py::decode_jwt",
        ],
        description="Review: security audit",
    ),
]


def run_python_small_eval() -> None:
    """Run evaluation against the python_small fixture."""
    fixture_dir = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "python_small"
    if not fixture_dir.exists():
        print(f"Fixture directory not found: {fixture_dir}")
        return

    # Use a temp copy to avoid polluting fixtures
    import shutil
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        shutil.copytree(fixture_dir, tmp_path, dirs_exist_ok=True)

        summary = evaluate(tmp_path, PYTHON_SMALL_CASES)
        print(format_report(summary))


if __name__ == "__main__":
    run_python_small_eval()
