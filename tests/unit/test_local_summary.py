from skeletongraph.parser.skeleton import SkeletonCore
from skeletongraph.parser.node_kinds import NodeKind
from skeletongraph.summary.local import build_local_summary


def test_local_summary_uses_docstring_first():
    sk = SkeletonCore(
        fqn="auth/middleware.py::validate_token",
        file_path="auth/middleware.py",
        line_start=1,
        line_end=4,
        kind=NodeKind.FUNCTION,
        signature="def validate_token(token: str) -> bool:",
        docstring="Validate JWT token and return whether it is usable.",
    )

    assert build_local_summary(sk, ["jwt"]) == "Validate JWT token and return whether it is usable."


def test_local_summary_uses_signature_and_body_keywords_without_model():
    sk = SkeletonCore(
        fqn="auth/middleware.py::validate_token",
        file_path="auth/middleware.py",
        line_start=1,
        line_end=4,
        kind=NodeKind.FUNCTION,
        signature="def validate_token(token: str, user_id: str) -> bool:",
    )

    summary = build_local_summary(sk, ["jwt", "expired", "authorization", "jwt"])

    assert "validate token" in summary
    assert "parameters: token, user_id" in summary
    assert "returns bool" in summary
    assert "uses jwt, expired, authorization" in summary
