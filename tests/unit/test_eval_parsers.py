import json

from skeletongraph.eval.parsers.antigravity import parse_antigravity_sg_session


def test_parse_sg_session_from_archived_file(tmp_path):
    archived = tmp_path / "sg_session.json"
    archived.write_text(
        json.dumps(
            {
                "turns": [
                    {
                        "prompt": "fix login middleware",
                        "token_count": 123,
                        "fqns_returned": ["auth/middleware.py::require_auth"],
                        "response_text": "Use the middleware entry point.",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    trace = parse_antigravity_sg_session(
        tmp_path,
        project_name="demo",
        agent="copilot",
        session_path=archived,
    )

    assert trace.agent == "copilot"
    assert trace.mode == "skeletongraph"
    assert trace.project == "demo"
    assert trace.task_prompt == "fix login middleware"
    assert [call.tool_type for call in trace.tool_calls] == [
        "query_context",
        "sg_context_file",
    ]
    assert trace.tool_calls[1].target == "auth/middleware.py"
