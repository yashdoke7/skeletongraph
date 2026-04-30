import json
import sqlite3

from skeletongraph.eval.parsers.antigravity import parse_antigravity_sg_session
from skeletongraph.eval.parsers.copilot import parse_copilot_json_export
from skeletongraph.eval.parsers.cursor import parse_cursor_session


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


def test_parse_cursor_current_global_storage(tmp_path):
    user_dir = tmp_path / "User"
    workspace_dir = user_dir / "workspaceStorage" / "abc"
    global_dir = user_dir / "globalStorage"
    workspace_dir.mkdir(parents=True)
    global_dir.mkdir(parents=True)

    workspace_db = workspace_dir / "state.vscdb"
    global_db = global_dir / "state.vscdb"
    composer_id = "composer-1"
    user_bubble = "user-1"
    tool_bubble = "tool-1"
    response_bubble = "response-1"

    with sqlite3.connect(workspace_db) as conn:
        conn.execute("CREATE TABLE ItemTable (key TEXT, value TEXT)")
        conn.execute("CREATE TABLE cursorDiskKV (key TEXT, value BLOB)")
        conn.execute(
            "INSERT INTO ItemTable VALUES (?, ?)",
            ("composer.composerData", json.dumps({"selectedComposerIds": [composer_id]})),
        )

    with sqlite3.connect(global_db) as conn:
        conn.execute("CREATE TABLE ItemTable (key TEXT, value TEXT)")
        conn.execute("CREATE TABLE cursorDiskKV (key TEXT, value BLOB)")
        conn.execute(
            "INSERT INTO cursorDiskKV VALUES (?, ?)",
            (
                f"composerData:{composer_id}",
                json.dumps(
                    {
                        "fullConversationHeadersOnly": [
                            {"bubbleId": user_bubble},
                            {"bubbleId": tool_bubble},
                            {"bubbleId": response_bubble},
                        ]
                    }
                ),
            ),
        )
        conn.execute(
            "INSERT INTO cursorDiskKV VALUES (?, ?)",
            (
                f"bubbleId:{composer_id}:{user_bubble}",
                json.dumps(
                    {
                        "type": 1,
                        "richText": json.dumps(
                            {
                                "root": {
                                    "children": [
                                        {"children": [{"text": "fix content length"}]}
                                    ]
                                }
                            }
                        ),
                    }
                ),
            ),
        )
        conn.execute(
            "INSERT INTO cursorDiskKV VALUES (?, ?)",
            (
                f"bubbleId:{composer_id}:{tool_bubble}",
                json.dumps(
                    {
                        "type": 2,
                        "toolFormerData": {
                            "name": "read_file_v2",
                            "params": json.dumps({"targetFile": str(tmp_path / "models.py")}),
                            "result": json.dumps({"contents": "def prepare_content_length():\n    pass"}),
                        },
                    }
                ),
            ),
        )
        conn.execute(
            "INSERT INTO cursorDiskKV VALUES (?, ?)",
            (
                f"bubbleId:{composer_id}:{response_bubble}",
                json.dumps({"type": 2, "text": "I found the header logic."}),
            ),
        )

    trace = parse_cursor_session(workspace_db, tmp_path, project_name="requests")

    assert trace.agent == "cursor"
    assert trace.task_prompt == "fix content length"
    assert trace.tool_call_count == 1
    assert trace.tool_calls[0].tool_type == "view_file"
    assert trace.total_response_tokens > 0


def test_parse_copilot_debug_jsonl_exact_api_tokens(tmp_path):
    log_path = tmp_path / "main.jsonl"
    events = [
        {
            "type": "user_message",
            "attrs": {"content": "fix content length"},
        },
        {
            "type": "llm_request",
            "name": "chat:gpt-5.2-codex",
            "attrs": {"inputTokens": 10, "outputTokens": 3},
        },
        {
            "type": "tool_call",
            "name": "grep_search",
            "attrs": {
                "args": json.dumps({"query": "Content-Length"}),
                "result": "models.py: self.headers['Content-Length'] = '0'",
            },
        },
        {
            "type": "llm_request",
            "name": "chat:gpt-5.2-codex",
            "attrs": {"inputTokens": 20, "outputTokens": 7},
        },
        {
            "type": "agent_response",
            "attrs": {
                "response": json.dumps(
                    [{"role": "assistant", "parts": [{"type": "text", "text": "Done"}]}]
                )
            },
        },
    ]
    log_path.write_text("\n".join(json.dumps(e) for e in events), encoding="utf-8")

    trace = parse_copilot_json_export(log_path, tmp_path, project_name="requests")

    assert trace.task_prompt == "fix content length"
    assert trace.api_input_tokens == 30
    assert trace.reasoning_tokens == 10
    assert trace.model_turns == 2
    assert trace.tool_call_count == 1
    assert trace.grep_count == 1


def test_parse_copilot_otel_debug_export_exact_api_tokens(tmp_path):
    export_path = tmp_path / "agent-debug-log.json"
    export_path.write_text(
        json.dumps(
            {
                "resourceSpans": [
                    {
                        "scopeSpans": [
                            {
                                "spans": [
                                    {
                                        "name": "chat:gpt-5.2-codex",
                                        "attributes": [
                                            {
                                                "key": "gen_ai.operation.name",
                                                "value": {"stringValue": "chat"},
                                            },
                                            {
                                                "key": "gen_ai.usage.input_tokens",
                                                "value": {"intValue": "100"},
                                            },
                                            {
                                                "key": "gen_ai.usage.output_tokens",
                                                "value": {"intValue": "20"},
                                            },
                                        ],
                                    },
                                    {
                                        "name": "grep_search",
                                        "attributes": [
                                            {
                                                "key": "gen_ai.operation.name",
                                                "value": {"stringValue": "execute_tool"},
                                            },
                                            {
                                                "key": "gen_ai.tool.name",
                                                "value": {"stringValue": "grep_search"},
                                            },
                                            {
                                                "key": "gen_ai.tool.call.arguments",
                                                "value": {"stringValue": json.dumps({"query": "Content-Length"})},
                                            },
                                            {
                                                "key": "gen_ai.tool.call.result",
                                                "value": {"stringValue": "models.py: Content-Length"},
                                            },
                                        ],
                                    },
                                ]
                            }
                        ]
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    trace = parse_copilot_json_export(export_path, tmp_path, project_name="requests")

    assert trace.api_input_tokens == 100
    assert trace.reasoning_tokens == 20
    assert trace.model_turns == 1
    assert trace.tool_call_count == 1
    assert trace.grep_count == 1
