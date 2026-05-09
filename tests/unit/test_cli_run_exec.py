import json

from skeletongraph.cli.run_exec import (
    RUN_SYSTEM_PROMPT,
    RunPlan,
    build_execution_prompt,
    write_run_log,
)


def test_build_execution_prompt_includes_context_and_contract():
    prompt = build_execution_prompt("fix auth", "## Target Code\npass")

    assert "fix auth" in prompt
    assert "## Target Code" in prompt
    assert "unified diff patch" in prompt
    assert RUN_SYSTEM_PROMPT


def test_run_plan_dict_has_routing_and_provider_fields():
    plan = RunPlan(
        prompt="fix auth",
        mode="debug_targeted",
        routed_tier="mlm",
        selected_tier="mlm",
        selected_model="claude-sonnet-4-6",
        cli_provider="anthropic",
        api_key_env=["ANTHROPIC_API_KEY"],
        api_key_configured=False,
        api_base=None,
        context_tokens=1200,
        confidence="HIGH",
        complexity_score=0.42,
        routing_reason="mode=debug_targeted",
        targets=["auth.py::validate"],
    )

    data = plan.to_dict()

    assert data["selected_model"] == "claude-sonnet-4-6"
    assert data["api_key_env"] == ["ANTHROPIC_API_KEY"]
    assert data["api_base"] is None
    assert data["targets"] == ["auth.py::validate"]


def test_write_run_log_appends_jsonl(tmp_path):
    log_path = tmp_path / "runs" / "run_log.jsonl"

    write_run_log(log_path, {"mode": "debug_targeted", "cost": 0.01})
    write_run_log(log_path, {"mode": "review", "cost": 0.02})

    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["mode"] == "debug_targeted"
    assert json.loads(lines[1])["cost"] == 0.02
