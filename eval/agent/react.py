"""The ReAct loop — model <-> tools, against an OpenAI-compatible endpoint.

Identical for every arm. The model is given the issue + the five tools; it
iterates (think -> tool call -> observe) until it calls `submit` or hits
MAX_TURNS. Every turn's token usage and tool calls are recorded.

vLLM serving note: start vLLM with `--enable-auto-tool-choice` and the matching
`--tool-call-parser` (hermes for Qwen2.5). See serve_model.sh.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import List

from . import config
from .tools import TOOL_SCHEMAS, ToolExecutor

# Valid tool names — used to validate text-parsed tool calls (below).
_VALID_TOOLS = {s["function"]["name"] for s in TOOL_SCHEMAS}

SYSTEM_PROMPT = """You are an autonomous software engineer fixing a bug in a \
repository. You can only see the code through your tools.

Available tools: search_code, list_files, read_file, edit_file, submit.

Process:
1. Use search_code to locate the code relevant to the issue.
2. read_file to inspect the relevant code.
3. edit_file to make the minimal correct fix.
4. Call submit when done.

Rules:
- Make the smallest change that correctly fixes the issue.
- Do NOT run or write tests — the test environment is not available.
- Do NOT explain at length; act through tools.
- When the fix is complete, call submit.
"""

USER_TEMPLATE = """Fix the following GitHub issue in this repository.

--- ISSUE ---
{issue}
"""


@dataclass
class Turn:
    index: int
    usage: dict
    tool_calls: List[dict] = field(default_factory=list)
    text: str = ""
    latency_s: float = 0.0


@dataclass
class Trajectory:
    task_id: str
    arm: str
    model: str
    turns: List[Turn] = field(default_factory=list)
    stopped: str = ""            # "submit" | "max_turns" | "error" | "no_tool"
    error: str = ""
    first_search_hits: List[str] = field(default_factory=list)
    wall_s: float = 0.0

    # ── aggregates ─────────────────────────────────────────────────────────
    def billed_input(self) -> int:
        return sum(t.usage.get("prompt_tokens", 0) for t in self.turns)

    def billed_output(self) -> int:
        return sum(t.usage.get("completion_tokens", 0) for t in self.turns)

    def cached_input(self) -> int:
        return sum(t.usage.get("cached_tokens", 0) for t in self.turns)

    def peak_context(self) -> int:
        return max((t.usage.get("prompt_tokens", 0) for t in self.turns), default=0)

    def tool_counts(self) -> dict:
        c: dict = {}
        for t in self.turns:
            for call in t.tool_calls:
                c[call["name"]] = c.get(call["name"], 0) + 1
        return c

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id, "arm": self.arm, "model": self.model,
            "stopped": self.stopped, "error": self.error,
            "n_turns": len(self.turns),
            "billed_input": self.billed_input(),
            "billed_output": self.billed_output(),
            "cached_input": self.cached_input(),
            "peak_context": self.peak_context(),
            "tool_counts": self.tool_counts(),
            "first_search_hits": self.first_search_hits,
            "wall_s": round(self.wall_s, 1),
            "imputed_cost": config.impute_cost(
                self.billed_input(), self.billed_output(), self.cached_input()),
            "turns": [
                {"index": t.index, "usage": t.usage, "latency_s": round(t.latency_s, 2),
                 "tool_calls": t.tool_calls, "text": t.text[:2000]}
                for t in self.turns
            ],
        }


def _client():
    from openai import OpenAI
    return OpenAI(base_url=config.API_BASE, api_key=config.API_KEY,
                  timeout=config.REQUEST_TIMEOUT)


def _usage(resp) -> dict:
    u = getattr(resp, "usage", None)
    if not u:
        return {"prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0}
    cached = 0
    details = getattr(u, "prompt_tokens_details", None)
    if details is not None:
        cached = getattr(details, "cached_tokens", 0) or 0
    return {
        "prompt_tokens": getattr(u, "prompt_tokens", 0) or 0,
        "completion_tokens": getattr(u, "completion_tokens", 0) or 0,
        "cached_tokens": cached,
    }


def _parse_text_tool_calls(content: str) -> List[dict]:
    """Fallback parser for tool calls emitted as plain-text JSON.

    Not every endpoint returns structured `tool_calls`: Ollama, and vLLM when
    its --tool-call-parser misfires, can leave the call as JSON in the message
    *content* instead. Without this, the ReAct loop sees "no tool call" and
    bails after 2 turns — silently corrupting the whole run.

    Extracts {"name", "arguments"} objects whose name is a real tool. Tolerates
    Hermes <tool_call> wrappers, markdown ```json fences, and embedded objects.
    """
    if not content:
        return []
    text = content.replace("<tool_call>", " ").replace("</tool_call>", " ")

    blobs: List[str] = []
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        blobs.append(stripped)
    blobs += re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    # embedded objects (one level of nesting — enough for {name, arguments:{...}})
    blobs += re.findall(r"\{(?:[^{}]|\{[^{}]*\})*\}", text)

    calls: List[dict] = []
    seen: set = set()
    for blob in blobs:
        try:
            obj = json.loads(blob)
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(obj, dict):
            continue
        name = obj.get("name") or obj.get("tool") or obj.get("function")
        if isinstance(name, dict):                 # {"function": {"name": ...}}
            name = name.get("name")
        args = obj.get("arguments")
        if args is None:
            args = obj.get("parameters")
        if args is None:
            args = obj.get("args")
        if not isinstance(name, str) or name not in _VALID_TOOLS:
            continue
        if not isinstance(args, dict):
            args = {}
        key = name + json.dumps(args, sort_keys=True)
        if key in seen:                            # dedup: same call matched twice
            continue
        seen.add(key)
        calls.append({"name": name, "arguments": args})
    return calls


def run_react(task: dict, arm: str, executor: ToolExecutor,
              model: str = "qwen-32b") -> Trajectory:
    """Run one task with one arm to completion. Returns the trajectory."""
    client = _client()
    traj = Trajectory(task_id=task["task_id"], arm=arm, model=model)
    t0 = time.time()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(issue=task["query"])},
    ]

    for step in range(config.MAX_TURNS):
        ts = time.time()
        try:
            resp = client.chat.completions.create(
                model=config.MODEL_NAME,
                messages=messages,
                tools=TOOL_SCHEMAS,
                tool_choice="auto",
                temperature=config.TEMPERATURE,
                seed=config.SEED,
            )
        except Exception as e:
            traj.stopped, traj.error = "error", f"{type(e).__name__}: {e}"
            break

        choice = resp.choices[0]
        msg = choice.message
        turn = Turn(index=step, usage=_usage(resp),
                    text=msg.content or "", latency_s=time.time() - ts)

        native_calls = msg.tool_calls or []
        # Fallback: model emitted the call as text JSON, not structured tool_calls.
        text_calls = ([] if native_calls
                      else _parse_text_tool_calls(msg.content or ""))

        if not native_calls and not text_calls:
            # model answered without acting — nudge once, else stop
            turn.tool_calls = []
            traj.turns.append(turn)
            messages.append({"role": "assistant", "content": msg.content or ""})
            if step > 0 and not traj.turns[step - 1].tool_calls:
                traj.stopped = "no_tool"
                break
            messages.append({"role": "user",
                              "content": "Use a tool, or call submit if done."})
            continue

        if native_calls:
            # ── structured tool_calls path ──────────────────────────────────
            messages.append({
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {"id": tc.id, "type": "function",
                     "function": {"name": tc.function.name,
                                  "arguments": tc.function.arguments}}
                    for tc in native_calls
                ],
            })
            for tc in native_calls:
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                result = executor.run(name, args)
                turn.tool_calls.append({"name": name, "args": args,
                                        "result": result[:1500]})
                messages.append({"role": "tool", "tool_call_id": tc.id,
                                 "content": result})
        else:
            # ── text-form tool calls (fallback) ─────────────────────────────
            # Feed results back as user messages — robust across endpoints that
            # reject synthetic tool_calls / tool-role pairing.
            messages.append({"role": "assistant", "content": msg.content or ""})
            for call in text_calls:
                name, args = call["name"], call["arguments"]
                result = executor.run(name, args)
                turn.tool_calls.append({"name": name, "args": args,
                                        "result": result[:1500]})
                messages.append({"role": "user",
                                 "content": f"[tool result: {name}]\n{result}"})

        traj.turns.append(turn)
        if executor.submitted:
            traj.stopped = "submit"
            break
    else:
        traj.stopped = "max_turns"

    traj.first_search_hits = executor.first_search_hits
    traj.wall_s = time.time() - t0
    return traj
