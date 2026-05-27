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

# API call robustness — retry transient failures so one blip doesn't drop a task
# (which would unpair the arm comparison). Substring match on the error string.
_MAX_API_ATTEMPTS = 6
# Substring tokens that indicate a transient failure worth retrying.
# Auth failures (401/403/404) are NOT retryable — they will never succeed.
# Status codes are matched as " 4xx " / " 5xx " with spaces so "500" in a
# PermissionDenied 403 message or "503" inside "Authorization" can't false-match.
_RETRYABLE_TOKENS = (
    "429", "RateLimit", "Timeout", "timed out", "APIConnection",
    "ServiceUnavailable", "InternalServer", "overloaded", "Overloaded",
    "Gateway", "temporarily",
)
# Exact HTTP status codes to retry (server-side transient errors only).
_RETRYABLE_STATUS = {"500", "502", "503", "504"}
# Auth / not-found errors — NEVER retry, fail immediately.
_FATAL_TOKENS = ("401", "403", "404", "Forbidden", "Unauthorized", "NotFound",
                 "PermissionDenied", "AuthenticationError")


def _is_retryable(err_msg: str) -> bool:
    # Fatal errors short-circuit first — never retry auth/not-found failures.
    if any(tok in err_msg for tok in _FATAL_TOKENS):
        return False
    if any(tok in err_msg for tok in _RETRYABLE_TOKENS):
        return True
    # Match status codes only when surrounded by non-digit chars (avoids
    # "500" matching inside "Error code: 4500" etc.)
    import re
    return any(re.search(r'\b' + s + r'\b', err_msg) for s in _RETRYABLE_STATUS)

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
    # get_api_key() returns the thread-local key set by run_stage (multi-account
    # NIM rotation) or falls back to the global API_KEY (single-account / vLLM).
    return OpenAI(base_url=config.API_BASE, api_key=config.get_api_key(),
                  timeout=config.REQUEST_TIMEOUT)


def _usage_from_dict(d: dict) -> dict:
    return {
        "prompt_tokens": d.get("prompt_tokens", 0) or 0,
        "completion_tokens": d.get("completion_tokens", 0) or 0,
        "cached_tokens": d.get("cached_tokens", 0) or 0,
    }


def _stream_completion(client, messages: list):
    """Stream a chat completion and reassemble into (content, tool_calls, usage).

    Streaming is critical for reasoning models (DeepSeek V4 Flash, etc.) that
    do chain-of-thought internally: in non-streaming mode the client waits for
    the COMPLETE response (including all hidden thinking tokens) before the first
    byte arrives, causing timeouts. Streaming starts delivering tokens immediately
    (low TTFT) so the full 300s REQUEST_TIMEOUT budget covers actual generation.

    Returns:
        content    – str, the assistant's text reply (may be empty if only tool calls)
        tool_calls – list of SimpleNamespace(id, function=SimpleNamespace(name, arguments))
                     Compatible with the native_calls processing path downstream.
        usage      – dict(prompt_tokens, completion_tokens, cached_tokens)
    """
    from types import SimpleNamespace

    content_parts: List[str] = []
    tc_acc: dict = {}   # chunk index → {"id": str, "name": str, "arguments": str}
    usage: dict = {"prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0}

    # stream_options={"include_usage": True} requests a final usage chunk.
    # Not all NIM endpoints honour it — fall back silently if it errors.
    for use_usage_opt in (True, False):
        kwargs: dict = dict(
            model=config.MODEL_NAME,
            messages=messages,
            tools=TOOL_SCHEMAS,
            tool_choice="auto",
            temperature=config.TEMPERATURE,
            seed=config.SEED,
            stream=True,
        )
        if use_usage_opt:
            kwargs["stream_options"] = {"include_usage": True}
        # Disable chain-of-thought for models that support the toggle.
        # Reasoning/thinking mode causes the model to buffer its full internal
        # chain before sending content chunks, making streaming no faster than
        # non-streaming for our purposes. SG_EVAL_DISABLE_THINKING=0 to re-enable.
        if getattr(config, "DISABLE_THINKING", True):
            kwargs.setdefault("extra_body", {})
            kwargs["extra_body"]["chat_template_kwargs"] = {"enable_thinking": False}
        try:
            stream = client.chat.completions.create(**kwargs)
            for chunk in stream:
                choices = getattr(chunk, "choices", None) or []

                # Usage-only final chunk (from stream_options)
                if not choices:
                    u = getattr(chunk, "usage", None)
                    if u:
                        usage["prompt_tokens"] = getattr(u, "prompt_tokens", 0) or 0
                        usage["completion_tokens"] = getattr(u, "completion_tokens", 0) or 0
                        details = getattr(u, "prompt_tokens_details", None)
                        if details:
                            usage["cached_tokens"] = getattr(details, "cached_tokens", 0) or 0
                    continue

                delta = choices[0].delta

                # Text content (skip reasoning_content — internal thinking tokens)
                if getattr(delta, "content", None):
                    content_parts.append(delta.content)

                # Incremental tool_call deltas — accumulate by index
                for tc in (getattr(delta, "tool_calls", None) or []):
                    i = tc.index
                    if i not in tc_acc:
                        tc_acc[i] = {"id": "", "name": "", "arguments": ""}
                    if tc.id:
                        tc_acc[i]["id"] = tc.id
                    fn = getattr(tc, "function", None)
                    if fn:
                        if getattr(fn, "name", None):
                            tc_acc[i]["name"] += fn.name
                        if getattr(fn, "arguments", None):
                            tc_acc[i]["arguments"] += fn.arguments

                # Usage sometimes comes on the last choice chunk instead
                u = getattr(chunk, "usage", None)
                if u and (getattr(u, "prompt_tokens", 0) or 0) > 0:
                    usage["prompt_tokens"] = getattr(u, "prompt_tokens", 0) or 0
                    usage["completion_tokens"] = getattr(u, "completion_tokens", 0) or 0
            break   # stream consumed successfully — exit the retry loop
        except Exception as e:
            if use_usage_opt and "stream_options" in str(e).lower():
                continue   # endpoint rejected stream_options — retry without it
            raise          # real error — propagate to the caller's retry logic

    content = "".join(content_parts)

    # Reconstruct tool_calls as SimpleNamespace objects so downstream code
    # (tc.id, tc.function.name, tc.function.arguments) works unchanged.
    tool_calls = [
        SimpleNamespace(
            id=tc_acc[i]["id"] or f"call_{i}",
            function=SimpleNamespace(
                name=tc_acc[i]["name"],
                arguments=tc_acc[i]["arguments"],
            ),
        )
        for i in sorted(tc_acc.keys())
        if tc_acc[i]["name"]   # skip malformed empty-name entries
    ]
    return content, tool_calls, usage


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


def _compact_history(messages: List[dict]) -> None:
    """Elide stale tool-output bodies in place — bounded context.

    Re-sending every prior read_file/search dump each turn is what makes input
    tokens grow linearly with turns. Real agent loops keep only recent
    observations verbatim. We stub the CONTENT of all but the last N
    tool-result messages (role=='tool', or the text-fallback user messages that
    carry tool output), keeping a short head so the model recalls what it was
    and can re-read on demand. Structure is preserved (we never drop a message),
    so tool_call_id pairing stays valid. Applied identically to every arm.
    """
    keep = getattr(config, "CONTEXT_KEEP_LAST_TOOL_OUTPUTS", 0)
    if not keep:
        return
    over = getattr(config, "CONTEXT_STUB_OVER_CHARS", 600)
    tool_idxs = [
        i for i, m in enumerate(messages)
        if m.get("role") == "tool"
        or (m.get("role") == "user"
            and str(m.get("content", "")).startswith("[tool result"))
    ]
    for i in tool_idxs[:-keep] if keep else tool_idxs:
        c = messages[i].get("content", "")
        if isinstance(c, str) and len(c) > over and "[…elided" not in c:
            messages[i]["content"] = (
                c[:200].rstrip()
                + "\n[…elided to save context — re-read the file/search if needed]")


def run_react(task: dict, arm: str, executor: ToolExecutor,
              model: str = "main") -> Trajectory:
    """Run one task with one arm to completion. Returns the trajectory."""
    client = _client()
    traj = Trajectory(task_id=task["task_id"], arm=arm, model=model)
    t0 = time.time()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(issue=task["query"])},
    ]

    for step in range(config.MAX_TURNS):
        _compact_history(messages)   # bounded context — elide stale tool dumps
        ts = time.time()
        # Retry not just rate limits but ALSO transient server/connection errors
        # (timeouts, 5xx, overloaded). On NIM these were the main cause of lost
        # tasks: a single transient 500/timeout used to hard-fail the whole run,
        # so arms ended up with different task subsets (unpaired n). Treating them
        # as retryable keeps the task set complete and the comparison paired.
        for attempt in range(_MAX_API_ATTEMPTS):
            try:
                content, native_calls, usage_dict = _stream_completion(client, messages)
                break  # success
            except Exception as e:
                err_msg = f"{type(e).__name__}: {e}"
                last_attempt = attempt >= _MAX_API_ATTEMPTS - 1
                if _is_retryable(err_msg) and not last_attempt:
                    time.sleep(min(60, 5 * (2 ** attempt)))  # 5,10,20,40,60,60...
                    continue
                traj.stopped, traj.error = "error", err_msg
                break
        else:
            traj.stopped, traj.error = "error", "exhausted retries (transient)"

        if traj.stopped == "error":
            break

        turn = Turn(index=step, usage=_usage_from_dict(usage_dict),
                    text=content, latency_s=time.time() - ts)

        # Fallback: model emitted the call as text JSON, not structured tool_calls.
        text_calls = ([] if native_calls
                      else _parse_text_tool_calls(content))

        if not native_calls and not text_calls:
            # model answered without acting — nudge once, else stop
            turn.tool_calls = []
            traj.turns.append(turn)
            messages.append({"role": "assistant", "content": content})
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
                "content": content,
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
            messages.append({"role": "assistant", "content": content})
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
