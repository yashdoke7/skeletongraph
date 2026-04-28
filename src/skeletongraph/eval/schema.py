"""
Unified data schema for cross-agent evaluation traces.

Every parser (Antigravity, Claude Code, Cursor, Codex, Copilot) produces
an AgentTrace conforming to this schema, enabling apples-to-apples comparison.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional
import json


class AgentType(Enum):
    ANTIGRAVITY = "antigravity"
    CLAUDE_CODE = "claude_code"
    CURSOR = "cursor"
    CODEX = "codex"
    COPILOT = "copilot"


class TraceMode(Enum):
    SKELETONGRAPH = "skeletongraph"
    NATIVE = "native"


@dataclass
class ToolCall:
    """A single tool invocation by the agent."""
    tool_type: str          # "view_file", "grep_search", "query_context", "expand_function", etc.
    target: str             # File path, search query, or FQN
    output_tokens: int      # Measured token size of the tool's response
    line_range: str = ""    # e.g. "1-800" for view_file, "" if unknown
    timestamp: float = 0.0


@dataclass
class AgentResponse:
    """A single response block from the agent."""
    turn_id: int
    text: str
    token_count: int  # len(text) // 4


@dataclass
class AgentTrace:
    """Complete trace of an agent session for one task.

    This is the core data structure that every parser produces.
    It captures enough information to compute all 4 metric tiers:
      - Tier A: Retrieval Efficiency (tool output tokens only)
      - Tier B: Full Conversation Cost (tool + response + history)
      - Tier C: Turn Efficiency (action counts)
      - Tier D: Task Completion Quality (manual annotation)
    """
    # Identity
    agent: str                          # "antigravity", "claude_code", etc.
    mode: str                           # "skeletongraph" or "native"
    task_prompt: str                    # The original user request
    project: str = ""                   # e.g. "flask", "fastapi"
    repo_total_tokens: Optional[int] = None # Whole codebase size limit

    # Layer 1: Tool Output Tokens (retrieval cost)
    tool_calls: List[ToolCall] = field(default_factory=list)

    # Layer 2: Agent Response Tokens
    agent_responses: List[AgentResponse] = field(default_factory=list)

    # Layer 3: Reasoning Tokens (if available from API)
    reasoning_tokens: Optional[int] = None

    # Layer 5: MCP schema overhead (only non-zero for SG mode)
    # This is the token cost of loading SG tool schemas every turn.
    # For native runs this is 0. For SG runs we calculate it from
    # the number of turns x schema cost per turn.
    mcp_schema_overhead_tokens: int = 0

    # Layer 4: Quality annotations (filled manually or by judge)
    task_completed: Optional[bool] = None       # Did the fix work?
    files_modified: List[str] = field(default_factory=list)
    test_passed: Optional[bool] = None

    # Timing
    wall_clock_seconds: Optional[float] = None

    # ── Computed Properties ────────────────────────────────────────────

    @property
    def total_tool_output_tokens(self) -> int:
        """Layer 1: Sum of all tool response sizes."""
        return sum(tc.output_tokens for tc in self.tool_calls)

    @property
    def total_response_tokens(self) -> int:
        """Layer 2: Sum of all agent response sizes."""
        return sum(r.token_count for r in self.agent_responses)

    @property
    def total_turns(self) -> int:
        """Number of user→agent round trips."""
        return max(len(self.agent_responses), 1)

    @property
    def tool_call_count(self) -> int:
        return len(self.tool_calls)

    @property
    def view_file_count(self) -> int:
        return sum(1 for tc in self.tool_calls if tc.tool_type == "view_file")

    @property
    def grep_count(self) -> int:
        return sum(1 for tc in self.tool_calls if tc.tool_type == "grep_search")

    @property
    def sg_tool_count(self) -> int:
        return sum(1 for tc in self.tool_calls if tc.tool_type in (
            "query_context", "expand_function", "search_index",
            "show_graph", "get_blast_radius", "get_dependencies",
        ))

    @property
    def unique_files_viewed(self) -> int:
        return len({tc.target for tc in self.tool_calls if tc.tool_type == "view_file"})

    @property
    def repeated_file_views(self) -> int:
        return self.view_file_count - self.unique_files_viewed

    @property
    def estimated_history_tokens(self) -> int:
        """Layer 3: Compute cumulative history re-submission cost.

        Each turn, the LLM API receives ALL prior conversation content.
        This computes the total across all turns:
          Turn 1: 0 history
          Turn 2: content of turn 1
          Turn 3: content of turns 1+2
          ...
        """
        # Build per-turn content sizes
        turn_sizes: List[int] = []

        # Group tool calls by approximate turn ordering
        # For simplicity, distribute tool calls evenly across turns
        tools_per_turn = max(1, len(self.tool_calls) // max(self.total_turns, 1))

        for i in range(self.total_turns):
            turn_tool_tokens = sum(
                tc.output_tokens
                for tc in self.tool_calls[i * tools_per_turn:(i + 1) * tools_per_turn]
            )
            turn_response_tokens = (
                self.agent_responses[i].token_count
                if i < len(self.agent_responses) else 0
            )
            turn_sizes.append(turn_tool_tokens + turn_response_tokens)

        # Cumulative history: sum of all prior turns at each turn
        total_history = 0
        cumulative = 0
        for size in turn_sizes:
            total_history += cumulative
            cumulative += size

        return total_history

    @property
    def total_conversation_tokens(self) -> int:
        """Total measurable tokens: Layers 1 + 2 + 3 + 5.
        Layer 4 (reasoning) is additive but optional -- included
        separately in the report since it's only available for some agents.
        """
        return (
            self.total_tool_output_tokens       # L1
            + self.total_response_tokens        # L2
            + self.estimated_history_tokens     # L3
            + self.mcp_schema_overhead_tokens   # L5
        )

    # ── Serialization ──────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict for JSON export."""
        return {
            "agent": self.agent,
            "mode": self.mode,
            "task_prompt": self.task_prompt,
            "project": self.project,
            "repo_total_tokens": self.repo_total_tokens,
            "metrics": {
                "layer1_tool_output_tokens": self.total_tool_output_tokens,
                "layer2_response_tokens": self.total_response_tokens,
                "layer3_history_tokens": self.estimated_history_tokens,
                "layer4_reasoning_tokens": self.reasoning_tokens,
                "layer5_mcp_schema_overhead": self.mcp_schema_overhead_tokens,
                "total_conversation_tokens": self.total_conversation_tokens,
                "total_turns": self.total_turns,
                "total_tool_calls": self.tool_call_count,
                "view_file_count": self.view_file_count,
                "grep_count": self.grep_count,
                "sg_tool_count": self.sg_tool_count,
                "unique_files_viewed": self.unique_files_viewed,
                "repeated_file_views": self.repeated_file_views,
                "wall_clock_seconds": self.wall_clock_seconds,
            },
            "quality": {
                "task_completed": self.task_completed,
                "files_modified": self.files_modified,
                "test_passed": self.test_passed,
            },
            "tool_calls": [
                {"type": tc.tool_type, "target": tc.target, "tokens": tc.output_tokens}
                for tc in self.tool_calls
            ],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: dict) -> AgentTrace:
        """Deserialize from dict."""
        trace = cls(
            agent=data["agent"],
            mode=data["mode"],
            task_prompt=data["task_prompt"],
            project=data.get("project", ""),
            repo_total_tokens=data.get("repo_total_tokens"),
        )
        trace.task_completed = data.get("quality", {}).get("task_completed")
        trace.files_modified = data.get("quality", {}).get("files_modified", [])
        trace.test_passed = data.get("quality", {}).get("test_passed")
        trace.wall_clock_seconds = data.get("metrics", {}).get("wall_clock_seconds")
        trace.reasoning_tokens = data.get("metrics", {}).get("layer4_reasoning_tokens")
        trace.mcp_schema_overhead_tokens = data.get("metrics", {}).get("layer5_mcp_schema_overhead", 0)

        for tc_data in data.get("tool_calls", []):
            trace.tool_calls.append(ToolCall(
                tool_type=tc_data["type"],
                target=tc_data["target"],
                output_tokens=tc_data["tokens"],
            ))

        return trace
