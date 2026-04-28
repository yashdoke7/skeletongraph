"""
Session memory for cross-turn context deduplication.

Unique feature: No competitor tracks what context the agent already has.

The session tracks:
  1. Which FQNs were returned in previous turns
  2. Which full bodies (Zone 2) the agent already received
  3. Cumulative token savings (per-session and per-project)

This enables "caveman mode" — skip re-including what the agent already saw,
dramatically reducing tokens on follow-up queries.

Example:
  Turn 1: "fix validate_token" → 600 tokens (full context)
  Turn 2: "now add tests" → 250 tokens (skips validate_token body,
           includes test patterns instead)
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set


@dataclass
class TurnRecord:
    """Record of a single query turn."""
    turn_id: int
    timestamp: float
    prompt: str
    fqns_returned: Set[str]             # All FQNs in the response
    zone2_fqns: Set[str]                # FQNs with full body (Zone 2)
    token_count: int                     # Tokens in the assembled context
    estimated_native_tokens: int         # Estimated if agent used grep/read
    confidence: str = ""
    # Agent response captured via previous_response parameter (L2 data)
    response_text: str = ""
    response_tokens: int = 0


@dataclass
class SessionStats:
    """Cumulative session statistics."""
    total_turns: int = 0
    total_sg_tokens: int = 0
    total_native_tokens_estimated: int = 0
    total_saved_tokens: int = 0

    @property
    def reduction_ratio(self) -> float:
        if self.total_sg_tokens == 0:
            return 0.0
        return self.total_native_tokens_estimated / self.total_sg_tokens

    @property
    def estimated_cost_saved_usd(self) -> float:
        """Estimate cost savings at GPT-4o input pricing ($2.50/1M tokens)."""
        return (self.total_saved_tokens / 1_000_000) * 2.50

    def to_dict(self) -> dict:
        return {
            "total_turns": self.total_turns,
            "total_sg_tokens": self.total_sg_tokens,
            "total_native_tokens_estimated": self.total_native_tokens_estimated,
            "total_saved_tokens": self.total_saved_tokens,
            "reduction_ratio": round(self.reduction_ratio, 1),
            "estimated_cost_saved_usd": round(self.estimated_cost_saved_usd, 4),
        }


class Session:
    """Cross-turn session memory manager.

    Persists to .skeletongraph/session/ for continuity across
    MCP server restarts (within the TTL window).
    """

    def __init__(self, ttl_minutes: int = 60, max_turns: int = 50) -> None:
        self._ttl_seconds = ttl_minutes * 60
        self._max_turns = max_turns
        self._turns: List[TurnRecord] = []
        self._stats = SessionStats()
        self._created_at = time.time()

    @property
    def is_expired(self) -> bool:
        return (time.time() - self._created_at) > self._ttl_seconds

    @property
    def turn_count(self) -> int:
        return len(self._turns)

    @property
    def stats(self) -> SessionStats:
        return self._stats

    def get_known_fqns(self) -> Set[str]:
        """Get all FQNs the agent has seen across all turns."""
        known: Set[str] = set()
        for turn in self._turns:
            known.update(turn.fqns_returned)
        return known

    def get_zone2_fqns(self) -> Set[str]:
        """Get FQNs where the agent already has the full body."""
        bodies: Set[str] = set()
        for turn in self._turns:
            bodies.update(turn.zone2_fqns)
        return bodies

    def record_turn(
        self,
        prompt: str,
        fqns_returned: Set[str],
        zone2_fqns: Set[str],
        token_count: int,
        estimated_native_tokens: int,
        confidence: str = "",
    ) -> TurnRecord:
        """Record a completed query turn."""
        turn = TurnRecord(
            turn_id=len(self._turns) + 1,
            timestamp=time.time(),
            prompt=prompt,
            fqns_returned=fqns_returned,
            zone2_fqns=zone2_fqns,
            token_count=token_count,
            estimated_native_tokens=estimated_native_tokens,
            confidence=confidence,
        )
        self._turns.append(turn)

        # Update stats
        self._stats.total_turns += 1
        self._stats.total_sg_tokens += token_count
        self._stats.total_native_tokens_estimated += estimated_native_tokens
        self._stats.total_saved_tokens += max(
            0, estimated_native_tokens - token_count
        )

        # Trim old turns if over max
        if len(self._turns) > self._max_turns:
            self._turns = self._turns[-self._max_turns:]

        return turn

    def should_skip_body(self, fqn: str) -> bool:
        """Check if the agent already has this function's full body.

        Used by the zone assembler to implement "caveman mode":
        if the agent already received the full body in a previous turn,
        send only the signature + "Body already provided in turn N".
        """
        return fqn in self.get_zone2_fqns()

    def get_last_prompt(self) -> Optional[str]:
        """Get the most recent prompt (for anaphora resolution)."""
        if self._turns:
            return self._turns[-1].prompt
        return None

    def get_last_target_fqns(self) -> Set[str]:
        """Get FQNs from the most recent turn (for 'it'/'that' resolution)."""
        if self._turns:
            return self._turns[-1].zone2_fqns
        return set()

    def reset(self) -> None:
        """Clear session state."""
        self._turns.clear()
        self._stats = SessionStats()
        self._created_at = time.time()

    # ── Persistence ────────────────────────────────────────────────────

    def save(self, project_root: Path) -> None:
        """Save session state to .skeletongraph/session/."""
        session_dir = project_root / ".skeletongraph" / "session"
        session_dir.mkdir(parents=True, exist_ok=True)

        data = {
            "created_at": self._created_at,
            "stats": self._stats.to_dict(),
            "turns": [
                {
                    "turn_id": t.turn_id,
                    "timestamp": t.timestamp,
                    "prompt": t.prompt,
                    "fqns_returned": sorted(t.fqns_returned),
                    "zone2_fqns": sorted(t.zone2_fqns),
                    "token_count": t.token_count,
                    "estimated_native_tokens": t.estimated_native_tokens,
                    "confidence": t.confidence,
                    "response_text": t.response_text,
                    "response_tokens": t.response_tokens,
                }
                for t in self._turns[-10:]  # Only persist last 10 turns
            ],
        }

        path = session_dir / "current.json"
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, project_root: Path, ttl_minutes: int = 60) -> Session:
        """Load session from disk, or create new if expired/missing."""
        session = cls(ttl_minutes=ttl_minutes)
        path = project_root / ".skeletongraph" / "session" / "current.json"

        if not path.exists():
            return session

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            session._created_at = data.get("created_at", time.time())

            # Check expiry
            if session.is_expired:
                return cls(ttl_minutes=ttl_minutes)

            # Restore stats
            stats = data.get("stats", {})
            session._stats = SessionStats(
                total_turns=stats.get("total_turns", 0),
                total_sg_tokens=stats.get("total_sg_tokens", 0),
                total_native_tokens_estimated=stats.get("total_native_tokens_estimated", 0),
                total_saved_tokens=stats.get("total_saved_tokens", 0),
            )

            # Restore turns
            for t in data.get("turns", []):
                session._turns.append(TurnRecord(
                    turn_id=t["turn_id"],
                    timestamp=t["timestamp"],
                    prompt=t["prompt"],
                    fqns_returned=set(t.get("fqns_returned", [])),
                    zone2_fqns=set(t.get("zone2_fqns", [])),
                    token_count=t.get("token_count", 0),
                    estimated_native_tokens=t.get("estimated_native_tokens", 0),
                    confidence=t.get("confidence", ""),
                    response_text=t.get("response_text", ""),
                    response_tokens=t.get("response_tokens", 0),
                ))

        except (json.JSONDecodeError, KeyError, OSError):
            return cls(ttl_minutes=ttl_minutes)

        return session
