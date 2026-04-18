"""
Elastic token budget manager.

Dynamic, not static. Zones expand/contract based on actual task needs.
Zone 1 (constraints) and Zone 4 (prompt) are non-negotiable.
Zone 2 (target code) gets what it needs.
Zone 3 (structural context) is the elastic pressure valve.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List


class Zone3Mode(Enum):
    """How much detail Zone 3 gets based on budget pressure."""
    FULL = "full"         # Signature + summary + decorators (~25 tok each)
    COMPACT = "compact"   # Signature only (~12 tok each)
    MINIMAL = "minimal"   # FQN + return type only (~8 tok each)
    NONE = "none"         # No Zone 3 at all


@dataclass
class Allocation:
    """Result of a budget allocation decision."""
    zone1_tokens: int       # Constraints (non-negotiable)
    zone2_tokens: int       # Target code (Tier 1 bodies)
    zone3_budget: int       # Available for structural context
    zone3_mode: Zone3Mode   # Detail level for Zone 3
    zone4_tokens: int       # Prompt (non-negotiable)
    total_tokens: int       # Total allocated
    utilization: float      # Fraction of hard limit used (0.0 - 1.0)
    warning: str = ""       # Warning if budget is tight


class TokenBudget:
    """Elastic budget manager for zone-based context assembly.

    Design principle: NEVER truncate critical code. Instead, compress
    peripheral context (Zone 3) progressively.

    Expansion tiers:
      1. Under soft target → Zone 3 gets full detail
      2. Soft target - 50% hard → Zone 3 goes compact
      3. 50% - 90% hard → Zone 3 goes minimal
      4. Over 90% hard → Zone 3 is dropped, warning emitted
    """

    def __init__(self, model_context_limit: int = 128_000) -> None:
        self.hard_limit = model_context_limit
        self.soft_target = int(model_context_limit * 0.25)  # Aim for 25% usage

    def allocate(
        self,
        zone1_tokens: int,
        zone2_tokens: int,
        zone3_candidates_count: int,
        zone4_tokens: int,
    ) -> Allocation:
        """Compute budget allocation across zones.

        Args:
            zone1_tokens: Tokens needed for constraints (always included).
            zone2_tokens: Tokens needed for target code bodies (always included).
            zone3_candidates_count: Number of Zone 3 skeleton entries available.
            zone4_tokens: Tokens needed for the prompt (always included).

        Returns:
            Allocation with budget per zone and Zone 3 detail mode.
        """
        # Non-negotiable: Zone 1 + Zone 2 + Zone 4
        fixed_cost = zone1_tokens + zone2_tokens + zone4_tokens
        warning = ""

        if fixed_cost < self.soft_target:
            # Under soft target — Zone 3 gets generous allocation
            zone3_budget = self.soft_target - fixed_cost
            zone3_mode = Zone3Mode.FULL

        elif fixed_cost < int(self.hard_limit * 0.5):
            # Between soft and 50% hard — Zone 3 gets compact
            zone3_budget = int(self.hard_limit * 0.5) - fixed_cost
            zone3_mode = Zone3Mode.COMPACT

        elif fixed_cost < int(self.hard_limit * 0.9):
            # Tight — Zone 3 gets minimal
            zone3_budget = min(
                int(self.hard_limit * 0.1),
                int(self.hard_limit * 0.9) - fixed_cost,
            )
            zone3_mode = Zone3Mode.MINIMAL
            warning = "Budget tight. Zone 3 reduced to minimal (FQN + return type)."

        else:
            # At hard limit — Zone 3 dropped
            zone3_budget = 0
            zone3_mode = Zone3Mode.NONE
            warning = (
                f"Target code alone uses {fixed_cost} tokens "
                f"({fixed_cost * 100 // self.hard_limit}% of {self.hard_limit} limit). "
                f"Zone 3 context dropped. Consider using read() for specific sections."
            )

        total = fixed_cost + zone3_budget

        return Allocation(
            zone1_tokens=zone1_tokens,
            zone2_tokens=zone2_tokens,
            zone3_budget=zone3_budget,
            zone3_mode=zone3_mode,
            zone4_tokens=zone4_tokens,
            total_tokens=total,
            utilization=total / self.hard_limit,
            warning=warning,
        )

    def estimate_zone3_entries(self, budget: int, mode: Zone3Mode) -> int:
        """How many Zone 3 entries fit in the given budget."""
        tokens_per_entry = {
            Zone3Mode.FULL: 25,
            Zone3Mode.COMPACT: 12,
            Zone3Mode.MINIMAL: 8,
            Zone3Mode.NONE: 0,
        }
        per_entry = tokens_per_entry[mode]
        if per_entry == 0:
            return 0
        return budget // per_entry
