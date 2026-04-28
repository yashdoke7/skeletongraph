"""
Evaluation scoring engine.

Computes all quality metrics from raw measurement data:
  - Token efficiency ratios
  - Precision / Recall / F1 for file retrieval  
  - Mean Reciprocal Rank (MRR) and Hit@k
  - Aggregate statistics with confidence intervals
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Set, Optional


# ── Token Efficiency ──────────────────────────────────────────────

@dataclass
class TokenEfficiencyScore:
    """Token efficiency metrics for one task."""
    task_id: str
    native_retrieval_tokens: int
    sg_retrieval_tokens: int
    native_total_tokens: int
    sg_total_tokens: int
    codebase_tokens: int = 0
    
    @property
    def retrieval_reduction_ratio(self) -> float:
        """How many times smaller SG retrieval is vs native."""
        if self.sg_retrieval_tokens == 0:
            return 0.0
        return self.native_retrieval_tokens / self.sg_retrieval_tokens
    
    @property
    def total_reduction_ratio(self) -> float:
        """How many times smaller SG total conversation is vs native."""
        if self.sg_total_tokens == 0:
            return 0.0
        return self.native_total_tokens / self.sg_total_tokens
    
    @property
    def native_cost_usd(self) -> float:
        """Estimated API cost for native agent at $3.00/1M input tokens."""
        return (self.native_total_tokens / 1_000_000) * 3.00
    
    @property
    def sg_cost_usd(self) -> float:
        """Estimated API cost for SG agent at $3.00/1M input tokens."""
        return (self.sg_total_tokens / 1_000_000) * 3.00
    
    @property
    def cost_savings_pct(self) -> float:
        """Percentage cost saved by using SG."""
        if self.native_cost_usd == 0:
            return 0.0
        return ((self.native_cost_usd - self.sg_cost_usd) / self.native_cost_usd) * 100.0


# ── Retrieval Quality ─────────────────────────────────────────────

@dataclass 
class RetrievalQualityScore:
    """P/R/F1/MRR metrics for file retrieval on one task."""
    task_id: str
    ground_truth_files: List[str]
    retrieved_files: List[str]          # Ordered by retrieval order
    
    @property
    def ground_truth_set(self) -> Set[str]:
        return set(self.ground_truth_files)
    
    @property
    def retrieved_set(self) -> Set[str]:
        return set(self.retrieved_files)
    
    @property
    def true_positives(self) -> int:
        return len(self.retrieved_set & self.ground_truth_set)
    
    @property
    def false_positives(self) -> int:
        return len(self.retrieved_set - self.ground_truth_set)
    
    @property
    def false_negatives(self) -> int:
        return len(self.ground_truth_set - self.retrieved_set)
    
    @property
    def precision(self) -> float:
        """What fraction of retrieved files were relevant?"""
        total = self.true_positives + self.false_positives
        return self.true_positives / total if total > 0 else 0.0
    
    @property
    def recall(self) -> float:
        """What fraction of relevant files were retrieved?"""
        total = self.true_positives + self.false_negatives
        return self.true_positives / total if total > 0 else 0.0
    
    @property
    def f1(self) -> float:
        """Harmonic mean of precision and recall."""
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    
    @property
    def mrr(self) -> float:
        """Mean Reciprocal Rank: 1/rank of first relevant file retrieved."""
        for i, f in enumerate(self.retrieved_files):
            if f in self.ground_truth_set:
                return 1.0 / (i + 1)
        return 0.0
    
    def hit_at_k(self, k: int = 3) -> bool:
        """Was at least one relevant file in the first k retrievals?"""
        top_k = set(self.retrieved_files[:k])
        return bool(top_k & self.ground_truth_set)
    
    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "mrr": round(self.mrr, 4),
            "hit_at_3": self.hit_at_k(3),
            "hit_at_5": self.hit_at_k(5),
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "ground_truth_count": len(self.ground_truth_files),
            "retrieved_count": len(self.retrieved_files),
        }


# ── Aggregate Statistics ──────────────────────────────────────────

def compute_aggregates(values: List[float]) -> dict:
    """Compute mean, median, std, min, max, and 95% CI for a list of values.
    
    Returns dict with all aggregate statistics.
    """
    if not values:
        return {
            "mean": 0.0, "median": 0.0, "std": 0.0,
            "min": 0.0, "max": 0.0, "ci_95_lower": 0.0, "ci_95_upper": 0.0,
            "count": 0,
        }
    
    n = len(values)
    sorted_vals = sorted(values)
    mean = sum(values) / n
    
    # Median
    if n % 2 == 0:
        median = (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2
    else:
        median = sorted_vals[n // 2]
    
    # Standard deviation 
    variance = sum((x - mean) ** 2 for x in values) / max(n - 1, 1)
    std = math.sqrt(variance)
    
    # 95% confidence interval (using t-distribution approximation)
    # For n >= 30, z ≈ 1.96; for smaller n, we use a rougher approximation
    z = 1.96 if n >= 30 else 2.0
    margin = z * std / math.sqrt(n)
    
    return {
        "mean": round(mean, 4),
        "median": round(median, 4),
        "std": round(std, 4),
        "min": round(min(values), 4),
        "max": round(max(values), 4),
        "ci_95_lower": round(mean - margin, 4),
        "ci_95_upper": round(mean + margin, 4),
        "count": n,
    }


def aggregate_retrieval_scores(scores: List[RetrievalQualityScore]) -> dict:
    """Aggregate multiple per-task retrieval scores into summary statistics."""
    if not scores:
        return {}
    
    return {
        "precision": compute_aggregates([s.precision for s in scores]),
        "recall": compute_aggregates([s.recall for s in scores]),
        "f1": compute_aggregates([s.f1 for s in scores]),
        "mrr": compute_aggregates([s.mrr for s in scores]),
        "hit_at_3_rate": sum(1 for s in scores if s.hit_at_k(3)) / len(scores),
        "hit_at_5_rate": sum(1 for s in scores if s.hit_at_k(5)) / len(scores),
        "task_count": len(scores),
    }


def aggregate_token_scores(scores: List[TokenEfficiencyScore]) -> dict:
    """Aggregate multiple per-task token efficiency scores into summary statistics."""
    if not scores:
        return {}
    
    return {
        "retrieval_reduction_ratio": compute_aggregates(
            [s.retrieval_reduction_ratio for s in scores]
        ),
        "total_reduction_ratio": compute_aggregates(
            [s.total_reduction_ratio for s in scores]
        ),
        "native_retrieval_tokens": compute_aggregates(
            [float(s.native_retrieval_tokens) for s in scores]
        ),
        "sg_retrieval_tokens": compute_aggregates(
            [float(s.sg_retrieval_tokens) for s in scores]
        ),
        "native_cost_usd": compute_aggregates(
            [s.native_cost_usd for s in scores]
        ),
        "sg_cost_usd": compute_aggregates(
            [s.sg_cost_usd for s in scores]
        ),
        "cost_savings_pct": compute_aggregates(
            [s.cost_savings_pct for s in scores]
        ),
        "task_count": len(scores),
    }
