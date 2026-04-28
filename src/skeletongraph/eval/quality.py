"""
Execution quality metrics for the SkeletonGraph evaluation framework.

Computes Precision, Recall, and F1 scores by comparing the files retrieved
during an agent session against the human-edited ground truth files from 
historical Pull Requests.
"""

from typing import List, Set
from pathlib import Path
from dataclasses import dataclass
from .schema import AgentTrace


@dataclass
class RetrievalQuality:
    """Precision/Recall metrics for file retrieval."""
    true_positives: int
    false_positives: int
    false_negatives: int

    @property
    def precision(self) -> float:
        """Percentage of retrieved files that were actually relevant."""
        total_retrieved = self.true_positives + self.false_positives
        return self.true_positives / total_retrieved if total_retrieved > 0 else 0.0

    @property
    def recall(self) -> float:
        """Percentage of relevant files that were successfully retrieved."""
        total_relevant = self.true_positives + self.false_negatives
        return self.true_positives / total_relevant if total_relevant > 0 else 0.0

    @property
    def f1_score(self) -> float:
        """Harmonic mean of precision and recall."""
        p = self.precision
        r = self.recall
        return 2 * (p * r) / (p + r) if (p + r) > 0 else 0.0


def evaluate_retrieval_quality(trace: AgentTrace, ground_truth_files: List[str]) -> RetrievalQuality:
    """
    Calculate precision and recall for an agent's file retrieval.
    
    Args:
        trace: The parsed AgentTrace containing the tool calls.
        ground_truth_files: List of file paths modified in the ground truth PR.
        
    Returns:
        RetrievalQuality object containing TP, FP, FN, and F1 score.
    """
    # 1. Normalize ground truth files
    truth_set: Set[str] = {Path(f).name for f in ground_truth_files}
    
    # 2. Extract uniquely retrieved files from the trace
    # For native agents, we check the target of view_file and patch
    # For SG, we check the zone targets (which we store in tool_calls target)
    retrieved_set: Set[str] = set()
    for tc in trace.tool_calls:
        # Simplistic extraction: try to parse filenames out of the target string
        target_name = Path(tc.target.split('#')[0]).name  # handle file.py#10-20
        if target_name and '.' in target_name: # heuristic for real files
            retrieved_set.add(target_name)

    # 3. Calculate metrics
    true_positives = len(retrieved_set.intersection(truth_set))
    false_positives = len(retrieved_set - truth_set)
    false_negatives = len(truth_set - retrieved_set)

    return RetrievalQuality(
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives
    )
