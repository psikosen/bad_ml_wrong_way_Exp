"""Evaluation metrics for GAB experiments.

Implements:
  - Action accuracy / macro-F1
  - Plan exact-match and edit distance
  - Retrieval Recall@k and MRR
  - Agent success rate and tool call statistics
"""

from collections import Counter
from typing import Dict, List, Tuple


def accuracy(predictions: List, labels: List) -> float:
    if not labels:
        return 0.0
    correct = sum(1 for p, l in zip(predictions, labels) if p == l)
    return correct / len(labels)


def macro_f1(predictions: List, labels: List, classes: List) -> float:
    """Compute macro-averaged F1 across all classes."""
    f1s = []
    for cls in classes:
        tp = sum(1 for p, l in zip(predictions, labels) if p == cls and l == cls)
        fp = sum(1 for p, l in zip(predictions, labels) if p == cls and l != cls)
        fn = sum(1 for p, l in zip(predictions, labels) if p != cls and l == cls)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        f1s.append(f1)
    return sum(f1s) / len(f1s) if f1s else 0.0


def plan_exact_match(pred_plans: List[List[str]], gold_plans: List[List[str]]) -> float:
    """Fraction of plans that match exactly."""
    if not gold_plans:
        return 0.0
    correct = sum(1 for p, g in zip(pred_plans, gold_plans) if p == g)
    return correct / len(gold_plans)


def edit_distance(a: List[str], b: List[str]) -> int:
    """Levenshtein edit distance between two token sequences."""
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[m][n]


def avg_plan_edit_distance(pred_plans: List[List[str]], gold_plans: List[List[str]]) -> float:
    """Average edit distance between predicted and gold plans."""
    if not gold_plans:
        return 0.0
    total = sum(edit_distance(p, g) for p, g in zip(pred_plans, gold_plans))
    return total / len(gold_plans)


def recall_at_k(retrieved_ids: List[List[str]], relevant_ids: List[List[str]], k: int = 5) -> float:
    """Recall@k: fraction of relevant items found in top-k retrieved."""
    if not relevant_ids:
        return 0.0
    recalls = []
    for ret, rel in zip(retrieved_ids, relevant_ids):
        top_k = set(ret[:k])
        rel_set = set(rel)
        if rel_set:
            recalls.append(len(top_k & rel_set) / len(rel_set))
    return sum(recalls) / len(recalls) if recalls else 0.0


def mrr(retrieved_ids: List[List[str]], relevant_ids: List[List[str]]) -> float:
    """Mean Reciprocal Rank."""
    if not relevant_ids:
        return 0.0
    rrs = []
    for ret, rel in zip(retrieved_ids, relevant_ids):
        rel_set = set(rel)
        rr = 0.0
        for rank, rid in enumerate(ret, 1):
            if rid in rel_set:
                rr = 1.0 / rank
                break
        rrs.append(rr)
    return sum(rrs) / len(rrs) if rrs else 0.0


def agent_success_rate(results: List[bool]) -> float:
    """Fraction of episodes where the agent succeeded."""
    if not results:
        return 0.0
    return sum(results) / len(results)


def avg_tool_calls(tool_counts: List[int]) -> float:
    """Average number of tool calls per episode."""
    if not tool_counts:
        return 0.0
    return sum(tool_counts) / len(tool_counts)
