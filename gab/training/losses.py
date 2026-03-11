"""Loss functions for Graph-Agent Brain training.

Implements:
  - Supervised task loss (action CE + plan CE)
  - Entropy regularization (exploration)
  - Memory contrastive loss (InfoNCE-style)
  - Distillation loss (KL with temperature)
  - Tool cost penalty
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def action_cross_entropy(action_logits: torch.Tensor, action_labels: torch.Tensor) -> torch.Tensor:
    """Standard cross-entropy for action prediction. L_action = -E[log pi(a|x)]"""
    return F.cross_entropy(action_logits, action_labels)


def plan_cross_entropy(
    plan_logits: torch.Tensor,
    plan_labels: torch.Tensor,
    plan_length: int = 5,
    plan_vocab_size: int = 10,
) -> torch.Tensor:
    """Teacher-forced plan token loss over first T positions.

    Args:
        plan_logits: [B, L, plan_vocab_size]
        plan_labels: [B, T] plan token IDs
        plan_length: T, number of plan positions to score
        plan_vocab_size: size of plan token vocabulary
    """
    logits_T = plan_logits[:, :plan_length, :].reshape(-1, plan_vocab_size)
    labels_T = plan_labels.reshape(-1)
    return F.cross_entropy(logits_T, labels_T)


def supervised_task_loss(
    action_logits: torch.Tensor,
    action_labels: torch.Tensor,
    plan_logits: torch.Tensor,
    plan_labels: torch.Tensor,
    lambda_plan: float = 0.5,
    plan_length: int = 5,
    plan_vocab_size: int = 10,
) -> torch.Tensor:
    """L_task = L_action + lambda_plan * L_plan"""
    l_action = action_cross_entropy(action_logits, action_labels)
    l_plan = plan_cross_entropy(plan_logits, plan_labels, plan_length, plan_vocab_size)
    return l_action + lambda_plan * l_plan


def entropy_regularization(action_logits: torch.Tensor) -> torch.Tensor:
    """Negative entropy of action distribution (minimizing this increases entropy).

    L_explore = -H(pi(.|x)) = sum p log p
    """
    probs = F.softmax(action_logits, dim=-1)
    log_probs = F.log_softmax(action_logits, dim=-1)
    return (probs * log_probs).sum(dim=-1).mean()


def memory_contrastive_loss(
    query_emb: torch.Tensor,
    positive_emb: torch.Tensor,
    negative_embs: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """InfoNCE-style contrastive loss for memory retrieval grounding.

    Args:
        query_emb: [B, D] query representations
        positive_emb: [B, D] required memory node embeddings
        negative_embs: [B, N, D] negative memory embeddings
        temperature: softmax temperature tau
    """
    # Positive similarity: [B]
    pos_sim = (query_emb * positive_emb).sum(dim=-1) / temperature
    # Negative similarities: [B, N]
    neg_sim = torch.bmm(negative_embs, query_emb.unsqueeze(-1)).squeeze(-1) / temperature
    # InfoNCE: -log(exp(pos) / (exp(pos) + sum(exp(neg))))
    logits = torch.cat([pos_sim.unsqueeze(-1), neg_sim], dim=-1)  # [B, 1+N]
    labels = torch.zeros(query_emb.size(0), dtype=torch.long, device=query_emb.device)
    return F.cross_entropy(logits, labels)


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 2.0,
) -> torch.Tensor:
    """KL divergence with softened distributions for teacher-student distillation.

    L_distill = T^2 * KL(sigma(z_teacher/T) || sigma(z_student/T))
    """
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
    kl = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")
    return temperature * temperature * kl


def tool_cost_penalty(actions: torch.Tensor, tool_action_ids: list, cost_per_call: float = 0.1) -> torch.Tensor:
    """Penalize tool usage: L_toolcost = sum c(a_t).

    Args:
        actions: [B] sampled action IDs
        tool_action_ids: list of action IDs that count as tool calls
        cost_per_call: cost per tool invocation
    """
    mask = torch.zeros_like(actions, dtype=torch.float)
    for tid in tool_action_ids:
        mask += (actions == tid).float()
    return (mask * cost_per_call).mean()
