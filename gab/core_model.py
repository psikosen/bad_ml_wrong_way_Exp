"""TinyTransformer-0.6M: the neural core for Graph-Agent Brain.

Architecture:
  - vocab_size=2960, max_len=128, d_model=96, nhead=4, num_layers=4, d_ff=192
  - Action head: d_model -> num_actions
  - Plan head: d_model -> plan_vocab_size (per-position)
  - Value head: d_model -> 1 (for RL)

Total parameters: ~599,462 (~0.6M)
"""

import torch
import torch.nn as nn


# Action vocabulary
ACTIONS = ["SEARCH", "MEMORY_READ", "MEMORY_WRITE", "EXEC", "TEACHER", "ANSWER"]
ACTION2ID = {a: i for i, a in enumerate(ACTIONS)}
ID2ACTION = {i: a for a, i in ACTION2ID.items()}

# Plan token vocabulary
PLAN_VOCAB = [
    "PAD", "PLAN", "SEARCH", "MEMORY_READ", "MEMORY_WRITE",
    "EXEC", "TEACHER", "CHECK", "ANSWER", "END",
]
PLAN2ID = {t: i for i, t in enumerate(PLAN_VOCAB)}
ID2PLAN = {i: t for t, i in PLAN2ID.items()}


class TinyTransformerCore(nn.Module):
    """0.6M-parameter transformer core for action selection, plan generation, and value estimation."""

    def __init__(
        self,
        vocab_size: int = 2960,
        max_len: int = 128,
        d_model: int = 96,
        nhead: int = 4,
        num_layers: int = 4,
        d_ff: int = 192,
        num_actions: int = 6,
        plan_vocab_size: int = 10,
    ):
        super().__init__()
        self.d_model = d_model

        # Embeddings
        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_len, d_model)

        # Transformer encoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_ff,
            dropout=0.0,
            activation="relu",
            batch_first=True,
            norm_first=True,
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.final_ln = nn.LayerNorm(d_model)

        # Output heads
        self.action_head = nn.Linear(d_model, num_actions)
        self.plan_head = nn.Linear(d_model, plan_vocab_size)
        self.value_head = nn.Linear(d_model, 1)

    def forward(self, input_ids: torch.Tensor):
        """Forward pass.

        Args:
            input_ids: [B, L] token IDs

        Returns:
            action_logits: [B, num_actions]
            plan_logits: [B, L, plan_vocab_size]
            value: [B]
        """
        B, L = input_ids.shape
        pos_ids = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, L)
        x = self.tok(input_ids) + self.pos(pos_ids)
        h = self.enc(x)
        h = self.final_ln(h)
        cls = h[:, 0, :]  # first token pooling
        action_logits = self.action_head(cls)
        value = self.value_head(cls).squeeze(-1)
        plan_logits = self.plan_head(h)  # [B, L, plan_vocab]
        return action_logits, plan_logits, value

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
