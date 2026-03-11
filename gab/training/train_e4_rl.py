"""RL training pipeline for E4: Execution + Feedback Loop.

Uses REINFORCE with baseline for policy gradient optimization.
"""

import random
from typing import Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW

from gab.tools.toy_executor import ToyExecutor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ACTIONS = ["EXEC", "MEMORY_READ", "ANSWER"]
A2I = {a: i for i, a in enumerate(ACTIONS)}


class TinyPolicy(nn.Module):
    def __init__(self, vocab: int = 512, d: int = 64):
        super().__init__()
        self.vocab = vocab
        self.emb = nn.Embedding(vocab, d)
        self.fc = nn.Linear(d, len(ACTIONS))

    def encode(self, text: str, max_len: int = 8) -> torch.Tensor:
        toks = text.lower().split()
        ids = [(hash(t) % (self.vocab - 1)) + 1 for t in toks][:max_len]
        ids += [0] * (max_len - len(ids))
        return torch.tensor(ids, dtype=torch.long)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        h = self.emb(ids).mean(dim=0)
        return self.fc(h)


def env_sample() -> Tuple[str, int]:
    a = random.randint(1, 9)
    b = random.randint(1, 9)
    op = random.choice(["+", "*"])
    task = f"compute {a} {op} {b}"
    gold = a + b if op == "+" else a * b
    return task, gold


def train_reinforce(steps: int = 2000, seed: int = 7, lr: float = 1e-3):
    random.seed(seed)
    torch.manual_seed(seed)

    policy = TinyPolicy().to(DEVICE)
    executor = ToyExecutor()
    opt = AdamW(policy.parameters(), lr=lr)
    baseline = 0.0
    beta = 0.99

    for t in range(steps):
        task, gold = env_sample()
        ids = policy.encode(task).to(DEVICE)
        logits = policy(ids)
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs=probs)
        action = dist.sample()
        logp = dist.log_prob(action)

        if ACTIONS[action.item()] == "EXEC":
            output, _, success = executor.execute(task)
            # Verify correctness
            if success and str(gold) in output:
                reward = 1.0
            else:
                reward = 0.0
        else:
            reward = 0.0

        baseline = beta * baseline + (1 - beta) * reward
        adv = reward - baseline
        loss = -logp * adv

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if (t + 1) % 200 == 0:
            print(
                f"step {t + 1}  reward={reward:.1f}  baseline={baseline:.3f}  "
                f"probs={probs.detach().cpu().numpy()}"
            )

    return policy


if __name__ == "__main__":
    train_reinforce()
