"""Supervised training pipeline for E2: Tiny Neural Planner.

Trains the 0.6M TinyTransformerCore on action classification + plan prediction.
"""

import random
from typing import List

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm

from gab.core_model import ACTIONS, PLAN2ID, PLAN_VOCAB, TinyTransformerCore
from gab.tokenization import HashTokenizer
from gab.training.losses import supervised_task_loss


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Plan templates per action
PLAN_TEMPLATES = {
    "SEARCH": ["PLAN", "SEARCH", "CHECK", "ANSWER", "END"],
    "MEMORY_READ": ["PLAN", "MEMORY_READ", "CHECK", "ANSWER", "END"],
    "MEMORY_WRITE": ["PLAN", "MEMORY_WRITE", "CHECK", "ANSWER", "END"],
    "EXEC": ["PLAN", "EXEC", "CHECK", "ANSWER", "END"],
    "TEACHER": ["PLAN", "TEACHER", "CHECK", "ANSWER", "END"],
    "ANSWER": ["PLAN", "ANSWER", "END", "PAD", "PAD"],
}


def generate_dataset(n: int = 60000, seed: int = 7):
    random.seed(seed)
    topics = ["grep flags", "awk syntax", "regex groups", "file permissions", "loops", "json schema"]
    templates = {
        "SEARCH": ["look up {t}", "find docs {t}", "search web for {t}"],
        "MEMORY_READ": ["recall {t}", "retrieve memory {t}", "what did we store about {t}"],
        "MEMORY_WRITE": ["save note about {t}", "store this fact about {t}", "write to memory {t}"],
        "EXEC": ["run test for {t}", "execute to verify {t}", "compute and check {t}"],
        "TEACHER": ["ask teacher {t}", "consult teacher {t}", "teacher help {t}"],
        "ANSWER": ["answer now {t}", "respond briefly {t}", "give short answer {t}"],
    }

    data = []
    for _ in range(n):
        y = random.choice(ACTIONS)
        x = random.choice(templates[y]).format(t=random.choice(topics))
        if random.random() < 0.3:
            x = x.replace("look up", "find").replace("answer now", "respond now")
        if random.random() < 0.2:
            x = "please " + x
        plan_ids = [PLAN2ID[t] for t in PLAN_TEMPLATES[y]]
        data.append((x, ACTIONS.index(y), plan_ids))

    random.shuffle(data)
    return data


def batchify(data, tok: HashTokenizer, batch_size: int = 128):
    for i in range(0, len(data), batch_size):
        batch = data[i : i + batch_size]
        x = torch.tensor([tok.encode(e[0]) for e in batch], dtype=torch.long)
        y = torch.tensor([e[1] for e in batch], dtype=torch.long)
        plan = torch.tensor([e[2] for e in batch], dtype=torch.long)
        yield x, y, plan


def train(epochs: int = 3, lr: float = 3e-4, batch_size: int = 128, seed: int = 7):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    data = generate_dataset()
    split = int(0.9 * len(data))
    train_data, test_data = data[:split], data[split:]

    tok = HashTokenizer(vocab_size=2960, max_len=128)
    model = TinyTransformerCore(
        vocab_size=2960,
        max_len=128,
        d_model=96,
        nhead=4,
        num_layers=4,
        d_ff=192,
        num_actions=len(ACTIONS),
        plan_vocab_size=len(PLAN_VOCAB),
    ).to(DEVICE)

    print(f"Model parameters: {model.count_parameters():,}")

    opt = AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    for epoch in range(epochs):
        model.train()
        pbar = tqdm(list(batchify(train_data, tok, batch_size)), desc=f"epoch {epoch}")
        for x, y, plan in pbar:
            x, y, plan = x.to(DEVICE), y.to(DEVICE), plan.to(DEVICE)
            action_logits, plan_logits, _ = model(x)
            loss = supervised_task_loss(
                action_logits, y, plan_logits, plan,
                lambda_plan=0.5, plan_length=5, plan_vocab_size=len(PLAN_VOCAB),
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        # Eval
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y, plan in batchify(test_data, tok, 256):
                x, y = x.to(DEVICE), y.to(DEVICE)
                action_logits, _, _ = model(x)
                pred = action_logits.argmax(dim=-1)
                correct += (pred == y).sum().item()
                total += y.numel()
        print(f"Epoch {epoch} test action accuracy = {correct / total:.4f}")

    return model


if __name__ == "__main__":
    train()
