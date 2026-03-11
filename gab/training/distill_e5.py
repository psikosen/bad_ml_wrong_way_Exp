"""Distillation + self-learning pipeline for E5.

Teacher traces -> student imitation, then optional RL fine-tuning.
"""

import random

import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm

from gab.tools.teacher_oracle import TeacherOracle

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ACTIONS = ["SEARCH", "MEMORY_READ", "EXEC", "ANSWER"]
A2I = {a: i for i, a in enumerate(ACTIONS)}

PLAN_VOCAB = ["PAD", "PLAN", "SEARCH", "MEMORY_READ", "EXEC", "CHECK", "ANSWER", "END"]
P2I = {t: i for i, t in enumerate(PLAN_VOCAB)}


class Student(nn.Module):
    def __init__(self, vocab: int = 1024, d: int = 64, max_len: int = 12):
        super().__init__()
        self.vocab = vocab
        self.max_len = max_len
        self.emb = nn.Embedding(vocab, d)
        self.fc_action = nn.Linear(d, len(ACTIONS))
        self.fc_plan = nn.Linear(d, len(PLAN_VOCAB))

    def encode(self, text: str) -> torch.Tensor:
        toks = text.lower().split()
        ids = [(hash(t) % (self.vocab - 1)) + 1 for t in toks][: self.max_len]
        ids += [0] * (self.max_len - len(ids))
        return torch.tensor(ids, dtype=torch.long)

    def forward(self, ids: torch.Tensor):
        h = self.emb(ids).mean(dim=0)
        return self.fc_action(h), self.fc_plan(h)


def make_tasks(n: int = 20000, seed: int = 7):
    random.seed(seed)
    topics = [
        "grep flags", "regex groups", "compute 3 + 5",
        "compute 4 * 7", "recall stored note",
    ]
    tasks = []
    for _ in range(n):
        t = random.choice(topics)
        if "compute" in t:
            tasks.append(t)
        else:
            tasks.append(
                random.choice([
                    "look up " + t, "search " + t,
                    "answer " + t, "recall " + t,
                ])
            )
    return tasks


def distill(epochs: int = 2, lr: float = 1e-3, seed: int = 7):
    random.seed(seed)
    torch.manual_seed(seed)

    teacher = TeacherOracle()
    tasks = make_tasks()
    student = Student().to(DEVICE)
    opt = AdamW(student.parameters(), lr=lr)
    ce = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        random.shuffle(tasks)
        pbar = tqdm(tasks[:10000], desc=f"distill epoch {epoch}")
        for task in pbar:
            plan, act = teacher.get_trajectory(task)
            ids = student.encode(task).to(DEVICE)
            logits_a, logits_p = student(ids)
            y_a = torch.tensor(A2I.get(act, A2I["ANSWER"]), dtype=torch.long, device=DEVICE)
            # Next plan token (after PLAN)
            plan_token = plan[1] if len(plan) > 1 else "PAD"
            y_p = torch.tensor(P2I.get(plan_token, 0), dtype=torch.long, device=DEVICE)
            loss = ce(logits_a.unsqueeze(0), y_a.unsqueeze(0)) + 0.3 * ce(
                logits_p.unsqueeze(0), y_p.unsqueeze(0)
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        # Eval
        correct = 0
        eval_tasks = tasks[10000:12000]
        for task in eval_tasks:
            _, act = teacher.get_trajectory(task)
            ids = student.encode(task).to(DEVICE)
            with torch.no_grad():
                logits_a, _ = student(ids)
                pred = ACTIONS[int(logits_a.argmax().item())]
            correct += int(pred == act)
        print(f"Distill epoch {epoch} eval accuracy: {correct / len(eval_tasks):.4f}")

    return student


if __name__ == "__main__":
    distill()
