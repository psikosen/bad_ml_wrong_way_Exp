#!/usr/bin/env python3
"""E1: Tiny NB baseline - Predict next action from task text.

Runs in seconds on CPU. Expected: 70-90% action accuracy on synthetic splits.

Usage:
    python runs/exp1_nb.py
"""

import math
import random
from collections import Counter, defaultdict

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from gab.nb_baseline import MultinomialNB, ACTIONS, tokenize


def gen_e1_dataset(n=20000, seed=7):
    random.seed(seed)
    templates = {
        "SEARCH": [
            "look up {topic}",
            "search for {topic}",
            "find docs on {topic}",
            "what is the latest {topic}",
        ],
        "MEMORY_READ": [
            "recall {topic} from memory",
            "what did we store about {topic}",
            "retrieve saved note on {topic}",
        ],
        "EXEC": [
            "run and test {topic}",
            "execute a command for {topic}",
            "compute {topic} and verify",
        ],
        "TEACHER": [
            "ask teacher about {topic}",
            "consult teacher for {topic}",
            "request guidance on {topic}",
        ],
        "ANSWER": [
            "answer directly: {topic}",
            "respond now about {topic}",
            "give a short answer for {topic}",
        ],
    }
    topics = [
        "grep flags", "awk syntax", "regex groups",
        "file permissions", "loops", "json schema",
    ]
    data = []
    for _ in range(n):
        y = random.choice(ACTIONS)
        tpl = random.choice(templates[y])
        x = tpl.format(topic=random.choice(topics))
        # paraphrase noise
        if random.random() < 0.25:
            x = x.replace("look up", "find").replace("answer", "respond")
        data.append((x, y))
    random.shuffle(data)
    return data


def accuracy(model, data):
    correct = 0
    for x, y in data:
        correct += int(model.predict(x) == y)
    return correct / max(1, len(data))


if __name__ == "__main__":
    data = gen_e1_dataset()
    split = int(0.8 * len(data))
    train, test = data[:split], data[split:]

    nb = MultinomialNB(alpha=1.0)
    nb.fit(train)

    print("=" * 50)
    print("E1: Tiny NB Baseline")
    print("=" * 50)
    print(f"Train size: {len(train)}, Test size: {len(test)}")
    print(f"Train accuracy: {accuracy(nb, train):.4f}")
    print(f"Test accuracy:  {accuracy(nb, test):.4f}")

    # Per-action breakdown
    from collections import defaultdict
    per_action = defaultdict(lambda: [0, 0])
    for x, y in test:
        pred = nb.predict(x)
        per_action[y][1] += 1
        if pred == y:
            per_action[y][0] += 1
    print("\nPer-action accuracy:")
    for action in ACTIONS:
        c, t = per_action[action]
        print(f"  {action:15s}: {c}/{t} = {c/max(1,t):.4f}")

    # Sample predictions
    print("\nSample predictions:")
    samples = [
        "please find documentation on regex groups",
        "what did we save about file permissions",
        "run and test loops",
        "consult teacher for awk syntax",
        "respond now about json schema",
    ]
    for s in samples:
        print(f"  > {s} => {nb.predict(s)}")
