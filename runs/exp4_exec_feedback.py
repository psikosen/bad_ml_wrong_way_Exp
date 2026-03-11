#!/usr/bin/env python3
"""E4: Execution + feedback loop (toy executor + REINFORCE).

Trains a policy to learn when to use EXEC vs other actions.
Expected: rising task success rate; improved tool timing.

Usage:
    python runs/exp4_exec_feedback.py
"""

import random
import sys
import os

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from gab.training.train_e4_rl import train_reinforce


if __name__ == "__main__":
    print("=" * 50)
    print("E4: Execution + Feedback Loop (REINFORCE)")
    print("=" * 50)

    policy = train_reinforce(steps=2000, seed=7)
    print("\nE4 complete.")
