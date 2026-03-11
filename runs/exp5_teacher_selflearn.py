#!/usr/bin/env python3
"""E5: Teacher distillation + self-learning.

Distills trajectories from teacher oracle; student imitates action sequences.
Expected: >= 90% imitation accuracy on held-out traces.

Usage:
    python runs/exp5_teacher_selflearn.py
"""

import random
import sys
import os

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from gab.training.distill_e5 import distill


if __name__ == "__main__":
    print("=" * 50)
    print("E5: Teacher Distillation + Self-Learning")
    print("=" * 50)

    student = distill(epochs=2, lr=1e-3, seed=7)
    print("\nE5 complete.")
