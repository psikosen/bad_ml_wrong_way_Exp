#!/usr/bin/env python3
"""E2: Tiny neural planner with the 0.6M core.

Short supervised training round on synthetic data.
Expected: higher generalization than NB; calibrated uncertainty.

Usage:
    python runs/exp2_neural_planner.py
"""

import random
import sys
import os

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from gab.training.train_e2_supervised import train


if __name__ == "__main__":
    print("=" * 50)
    print("E2: Tiny Neural Planner (0.6M core)")
    print("=" * 50)

    torch.manual_seed(7)
    np.random.seed(7)
    random.seed(7)

    model = train(epochs=2, lr=3e-4, batch_size=128, seed=7)
    print(f"\nFinal model parameters: {model.count_parameters():,}")
    print("E2 complete.")
