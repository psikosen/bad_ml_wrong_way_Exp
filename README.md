# Graph-Agent Brain (GAB)

A research prototype for graph-augmented agent learning with a 0.6M-parameter transformer core.

## Overview

GAB separates cognitive structure (a compact policy/reasoning core) from knowledge acquisition (tools and memory). It implements five progressive experiments:

| Experiment | Description | Signal | Run time |
|---|---|---|---|
| E1 | Naive Bayes action baseline | Supervised | Seconds (CPU) |
| E2 | 0.6M neural planner | Supervised CE | Minutes (CPU/GPU) |
| E3 | Graph memory integration | Contrastive + supervised | Minutes |
| E4 | Execution + feedback loop | REINFORCE RL | Minutes |
| E5 | Teacher distillation + self-learning | KL distillation + RL | Minutes |

## Architecture

- **Core**: 0.6M-parameter TinyTransformer (d=96, 4 heads, 4 layers, ff=192)
- **Memory**: Hybrid property graph + vector index (optional FAISS/Neo4j)
- **Tools**: Search, executor (sandboxed), teacher oracle
- **Actions**: SEARCH, MEMORY_READ, MEMORY_WRITE, EXEC, TEACHER, ANSWER

## Setup

```bash
pip install -r requirements.txt
```

Optional dependencies (FAISS, Neo4j) are not required for baseline experiments.

## Running Experiments

```bash
# E1: NB baseline (seconds)
python runs/exp1_nb.py

# E2: Neural planner (minutes)
python runs/exp2_neural_planner.py

# E3: Graph memory (seconds)
python runs/exp3_graph_memory.py

# E4: RL feedback loop (minutes)
python runs/exp4_exec_feedback.py

# E5: Distillation (minutes)
python runs/exp5_teacher_selflearn.py

# Run test suite
python -m gab.eval.test_suite
```

## Project Structure

```
graph-agent-brain/
  requirements.txt
  gab/
    __init__.py
    tokenization.py        # Hash-based tokenizer
    core_model.py          # 0.6M TinyTransformer
    nb_baseline.py         # Multinomial NB baseline
    memory/
      graph_memory.py      # Hybrid graph + vector memory
      faiss_index.py       # Optional FAISS wrapper
      neo4j_store.py       # Optional Neo4j persistence
    tools/
      toy_search.py        # Simulated search tool
      toy_executor.py      # Safe arithmetic executor
      teacher_oracle.py    # Rule-based teacher for distillation
    training/
      losses.py            # All loss functions (CE, InfoNCE, KL, RL)
      train_e2_supervised.py
      train_e4_rl.py
      distill_e5.py
    eval/
      metrics.py           # Accuracy, F1, Recall@k, MRR, edit distance
      test_suite.py        # Sanity checks for all modules
  runs/
    exp1_nb.py             # E1 runner
    exp2_neural_planner.py # E2 runner
    exp3_graph_memory.py   # E3 runner
    exp4_exec_feedback.py  # E4 runner
    exp5_teacher_selflearn.py # E5 runner
  data/
    synthetic/             # Generated datasets
    traces/                # Teacher traces
```
