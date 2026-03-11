#!/usr/bin/env python3
"""E3: Graph memory integration (in-memory + optional FAISS + optional Neo4j).

Demonstrates retrieval from vector store + property graph.
Expected: strong retrieval accuracy; Recall@k >= 0.9 for k=5.

Usage:
    python runs/exp3_graph_memory.py
"""

import random
import sys
import os

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from gab.memory.graph_memory import GraphMemory, MemoryNode, embed_text
from gab.eval.metrics import recall_at_k, mrr


def build_test_memory():
    """Build a small memory store with known facts."""
    mem = GraphMemory(dim=64)
    facts = [
        ("n1", "grep -r searches recursively through directories"),
        ("n2", "awk uses single quotes around the program text"),
        ("n3", "chmod 755 gives rwx for owner and rx for group and others"),
        ("n4", "sed -i edits files in place"),
        ("n5", "find command searches for files in directory hierarchy"),
        ("n6", "tar -xzf extracts gzipped tar archives"),
        ("n7", "git branch creates and lists branches"),
        ("n8", "python decorators wrap functions with additional behavior"),
        ("n9", "json schema validates the structure of json data"),
        ("n10", "regex capture groups use parentheses"),
    ]
    for nid, text in facts:
        mem.add_node(MemoryNode(nid, text, confidence=0.9))

    # Add edges
    mem.add_edge("n1", "n5", "RELATED_TO", 0.7)  # grep and find
    mem.add_edge("n7", "n1", "USED_IN", 0.5)  # git and grep
    return mem, facts


def run_retrieval_eval():
    mem, facts = build_test_memory()

    # Queries with known relevant nodes
    queries = [
        ("how to search recursively with grep", ["n1"]),
        ("awk program syntax", ["n2"]),
        ("file permissions chmod", ["n3"]),
        ("edit files in place", ["n4"]),
        ("find files in directories", ["n5"]),
        ("extract tar archive", ["n6"]),
        ("git branching", ["n7"]),
        ("python function decorators", ["n8"]),
        ("validate json structure", ["n9"]),
        ("regex parentheses groups", ["n10"]),
    ]

    retrieved_ids = []
    relevant_ids = []
    for query, rel in queries:
        results = mem.search(query, k=5)
        ret = [nid for nid, score in results]
        retrieved_ids.append(ret)
        relevant_ids.append(rel)

    r_at_5 = recall_at_k(retrieved_ids, relevant_ids, k=5)
    mrr_val = mrr(retrieved_ids, relevant_ids)

    print(f"Recall@5: {r_at_5:.4f}")
    print(f"MRR:      {mrr_val:.4f}")

    # Show detailed results
    print("\nDetailed retrieval results:")
    for i, (query, rel) in enumerate(queries):
        results = mem.search(query, k=3)
        top = [(nid, f"{score:.3f}") for nid, score in results]
        hit = "HIT" if rel[0] in [r[0] for r in results] else "MISS"
        print(f"  [{hit}] \"{query}\" -> {top}")

    # Test graph edges
    print("\nGraph neighbors of n1 (grep):")
    for neighbor_id, edge in mem.get_neighbors("n1"):
        node = mem.get_node(neighbor_id)
        print(f"  -> {neighbor_id} ({edge.rel_type}, w={edge.weight}): {node.text[:50]}")

    return r_at_5, mrr_val


if __name__ == "__main__":
    try:
        import faiss
        print("FAISS: available")
    except ImportError:
        print("FAISS: not available (using brute-force fallback)")

    print("=" * 50)
    print("E3: Graph Memory Integration")
    print("=" * 50)

    r_at_5, mrr_val = run_retrieval_eval()

    print(f"\nSuccess criteria: Recall@5 >= 0.9: {'PASS' if r_at_5 >= 0.9 else 'FAIL'}")
    print("E3 complete.")
