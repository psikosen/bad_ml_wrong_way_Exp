"""Test suite for GAB experiments.

Runs basic sanity checks on all modules.
"""

import sys


def test_tokenizer():
    from gab.tokenization import HashTokenizer
    tok = HashTokenizer(vocab_size=2960, max_len=128)
    ids = tok.encode("hello world test")
    assert len(ids) == 128
    assert ids[0] != 0  # non-pad for first token
    assert ids[-1] == 0  # padding at end
    print("  [PASS] tokenizer")


def test_core_model():
    import torch
    from gab.core_model import TinyTransformerCore
    model = TinyTransformerCore(
        vocab_size=2960, max_len=128, d_model=96, nhead=4,
        num_layers=4, d_ff=192, num_actions=6, plan_vocab_size=10,
    )
    params = model.count_parameters()
    assert 590000 < params < 610000, f"Expected ~599K params, got {params}"
    x = torch.randint(0, 2960, (2, 128))
    action_logits, plan_logits, value = model(x)
    assert action_logits.shape == (2, 6)
    assert plan_logits.shape == (2, 128, 10)
    assert value.shape == (2,)
    print(f"  [PASS] core_model ({params:,} params)")


def test_nb_baseline():
    from gab.nb_baseline import MultinomialNB
    nb = MultinomialNB()
    nb.fit([("search for grep", "SEARCH"), ("recall memory", "MEMORY_READ")])
    pred = nb.predict("search for awk")
    assert pred == "SEARCH"
    print("  [PASS] nb_baseline")


def test_graph_memory():
    from gab.memory.graph_memory import GraphMemory, MemoryNode
    mem = GraphMemory(dim=64)
    mem.add_node(MemoryNode("n1", "grep -r searches recursively"))
    mem.add_node(MemoryNode("n2", "awk uses single quotes"))
    results = mem.search("grep recursive", k=2)
    assert len(results) == 2
    assert results[0][0] == "n1"  # grep node should rank first
    print("  [PASS] graph_memory")


def test_tools():
    from gab.tools.toy_search import ToySearch
    from gab.tools.toy_executor import ToyExecutor
    from gab.tools.teacher_oracle import TeacherOracle

    search = ToySearch()
    result = search.search("grep flags")
    assert "grep" in result.lower()

    executor = ToyExecutor()
    output, code, success = executor.execute("compute 3 + 5")
    assert success and "8" in output

    teacher = TeacherOracle()
    plan, action = teacher.get_trajectory("compute 2 + 3")
    assert action == "EXEC"
    print("  [PASS] tools")


def test_losses():
    import torch
    from gab.training.losses import (
        action_cross_entropy, entropy_regularization,
        memory_contrastive_loss, distillation_loss,
    )

    logits = torch.randn(4, 6)
    labels = torch.randint(0, 6, (4,))
    loss = action_cross_entropy(logits, labels)
    assert loss.item() > 0

    ent = entropy_regularization(logits)
    assert ent.item() < 0  # negative entropy

    q = torch.randn(4, 64)
    pos = torch.randn(4, 64)
    neg = torch.randn(4, 3, 64)
    cl = memory_contrastive_loss(q, pos, neg)
    assert cl.item() > 0

    student = torch.randn(4, 6)
    teacher = torch.randn(4, 6)
    dl = distillation_loss(student, teacher)
    assert dl.item() >= 0
    print("  [PASS] losses")


def test_metrics():
    from gab.eval.metrics import accuracy, macro_f1, recall_at_k, mrr, edit_distance

    assert accuracy([1, 2, 3], [1, 2, 3]) == 1.0
    assert accuracy([1, 2, 3], [3, 2, 1]) < 1.0

    f1 = macro_f1(["A", "B", "A"], ["A", "B", "B"], ["A", "B"])
    assert 0 < f1 < 1

    assert recall_at_k([["a", "b", "c"]], [["b"]], k=3) == 1.0
    assert mrr([["a", "b"]], [["b"]]) == 0.5

    assert edit_distance(["A", "B", "C"], ["A", "B", "C"]) == 0
    assert edit_distance(["A", "B"], ["A", "C"]) == 1
    print("  [PASS] metrics")


def run_all():
    print("Running GAB test suite...")
    tests = [
        test_tokenizer,
        test_core_model,
        test_nb_baseline,
        test_graph_memory,
        test_tools,
        test_losses,
        test_metrics,
    ]
    failures = 0
    for test_fn in tests:
        try:
            test_fn()
        except Exception as e:
            print(f"  [FAIL] {test_fn.__name__}: {e}")
            failures += 1

    print(f"\n{len(tests) - failures}/{len(tests)} tests passed.")
    return failures == 0


if __name__ == "__main__":
    success = run_all()
    sys.exit(0 if success else 1)
