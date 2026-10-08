import numpy as np
from core.tree import MemoryTree
from core.counters import Counters
from core.types import Conversation
from baselines.centroid_tree import CentroidTree
from baselines.brute_force import BruteForce

def clustered(n=400, d=32, k=20, noise=0.1, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(k, d))
    X = centers[rng.integers(0, k, n)] + noise * rng.normal(size=(n, d))
    return X.astype("float32")

def build(X, B=8):
    t = MemoryTree(B)
    for i, v in enumerate(X):
        t.insert(f"c{i}", v)
    return t

def test_every_leaf_reachable_and_counts_consistent():
    X = clustered(); t = build(X)
    assert {l.conv_id for l in t.iter_leaves()} == {f"c{i}" for i in range(len(X))}
    assert t.root.count == len(X)
    unit = X / np.linalg.norm(X, axis=1, keepdims=True)
    assert np.allclose(t.root.sum, unit.sum(0), atol=1e-3)           # root describes everything beneath it

def test_structure_is_balanced_and_bounded():
    t = build(clustered(1000), B=8)
    s = t.stats()
    assert s["uniform_depth"] and s["max_children"] <= 8 and s["leaves"] == 1000
    assert s["height"] <= 6

def test_parent_pointers_and_sums():
    t = build(clustered(300), B=5)
    stack = [t.root]
    while stack:
        n = stack.pop()
        if n.is_leaf: continue
        assert n.count == sum(c.count for c in n.children)
        assert np.allclose(n.sum, np.sum([c.sum for c in n.children], axis=0), atol=1e-3)
        for c in n.children:
            assert c.parent is n
            stack.append(c)

def test_deterministic_structure():
    X = clustered(300)
    assert build(X).to_dict() == build(X).to_dict()

def test_unlimited_beam_equals_brute_force():
    X = clustered(250, noise=0.3)
    convs = [Conversation(f"c{i}", "", 0.0, v) for i, v in enumerate(X)]
    bf = BruteForce(); bf.build(convs)
    tr = CentroidTree(max_children=6, beam=10**6); tr.build(convs)
    rng = np.random.default_rng(1)
    for _ in range(15):
        q = rng.normal(size=X.shape[1]).astype("float32")
        a = [r.conv_id for r in bf.retrieve("q", q, 5)[0]]
        b = [r.conv_id for r in tr.retrieve("q", q, 5)[0]]
        assert a == b

def test_self_queries_found_with_small_beam_and_far_fewer_comparisons():
    X = clustered(1000)
    convs = [Conversation(f"c{i}", "", 0.0, v) for i, v in enumerate(X)]
    tr = CentroidTree(max_children=8, beam=3); tr.build(convs)
    hits, comps = 0, []
    for i in range(0, 1000, 25):
        res, c = tr.retrieve("q", X[i], 5)
        hits += any(r.conv_id == f"c{i}" for r in res); comps.append(c.embedding_comparisons)
        assert c.llm_calls == 0 and c.nodes_visited == c.embedding_comparisons
    assert hits / 40 >= 0.95
    assert np.mean(comps) < 0.25 * 1000

def test_degenerate_identical_vectors_do_not_crash():
    t = MemoryTree(4)
    for i in range(50):
        t.insert(f"c{i}", np.ones(8, dtype="float32"))
    s = t.stats()
    assert s["leaves"] == 50 and s["max_children"] <= 4 and s["uniform_depth"]

def test_empty_tree_and_duplicates():
    t = MemoryTree(4)
    assert t.search(np.ones(4), 5, 3, Counters()) == []
    t.insert("a", np.ones(4))
    try:
        t.insert("a", np.ones(4)); assert False
    except ValueError:
        pass

def test_export_dict_shape():
    d = build(clustered(60), B=4).to_dict(max_depth=2)
    assert "count" in d and all("collapsed" in c or "conv_id" in c for c in d["children"])


def test_min_fill_keeps_nodes_from_being_tiny_and_stays_exact():
    X = clustered(600)
    t = MemoryTree(8, min_fill=0.35)
    for i, v in enumerate(X): t.insert(f"c{i}", v)
    stack, counts = [t.root], []
    while stack:
        n = stack.pop()
        if n.is_leaf: continue
        if n is not t.root: counts.append(len(n.children))
        stack.extend(n.children)
    assert min(counts) >= 4                                   # ceil(0.35 * 9)
    s = t.stats()
    assert s["uniform_depth"] and s["max_children"] <= 8 and s["leaves"] == 600
    convs = [Conversation(f"c{i}", "", 0.0, v) for i, v in enumerate(X)]
    bf = BruteForce(); bf.build(convs)
    tr = CentroidTree(8, 10**6, min_fill=0.35); tr.build(convs)
    q = np.random.default_rng(3).normal(size=X.shape[1]).astype("float32")
    assert [r.conv_id for r in bf.retrieve("q", q, 5)[0]] == [r.conv_id for r in tr.retrieve("q", q, 5)[0]]

def test_min_fill_validation_and_degenerate_input():
    for bad in (-0.1, 0.6):
        try:
            MemoryTree(8, min_fill=bad); assert False
        except ValueError:
            pass
    t = MemoryTree(4, min_fill=0.4)
    for i in range(40): t.insert(f"c{i}", np.ones(8, dtype="float32"))
    assert t.stats()["leaves"] == 40 and t.stats()["uniform_depth"]
