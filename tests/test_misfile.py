import numpy as np
from core.counters import Counters
from core.cues import QueryCues
from core.misfile import pick_misfiled, plan_moves
from core.tree import MemoryTree
from tests.test_cuetree import make, build_tree, N, D

def check_invariants(t, n_leaves):
    seen, stack, depths = set(), [(t.root, 1)], set()
    while stack:
        n, d = stack.pop()
        if n.is_leaf:
            seen.add(n.conv_id); depths.add(d); continue
        assert n.children, "empty internal node left behind"
        assert n.count == sum(c.count for c in n.children)
        assert np.allclose(n.sum, np.sum([c.sum for c in n.children], axis=0), atol=1e-3)
        assert n.ents == set().union(*(c.ents for c in n.children))
        assert n.tmin == min(c.tmin for c in n.children) and n.tmax == max(c.tmax for c in n.children)
        for c in n.children:
            assert c.parent is n
            stack.append((c, d + 1))
    assert len(seen) == n_leaves and len(depths) == 1               # nothing lost, depth still uniform

def moved_tree(frac=0.2, seed=3, B=6):
    convs = make(); t = build_tree(convs, B=B)
    ids = [c.id for c in convs]
    chosen = pick_misfiled(ids[:50], ids[50:], frac, seed)
    moves = plan_moves(t, chosen, seed)
    t.move_leaves(moves)
    return convs, t, chosen, moves

def test_move_keeps_all_invariants():
    convs, t, chosen, moves = moved_tree()
    check_invariants(t, N)
    assert t.root.count == N

def test_moves_really_cross_top_level_branches():
    convs = make(); t = build_tree(convs, B=6)
    before = {c.id: t.top_branch(t.leaves[c.id].parent).uid for c in convs}
    chosen = pick_misfiled([c.id for c in convs[:50]], [c.id for c in convs[50:]], 0.2, 3)
    t.move_leaves(plan_moves(t, chosen, 3))
    assert chosen and all(t.top_branch(t.leaves[i].parent).uid != before[i] for i in chosen if t.leaves[i].parent is not None)

def test_stratified_sampling_hits_the_evidence_group():
    ev, oth = [f"e{i}" for i in range(40)], [f"o{i}" for i in range(400)]
    chosen = pick_misfiled(ev, oth, 0.10, 1)
    assert sum(c.startswith("e") for c in chosen) == 4 and sum(c.startswith("o") for c in chosen) == 40
    assert chosen == pick_misfiled(ev, oth, 0.10, 1) and chosen != pick_misfiled(ev, oth, 0.10, 2)

def test_move_into_own_parent_is_a_noop_and_zero_fraction_changes_nothing():
    convs = make(); t = build_tree(convs, B=6)
    a = t.to_dict()
    leaf = t.leaves["c5"]; t.move_leaves([("c5", leaf.parent)])
    assert t.to_dict() == a
    assert pick_misfiled(["a"], ["b"], 0.0, 1) == []

def test_cue_gate_finds_a_misfiled_leaf_because_summaries_were_recomputed():
    convs, t, chosen, moves = moved_tree(frac=0.3)
    for cid in chosen[:15]:
        i = int(cid[1:])
        hits = t.search(convs[i].embedding, 5, 3, Counters(), QueryCues(frozenset({f"u{i}"})))   # unique entity
        assert hits and hits[0][0] == cid

def test_entity_flat_search_matches_the_flat_filter_and_handles_no_cue():
    from baselines.flat_cue import FlatCueFilter
    from tests.test_cuetree import fake_ner
    convs = make(); t = build_tree(convs, B=6)
    flat = FlatCueFilter(ner=fake_ner); flat.build(convs)
    q = np.random.default_rng(2).normal(size=D).astype("float32")
    got = [i for i, _ in t.entity_flat_search(q, 5, Counters(), QueryCues(frozenset({"e3"})))]
    assert got == [r.conv_id for r in flat.retrieve("e3", q, 5)[0]]
    assert t.entity_flat_search(q, 5, Counters(), QueryCues()) == []
    assert t.entity_flat_search(q, 5, Counters(), QueryCues(frozenset({"never_seen"}))) == []

def test_nodes_can_overfill_after_misfiling_and_search_still_works():
    convs, t, chosen, moves = moved_tree(frac=0.5, B=4)
    assert t.stats()["max_children"] >= 4
    assert t.search(convs[0].embedding, 5, 3, Counters())


def clustered_tree(n=600, k=30, noise=0.08, B=6):
    from tests.test_tree import clustered, build
    X = clustered(n, d=32, k=k, noise=noise, seed=4)
    return X, build(X, B=B)

def test_audit_repairs_deliberate_misfiling_and_keeps_invariants():
    X, t = clustered_tree()
    ids = [f"c{i}" for i in range(len(X))]
    chosen = pick_misfiled(ids[:60], ids[60:], 0.2, 5)
    t.move_leaves(plan_moves(t, chosen, 5))
    def self_hits(tree):
        return np.mean([any(h == f"c{i}" for h, _ in tree.search(X[i], 5, 1, Counters())) for i in range(len(X))])
    before = self_hits(t)
    stats = t.audit_refile(margin=0.05, rounds=2)
    check_invariants(t, len(X))
    assert self_hits(t) > before + 0.05                               # beam 1: far more leaves reachable after repair
    caught = len(set(stats["moved"]) & set(chosen)) / len(chosen)
    assert caught > 0.5 and stats["comparisons"] > 0

def test_audit_barely_touches_a_clean_clustered_tree():
    X, t = clustered_tree()
    stats = t.audit_refile(margin=0.10)
    assert len(stats["moved"]) < 0.10 * len(X)
    check_invariants(t, len(X))

def test_audit_with_huge_margin_changes_nothing():
    X, t = clustered_tree()
    a = t.to_dict()
    assert t.audit_refile(margin=5.0)["moved"] == [] and t.to_dict() == a
