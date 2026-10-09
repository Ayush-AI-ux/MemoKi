import time
import numpy as np
import pytest
from core.counters import Counters
from core.cues import QueryCues
from core.timecue import TimeWindow, DAY
from core.tree import MemoryTree
from core.types import Conversation
from baselines.brute_force import BruteForce
from baselines.centroid_tree import CentroidTree
from baselines.cue_tree import CueTree

N, D = 300, 16

def make(seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(N, D)).astype("float32")
    convs = []
    for i in range(N):
        ents = frozenset({f"e{j}" for j in rng.integers(0, 40, 3)} | {f"u{i}"})   # 3 shared + 1 unique entity
        c = Conversation(f"c{i}", "", float(i) * DAY, X[i])
        c.cues["entities"] = ents
        convs.append(c)
    return convs

def fake_ner(texts):                       # query text IS the entity list, e.g. "e5 e7"
    return [frozenset(t.split()) for t in texts]

def build_tree(convs, B=6, mf=0.35):
    t = MemoryTree(B, mf)
    for c in convs: t.insert(c.id, c.embedding, c.cues["entities"], c.timestamp)
    return t

def unit(M): return M / np.linalg.norm(M, axis=-1, keepdims=True)

def test_summaries_are_unions_of_children():
    t = build_tree(make())
    stack = [t.root]
    while stack:
        n = stack.pop()
        if n.is_leaf: continue
        assert n.ents == set().union(*(c.ents for c in n.children))
        assert n.tmin == min(c.tmin for c in n.children) and n.tmax == max(c.tmax for c in n.children)
        stack.extend(n.children)

def test_unlimited_beam_entity_gate_equals_brute_force_on_matching_leaves():
    convs = make(); t = build_tree(convs)
    rng = np.random.default_rng(5)
    for ent in ["e3", "e17", "e29"]:
        q = rng.normal(size=D).astype("float32")
        match = [c for c in convs if ent in c.cues["entities"]]
        assert len(match) >= 3
        sims = unit(np.stack([c.embedding for c in match])) @ (q / np.linalg.norm(q))
        expect = [match[i].id for i in np.argsort(-sims, kind="stable")[:5]]
        got = [i for i, _ in t.search(q, 5, 10**6, Counters(), QueryCues(frozenset({ent})))]
        assert got == expect

def test_selective_entity_cuts_embedding_comparisons_and_counts_cue_checks():
    convs = make(); t = build_tree(convs)
    q = convs[7].embedding
    plain, gated = Counters(), Counters()
    t.search(q, 5, 3, plain)
    hits = t.search(q, 5, 3, gated, QueryCues(frozenset({"u7"})))      # unique entity: one matching leaf
    assert hits[0][0] == "c7"
    assert gated.cue_checks > 0 and gated.embedding_comparisons < plain.embedding_comparisons
    assert gated.levels_relaxed == 0

def test_unknown_entity_relaxes_and_equals_ungated_search():
    convs = make(); t = build_tree(convs)
    q = np.random.default_rng(2).normal(size=D).astype("float32")
    c1, c2 = Counters(), Counters()
    a = t.search(q, 5, 4, c1)
    b = t.search(q, 5, 4, c2, QueryCues(frozenset({"never_seen"})))
    assert a == b and c2.levels_relaxed >= 1

def test_time_gate_unlimited_beam_returns_only_leaves_in_window():
    convs = make(); t = build_tree(convs)
    w = TimeWindow(50 * DAY, 80 * DAY, "x")
    q = np.random.default_rng(4).normal(size=D).astype("float32")
    got = [i for i, _ in t.search(q, 8, 10**6, Counters(), QueryCues(frozenset(), w, 0.0))]
    assert got and all(50 <= int(i[1:]) <= 80 for i in got)
    slack = [i for i, _ in t.search(q, 40, 10**6, Counters(), QueryCues(frozenset(), w, 5.0))]
    assert all(45 <= int(i[1:]) <= 85 for i in slack) and any(int(i[1:]) < 50 or int(i[1:]) > 80 for i in slack)

def test_queries_without_cues_behave_exactly_like_the_no_cue_tree():
    convs = make()
    cue = CueTree(6, 3, 0.35, use_fallback=False, ner=fake_ner); cue.build(convs)
    base = CentroidTree(6, 3, 0.35); base.build(convs)
    rng = np.random.default_rng(9)
    for _ in range(10):
        q = rng.normal(size=D).astype("float32")
        r1, c1 = cue.retrieve("", q, 5, query_ts=0.0)
        r2, c2 = base.retrieve("", q, 5)
        assert [r.conv_id for r in r1] == [r.conv_id for r in r2]
        assert c1.embedding_comparisons == c2.embedding_comparisons and c1.cue_checks == 0 and c1.llm_calls == 0

def test_fallback_triggers_on_weak_match_and_is_fully_counted():
    convs = make()
    cue = CueTree(6, 3, 0.35, use_fallback=True, tau=2.0, ner=fake_ner); cue.build(convs)   # cosine can never reach 2
    bf = BruteForce(); bf.build(convs)
    q = np.random.default_rng(1).normal(size=D).astype("float32")
    r, c = cue.retrieve("", q, 5, query_ts=0.0)
    assert c.fallback_triggered and c.embedding_comparisons >= N
    assert [x.conv_id for x in r] == [x.conv_id for x in bf.retrieve("", q, 5)[0]]
    cue.tau = -1.0                                                   # never weak -> no fallback
    assert not cue.retrieve("", q, 5, query_ts=0.0)[1].fallback_triggered

def test_build_refuses_missing_entity_cues():
    convs = [Conversation("a", "", 0.0, np.ones(4, dtype="float32"))]
    with pytest.raises(ValueError):
        CueTree(ner=fake_ner).build(convs)

def test_prep_time_reported_separately_from_search_time():
    def slow_ner(texts):
        time.sleep(0.03); return [frozenset() for _ in texts]
    convs = make()
    cue = CueTree(6, 3, 0.35, ner=slow_ner); cue.build(convs)
    _, c = cue.retrieve("x", convs[0].embedding, 5, query_ts=0.0)
    assert c.prep_ms >= 25 and c.latency_ms < c.prep_ms
    _, c2 = cue.retrieve("x", convs[0].embedding, 5, query_ts=0.0)   # cached prep: same reported cost
    assert c2.prep_ms == c.prep_ms                                    # reported cost is remembered ...
    assert c2.prep_spent_ms < 5 and c2.latency_ms >= 0                # ... but a cache hit is not subtracted from search time

def test_ablation_switches_change_the_cues_used():
    convs = make()
    cue = CueTree(6, 3, 0.35, use_names=False, use_time=False, ner=fake_ner); cue.build(convs)
    q, _ = cue.prepare("e3", 100 * DAY)
    assert not q.active
    cue.use_names = True
    assert cue.prepare("e3", 100 * DAY)[0].entities == frozenset({"e3"})
