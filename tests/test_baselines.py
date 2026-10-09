import numpy as np
import pytest
from core.counters import Counters
from core.cues import QueryCues
from core.timecue import parse_time_window, DAY
from baselines.brute_force import BruteForce
from baselines.cue_tree import CueTree
from baselines.flat_cue import FlatCueFilter
from tests.test_cuetree import make, fake_ner, build_tree, N, D

def no_ner(texts): return [frozenset() for _ in texts]

def ids(res): return [r.conv_id for r in res]

def test_flat_filter_equals_tree_with_unlimited_beam_for_entity_queries():
    convs = make(); t = build_tree(convs)
    flat = FlatCueFilter(ner=fake_ner); flat.build(convs)
    rng = np.random.default_rng(3)
    for text in ["e3", "e17", "e3 e29", "u42"]:
        q = rng.normal(size=D).astype("float32")
        a = [i for i, _ in t.search(q, 5, 10**6, Counters(), QueryCues(frozenset(text.split())))]
        assert a == ids(flat.retrieve(text, q, 5, query_ts=None)[0])

def test_flat_filter_equals_tree_with_unlimited_beam_for_time_queries():
    convs = make(); t = build_tree(convs)
    flat = FlatCueFilter(ner=no_ner); flat.build(convs)
    ref = 100 * DAY                                                   # 'in February' -> Feb 1970 = leaves 31..58
    w = parse_time_window("what happened in February", ref)
    q = np.random.default_rng(8).normal(size=D).astype("float32")
    a = [i for i, _ in t.search(q, 5, 10**6, Counters(), QueryCues(frozenset(), w, 7.0))]
    r, c = flat.retrieve("what happened in February", q, 5, query_ts=ref)
    assert a == ids(r) and c.cue_checks == N                          # a date scan over every leaf is counted

def test_flat_filter_without_cues_or_with_unknown_entity_is_brute_force():
    convs = make()
    flat = FlatCueFilter(ner=fake_ner); flat.build(convs)
    bf = BruteForce(); bf.build(convs)
    q = np.random.default_rng(1).normal(size=D).astype("float32")
    for text in ["", "never_seen"]:
        r, c = flat.retrieve(text, q, 5, query_ts=None)
        assert ids(r) == ids(bf.retrieve("", q, 5)[0]) and c.embedding_comparisons == N
    assert flat.retrieve("never_seen", q, 5)[1].levels_relaxed == 1

def test_flat_filter_is_cheaper_than_brute_force_on_selective_entity_and_counts_lookups():
    convs = make()
    flat = FlatCueFilter(ner=fake_ner); flat.build(convs)
    r, c = flat.retrieve("u7", convs[7].embedding, 5)
    assert ids(r) == ["c7"] and c.embedding_comparisons == 1 and c.cue_checks == 1 and c.llm_calls == 0

def test_flat_filter_refuses_missing_entities():
    from core.types import Conversation
    with pytest.raises(ValueError):
        FlatCueFilter(ner=fake_ner).build([Conversation("a", "", 0.0, np.ones(4, dtype="float32"))])

def test_faiss_flat_equals_brute_force():
    pytest.importorskip("faiss")
    from baselines.faiss_index import FaissFlat
    convs = make(); bf = BruteForce(); bf.build(convs); ff = FaissFlat(); ff.build(convs)
    rng = np.random.default_rng(6)
    for _ in range(10):
        q = rng.normal(size=D).astype("float32")
        assert ids(bf.retrieve("", q, 5)[0]) == ids(ff.retrieve("", q, 5)[0])
    assert ff.retrieve("", q, 5)[1].embedding_comparisons == N

def test_hnsw_finds_self_queries_counts_cost_and_is_repeatable():
    pytest.importorskip("faiss")
    from baselines.faiss_index import FaissHNSW
    convs = make()
    h1 = FaissHNSW(16, 40, 64); h1.build(convs)
    h2 = FaissHNSW(16, 40, 64); h2.build(convs)
    hits, comps = 0, []
    for i in range(0, N, 10):
        r1, c1 = h1.retrieve("", convs[i].embedding, 5)
        r2, _ = h2.retrieve("", convs[i].embedding, 5)
        assert ids(r1) == ids(r2)
        hits += r1[0].conv_id == f"c{i}"; comps.append(c1.embedding_comparisons)
    assert hits / 30 >= 0.95 and 0 < np.mean(comps) < 1.5 * N and c1.llm_calls == 0


def test_hybrid_uses_filter_when_cues_and_hnsw_when_none():
    pytest.importorskip("faiss")
    from baselines.flat_cue import HybridCueHNSW
    from baselines.faiss_index import FaissHNSW
    convs = make()
    hy = HybridCueHNSW(ner=fake_ner, M=16, ef_construction=40, ef_search=64); hy.build(convs)
    flat = FlatCueFilter(ner=fake_ner); flat.build(convs)
    hn = FaissHNSW(16, 40, 64); hn.build(convs)
    q = np.random.default_rng(5).normal(size=D).astype("float32")
    r, c = hy.retrieve("e3", q, 5)                                    # entity query -> same as the flat filter
    assert ids(r) == ids(flat.retrieve("e3", q, 5)[0]) and c.cue_checks == 1
    r2, c2 = hy.retrieve("", q, 5)                                    # no cues -> same as plain HNSW, counted from FAISS
    assert ids(r2) == ids(hn.retrieve("", q, 5)[0]) and 0 < c2.embedding_comparisons < 2 * N

def test_flat_filter_no_cue_path_does_not_change_results():
    convs = make(); flat = FlatCueFilter(ner=fake_ner); flat.build(convs)
    bf = BruteForce(); bf.build(convs)
    q = np.random.default_rng(11).normal(size=D).astype("float32")
    assert ids(flat.retrieve("", q, 5)[0]) == ids(bf.retrieve("", q, 5)[0])
