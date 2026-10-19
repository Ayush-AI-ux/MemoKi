from core.recency import apply_recency, pair_outcome
from core.types import Result, Conversation, Query
from pipeline.update_pairs import build_update_pairs

TS = {"old": 100.0, "mid": 200.0, "new": 300.0, "far": 50.0}

def R(*pairs): return [Result(i, s) for i, s in pairs]

def test_newer_overtakes_older_when_scores_are_close():
    res = R(("old", 0.60), ("new", 0.59), ("far", 0.30))
    out = apply_recency(res, TS, 3, delta=0.02, lam=0.02)
    assert [r.conv_id for r in out][:2] == ["new", "old"]

def test_candidates_outside_delta_are_untouched():
    res = R(("old", 0.60), ("new", 0.50))                                  # gap 0.10 > delta
    assert [r.conv_id for r in apply_recency(res, TS, 2, 0.02, 0.05)] == ["old", "new"]

def test_zero_lambda_or_delta_is_a_noop_and_k_is_respected():
    res = R(("old", 0.6), ("new", 0.59), ("mid", 0.58), ("far", 0.1))
    assert apply_recency(res, TS, 2, 0.05, 0.0) == res[:2] and apply_recency(res, TS, 2, 0.0, 0.05) == res[:2]
    assert len(apply_recency(res, TS, 3, 0.05, 0.02)) == 3

def test_bonus_is_bounded_by_lambda():
    res = R(("old", 0.60), ("new", 0.55))
    out = {r.conv_id: r.score for r in apply_recency(res, TS, 2, 0.10, 0.01)}
    assert abs(out["new"] - 0.56) < 1e-9 and abs(out["old"] - 0.60) < 1e-9   # newest gets +lambda, oldest +0
    assert [r.conv_id for r in apply_recency(res, TS, 2, 0.10, 0.01)] == ["old", "new"]   # lambda too small to flip a 0.05 gap

def test_deterministic_ties():
    res = R(("b", 0.5), ("a", 0.5))
    ts = {"a": 1.0, "b": 1.0}
    assert [r.conv_id for r in apply_recency(res, ts, 2, 0.1, 0.1)] == ["a", "b"]

def test_pair_outcome_cases():
    assert pair_outcome(["n", "o"], "o", "n") == {"fresh_first": True, "stale": False, "newer_found": True, "neither": False}
    assert pair_outcome(["o", "n"], "o", "n")["stale"] and not pair_outcome(["o", "n"], "o", "n")["fresh_first"]
    assert pair_outcome(["o", "x"], "o", "n")["stale"] and not pair_outcome(["o", "x"], "o", "n")["newer_found"]
    assert not pair_outcome(["n", "x"], "o", "n")["stale"]
    assert pair_outcome(["x", "y"], "o", "n")["neither"] and not pair_outcome(["x", "y"], "o", "n")["stale"]

def test_update_pairs_pick_earliest_and_latest_and_skip_bad_cases():
    pool = {i: Conversation(i, "", t) for i, t in {"a": 1.0, "b": 2.0, "c": 3.0, "d": 5.0, "e": 5.0, "f": 9.0}.items()}
    qs = [Query("q1", "t", {"a", "c"}, "knowledge-update"), Query("q2", "t", {"c"}, "knowledge-update"),
          Query("q3", "t", {"d", "e"}, "knowledge-update"), Query("q4", "t", {"a", "b", "f"}, "knowledge-update"),
          Query("q5", "t", {"a", "b"}, "multi-session")]
    pairs, st = build_update_pairs(qs, pool)
    assert [(p["qid"], p["older"], p["newer"]) for p in pairs] == [("q1", "a", "c"), ("q4", "a", "f")]
    assert st["knowledge_update_queries"] == 4 and st["dropped_same_timestamp"] == 1 and st["pairs_with_more_than_two_evidence"] == 1
    assert build_update_pairs(qs, pool, allowed_ids={"q1"})[0][0]["qid"] == "q1"


def test_wrapper_adds_no_comparisons_and_zero_lambda_equals_inner():
    import numpy as np
    from baselines.brute_force import BruteForce
    from baselines.recency_wrapper import RecencyWrapper
    rng = np.random.default_rng(0)
    convs = [Conversation(f"c{i}", "", float(i), rng.normal(size=8).astype("float32")) for i in range(60)]
    bf = BruteForce(); bf.build(convs)
    w = RecencyWrapper(BruteForce(), 0.05, 0.0, pool=10); w.build(convs)
    q = rng.normal(size=8).astype("float32")
    r0, c0 = bf.retrieve("", q, 5); r1, c1 = w.retrieve("", q, 5)
    assert [r.conv_id for r in r0] == [r.conv_id for r in r1]
    assert c1.embedding_comparisons == c0.embedding_comparisons == 60 and c1.llm_calls == 0
    w.delta, w.lam = 0.5, 0.5                                          # a strong rule still returns exactly k
    assert len(w.retrieve("", q, 5)[0]) == 5


def test_pairwise_outcome():
    from core.recency import pairwise_outcome
    assert pairwise_outcome(["x", "o", "y", "n"], "o", "n") == {"both": True, "stale_pair": True, "fresh_first": False}
    assert pairwise_outcome(["n", "o"], "o", "n") == {"both": True, "stale_pair": False, "fresh_first": True}
    assert pairwise_outcome(["o", "x"], "o", "n") == {"both": False, "stale_pair": False, "fresh_first": False}


def test_local_mode_reorders_near_ties_far_below_the_best_result():
    ts = {"best": 400.0, "old": 100.0, "new": 300.0}
    res = R(("best", 0.90), ("old", 0.50), ("new", 0.49))          # the pair sits 0.4 below the top result
    top = apply_recency(res, ts, 3, delta=0.05, lam=0.05, mode="top")
    loc = apply_recency(res, ts, 3, delta=0.05, lam=0.05, mode="local")
    assert [r.conv_id for r in top] == ["best", "old", "new"]      # 'top' mode never sees the pair as close
    assert [r.conv_id for r in loc] == ["best", "new", "old"]      # 'local' mode swaps the near-tie
    assert apply_recency(res, ts, 3, 0.05, 0.0, "local") == res

def test_local_mode_ignores_pairs_that_are_not_close_and_is_bounded():
    ts = {"old": 100.0, "new": 300.0}
    res = R(("old", 0.60), ("new", 0.40))
    assert [r.conv_id for r in apply_recency(res, ts, 2, 0.05, 0.05, "local")] == ["old", "new"]
    out = {r.conv_id: r.score for r in apply_recency(R(("old", 0.50), ("new", 0.49)), ts, 2, 0.05, 0.01, "local")}
    assert abs(out["new"] - 0.50) < 1e-9 and abs(out["old"] - 0.50) < 1e-9 or out["new"] <= 0.49 + 0.01 + 1e-9
