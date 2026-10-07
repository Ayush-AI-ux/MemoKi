import json, numpy as np, pytest
from pipeline.loaders import parse_lme_date, load_longmemeval, load_locomo, parse_locomo_date
from pipeline.store import make_splits, select_queries, build_store
from pipeline.embed import embed_conversations

def _fake_lme(tmp_path, n_q=30, pool_n=300):
    sess = {f"s{i}": [{"role": "user", "content": f"hello topic {i}"},
                      {"role": "assistant", "content": f"reply {i}"}] for i in range(pool_n)}
    data = []
    for q in range(n_q):
        ev = [f"s{q}"]
        hay = ev + [f"s{(q*7+j) % pool_n}" for j in range(1, 6)]
        data.append({"question_id": f"q{q}" + ("_abs" if q == 5 else ""), "question_type": "single-session-user",
                     "question": f"what about topic {q}?", "question_date": "2023/06/01 (Thu) 10:00",
                     "answer_session_ids": ev if q != 6 else [],
                     "haystack_session_ids": hay, "haystack_dates": ["2023/05/20 (Sat) 02:21"] * len(hay),
                     "haystack_sessions": [sess[h] for h in hay]})
    p = tmp_path / "lme.json"; p.write_text(json.dumps(data)); return p

def test_dates():
    assert parse_lme_date("2023/05/20 (Sat) 02:21") == 1684549260.0
    assert parse_locomo_date("1:56 pm on 8 May, 2023") > 0

def test_lme_loader_skips_abstention(tmp_path):
    pool, qs = load_longmemeval(_fake_lme(tmp_path))
    ids = {q.id for q in qs}
    assert "q5_abs" not in ids and "q6" not in ids and len(qs) == 28
    assert "user: hello topic 3" in pool["s3"].text

def test_split_deterministic_and_disjoint(tmp_path):
    _, qs = load_longmemeval(_fake_lme(tmp_path))
    a, b = make_splits(qs, 0.3, 0), make_splits(qs, 0.3, 0)
    assert a == b and not set(a["validation"]) & set(a["test"])
    assert len(a["validation"]) + len(a["test"]) == len(qs)

def test_store_deterministic_nested_contains_evidence(tmp_path):
    pool, qs = load_longmemeval(_fake_lme(tmp_path))
    sel = select_queries(qs, [q.id for q in qs], set(pool), 10, 2, seed=0)
    s1 = build_store(pool, sel, 100, seed=11); s2 = build_store(pool, sel, 100, seed=11)
    assert [c.id for c in s1] == [c.id for c in s2] and len(s1) == 100
    ev = {r for q in sel for r in q.relevant_ids}
    assert ev <= {c.id for c in s1}
    big = build_store(pool, sel, 150, seed=11)
    assert {c.id for c in s1} <= {c.id for c in big}      # nested
    assert [q.id for q in sel] == [q.id for q in select_queries(qs, [q.id for q in qs], set(pool), 10, 2, seed=0)]

def test_store_too_big_raises(tmp_path):
    pool, qs = load_longmemeval(_fake_lme(tmp_path))
    sel = select_queries(qs, [q.id for q in qs], set(pool), 5, 2, seed=0)
    with pytest.raises(ValueError):
        build_store(pool, sel, 10_000, seed=0)

def test_embedding_cache(tmp_path):
    pool, _ = load_longmemeval(_fake_lme(tmp_path))
    convs = list(pool.values())[:20]
    calls = []
    def fake(texts):
        calls.append(len(texts)); return np.random.default_rng(len(texts)).normal(size=(len(texts), 8)).astype("float32")
    embed_conversations(convs, fake, tmp_path / "c")
    embed_conversations(convs, fake, tmp_path / "c")
    assert calls == [20]                                   # second run fully cached
    assert all(c.embedding is not None for c in convs)

def test_locomo_loader(tmp_path):
    sample = {"sample_id": "conv-1", "qa": [{"question": "q?", "answer": "a", "evidence": ["D1:3", "D2:1"], "category": 1},
                                             {"question": "none", "answer": "x", "category": 5}],
              "conversation": {"speaker_a": "A", "speaker_b": "B",
                               "session_1_date_time": "1:56 pm on 8 May, 2023", "session_1": [{"speaker": "A", "dia_id": "D1:1", "text": "hi"}],
                               "session_2_date_time": "2:00 pm on 9 May, 2023", "session_2": [{"speaker": "B", "dia_id": "D2:1", "text": "yo"}]}}
    p = tmp_path / "l.json"; p.write_text(json.dumps([sample]))
    convs, qs, _ = load_locomo(p)
    assert len(convs) == 2 and len(qs) == 1 and qs[0].relevant_ids == {"conv-1_s1", "conv-1_s2"}
