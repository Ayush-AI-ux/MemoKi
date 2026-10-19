import json, sys, types
from spike.common import load_slice, to_messages, extrapolate
from spike.meter import Meter

def fake_locomo(tmp_path):
    def turn(sp, i): return {"speaker": sp, "dia_id": f"D1:{i}", "text": f"hello {i}"}
    conv = {"speaker_a": "Ann", "speaker_b": "Bob",
            "session_1_date_time": "1:00 pm on 1 May, 2023", "session_1": [turn("Ann", 1), turn("Bob", 2), turn("Ann", 3)],
            "session_2_date_time": "1:00 pm on 2 May, 2023", "session_2": [turn("Bob", 1), turn("Ann", 2)],
            "session_3_date_time": "1:00 pm on 3 May, 2023", "session_3": [turn("Ann", 1)]}
    qa = [{"question": "q1", "answer": "a", "evidence": ["D1:1"], "category": 1},
          {"question": "q2", "answer": "a", "evidence": ["D2:1", "D3:1"], "category": 2},
          {"question": "q3", "answer": "a", "evidence": ["D2:2"], "category": 1},
          {"question": "q4", "answer": "a", "category": 5}]
    p = tmp_path / "l.json"; p.write_text(json.dumps([{"sample_id": "conv-x", "conversation": conv, "qa": qa}])); return p

def test_load_slice_keeps_only_questions_fully_inside_the_slice(tmp_path):
    sl = load_slice(fake_locomo(tmp_path), "conv-x", 2)
    assert [s["n"] for s in sl["sessions"]] == [1, 2]
    assert [q["question"] for q in sl["questions"]] == ["q1", "q3"]          # q2 needs session 3, q4 has no evidence

def test_messages_roles_and_granularity(tmp_path):
    sl = load_slice(fake_locomo(tmp_path), "conv-x", 1)
    s = sl["sessions"][0]["turns"]
    whole = to_messages(s, "Ann", "session"); pairs = to_messages(s, "Ann", "pair")
    assert len(whole) == 1 and len(whole[0]) == 3 and len(pairs) == 2 and len(pairs[1]) == 1
    assert whole[0][0] == {"role": "user", "content": "Ann: hello 1"} and whole[0][1]["role"] == "assistant"

def test_extrapolate_is_linear_in_turns():
    out = extrapolate({"calls": 10, "seconds": 100}, measured_turns=50, measured_sessions=2)
    assert abs(out["calls"] - 10 / 50 * 5882) < 1e-6

def test_meter_records_characters_and_options_sent():
    class Client:
        def chat(self, *a, **k): return {"prompt_eval_count": 5, "eval_count": 1}
    fake = types.SimpleNamespace(Client=Client)
    m = Meter(); m.install(fake)
    Client().chat(model="x", messages=[{"role": "system", "content": "abcd"}, {"role": "user", "content": "efghij"}], options={"num_ctx": 2048})
    assert m.calls[0]["chars"] == 10 and m.calls[0]["options"] == {"num_ctx": 2048}

def test_meter_counts_calls_and_tokens_from_either_response_style():
    class Resp:                                    # attribute-style response
        prompt_eval_count, eval_count = 100, 20
    class Client:
        def chat(self, *a, **k): return Resp() if k.get("style") == "obj" else {"prompt_eval_count": 7, "eval_count": 3}
    fake = types.SimpleNamespace(Client=Client)
    m = Meter(); m.install(fake)
    c = Client(); c.chat(style="obj"); mark = m.mark(); c.chat(); c.chat()
    assert len(m.calls) == 3 and m.since(mark) == {"calls": 2, "seconds": m.since(mark)["seconds"], "prompt_tokens": 14, "output_tokens": 6}
    assert m.calls[0]["prompt_tokens"] == 100 and m.calls[0]["output_tokens"] == 20


def test_search_compat_handles_old_and_new_mem0_signatures_and_result_shapes():
    from spike.common import search_compat, result_texts
    class New:                                       # Mem0 2.x: rejects a top-level user_id
        def search(self, q, **kw):
            if "user_id" in kw: raise ValueError("Top-level entity parameters not supported; use filters")
            assert kw["filters"] == {"user_id": "u"}; return {"results": [{"memory": "likes tea"}, {"memory": "has a dog"}]}
    class Old:                                       # older Mem0: wants user_id, returns a plain list
        def search(self, q, user_id=None, limit=None):
            assert user_id == "u"; return [{"memory": "a"}, {"text": "b"}]
    assert result_texts(search_compat(New(), "q", "u", 5)) == ["likes tea", "has a dog"]
    class OldStrict(Old):
        def search(self, q, user_id=None, limit=None):
            return super().search(q, user_id=user_id, limit=limit)
    class OldKw:
        def search(self, q, user_id=None, limit=None):
            assert user_id == "u"; return [{"memory": "a"}, {"text": "b"}]
    assert result_texts(search_compat(OldKw(), "q", "u", 5)) == ["a", "b"]
