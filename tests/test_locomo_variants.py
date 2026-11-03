import json, hashlib
import numpy as np
from eval.locomo_variants import chance_hit, conv_data, main

def tiny(tmp_path):
    turn = lambda sp, i, w: {"speaker": sp, "dia_id": f"D1:{i}", "text": w}
    conv = {"speaker_a": "A", "speaker_b": "B",
            "session_1_date_time": "x", "session_1": [turn("A", i, f"alpha {i}") for i in range(1, 9)],
            "session_2_date_time": "x", "session_2": [turn("B", i, f"beta {i}") for i in range(1, 7)],
            "session_3_date_time": "x", "session_3": [turn("A", i, f"gamma {i}") for i in range(1, 5)]}
    qa = [{"question": "alpha 3", "answer": "a", "evidence": ["D1:3"], "category": 1},
          {"question": "beta 2", "answer": "a", "evidence": ["D2:2", "D3:1"], "category": 2},
          {"question": "none", "answer": "a", "category": 5}]
    p = tmp_path / "l.json"; p.write_text(json.dumps([{"sample_id": "c", "conversation": conv, "qa": qa}])); return p

def test_chance_formula():
    assert abs(chance_hit(19, 1, 5) - 5 / 19) < 1e-9
    assert chance_hit(10, 10, 1) == 1.0 and abs(chance_hit(4, 2, 2) - (1 - 1 / 6)) < 1e-9
    assert chance_hit(3, 1, 5) == 1.0

def test_conv_data_maps_evidence_to_zero_based_sessions_and_skips_unanswerable(tmp_path):
    sessions, qs = conv_data(json.load(open(tiny(tmp_path)))[0])
    assert len(sessions) == 3 and [q["ev"] for q in qs] == [{0}, {1, 2}]

def test_main_runs_and_a_word_matching_encoder_finds_the_evidence(tmp_path, capsys):
    def word_enc(texts):
        vocab = ["alpha", "beta", "gamma"]
        v = np.array([[t.lower().count(w) + 1e-3 for w in vocab] for t in texts], dtype="float32")
        return v / np.linalg.norm(v, axis=1, keepdims=True)
    r = main(encode256=word_enc, encode512=word_enc, locomo=str(tiny(tmp_path)), argv=[])
    out = capsys.readouterr().out
    assert "chance (random ranking" in out and "matched windows only" in out
    assert r["n_q"] == 2 and np.mean(r["hit"][r["names"][2]][1]) == 1.0 and np.mean(r["hit"][r["names"][3]][1]) == 1.0
