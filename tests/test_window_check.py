import hashlib
import numpy as np
from core.types import Conversation
from eval.window_check import chunk_words, window_vectors, sign_test

def bow(texts, dim=16384):
    out = np.zeros((len(texts), dim), dtype="float32")
    for i, t in enumerate(texts):
        for w in t.split():
            out[i, int(hashlib.md5(w.encode()).hexdigest()[:6], 16) % dim] += 1
    return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)

def test_chunk_words():
    assert chunk_words("a b c d e", 2) == ["a b", "c d", "e"] and chunk_words("", 5) == [""]
    assert len(chunk_words(" ".join(["x"] * 500), 150)) == 4

def test_window_cache_encodes_once_and_resumes(tmp_path):
    calls = []
    def enc(texts): calls.append(len(texts)); return bow(texts)
    convs = [Conversation(f"s{i}", " ".join(f"w{i}x{j}" for j in range(300))) for i in range(6)]
    v = window_vectors(convs[:3], enc, tmp_path, 150, chunk=2, log=lambda *_: None)
    assert sum(calls) == 6 and v["s0"].shape[0] == 2                  # 3 sessions x 2 windows
    v2 = window_vectors(convs, enc, tmp_path, 150, chunk=2, log=lambda *_: None)
    assert sum(calls) == 12 and len(v2) == 6                          # only the 3 new sessions were encoded
    window_vectors(convs, enc, tmp_path, 150, chunk=2, log=lambda *_: None)
    assert sum(calls) == 12                                           # fully cached

def test_sign_test():
    a, b = np.array([1, 1, 1, 0, 1.0]), np.array([0, 0, 1, 0, 0.0])
    assert sign_test(a, b)[:2] == (3, 0) and abs(sign_test(a, b)[2] - 0.25) < 1e-9
