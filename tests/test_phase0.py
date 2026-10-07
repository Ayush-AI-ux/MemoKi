import numpy as np
from core.config import load_config
from core.types import Conversation
from baselines.brute_force import BruteForce
from eval.metrics import recall_at_k

def _make(n=200, d=16, seed=0):
    rng = np.random.default_rng(seed)
    return [Conversation(f"c{i}", f"text {i}", float(i), rng.normal(size=d)) for i in range(n)]

def test_config_loads():
    cfg = load_config()
    assert cfg["top_k"] == 5 and len(cfg["seeds"]) >= 3

def test_brute_force_finds_itself_and_counts():
    convs = _make()
    bf = BruteForce(); bf.build(convs)
    res, c = bf.retrieve("q", convs[42].embedding, k=5)
    assert res[0].conv_id == "c42"
    assert c.embedding_comparisons == 200 and c.llm_calls == 0
    assert recall_at_k([r.conv_id for r in res], {"c42"}) == 1.0

def test_deterministic():
    convs = _make()
    bf = BruteForce(); bf.build(convs)
    a, _ = bf.retrieve("q", convs[7].embedding)
    b, _ = bf.retrieve("q", convs[7].embedding)
    assert [r.conv_id for r in a] == [r.conv_id for r in b]
