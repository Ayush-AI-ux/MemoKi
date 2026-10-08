import json
from core.cues import normalize_entity
from core.types import Conversation
from pipeline.cue_cache import extract_entities_cached, cache_file, load_entity_cache

class FakeNER:
    tag = "fake|v1|100"
    def __init__(self): self.calls = 0
    def __call__(self, texts):
        self.calls += len(texts)
        return [frozenset(w.lower() for w in t.split() if w[:1].isupper()) for t in texts]

def convs(n=10):
    return [Conversation(f"c{i}", f"Alice met Bob{i} in Paris", float(i)) for i in range(n)]

def test_normalize():
    assert normalize_entity("the 'Effective Time Management'") == "effective time management"
    assert normalize_entity("  St. Mary\u2019s Church ") == "st. marys church"
    assert normalize_entity("A  Big   Day") == "big day"
    assert normalize_entity("'the'") == "the" or normalize_entity("'the'") == ""

def test_extract_fills_cues_and_caches(tmp_path):
    ner, cs = FakeNER(), convs()
    extract_entities_cached(cs, ner, tmp_path, chunk=4, log=lambda *_: None)
    assert cs[3].cues["entities"] == frozenset({"alice", "bob3", "paris"})
    assert ner.calls == 10
    ner2, cs2 = FakeNER(), convs()
    extract_entities_cached(cs2, ner2, tmp_path, chunk=4, log=lambda *_: None)
    assert ner2.calls == 0 and cs2[0].cues["entities"] == cs[0].cues["entities"]

def test_resume_after_interruption(tmp_path):
    ner, cs = FakeNER(), convs()
    extract_entities_cached(cs, ner, tmp_path, chunk=4, limit=5, log=lambda *_: None)
    assert ner.calls == 5
    f = cache_file(tmp_path, ner)
    with open(f, "a") as fh: fh.write('{"id": "c9", "e": ["half')   # simulated crash mid-write
    ner2, cs2 = FakeNER(), convs()
    extract_entities_cached(cs2, ner2, tmp_path, chunk=4, log=lambda *_: None)
    assert ner2.calls == 5                                # only the 5 missing ones
    assert len(load_entity_cache(f)) == 10

def test_deterministic_cache_content(tmp_path):
    ner = FakeNER()
    extract_entities_cached(convs(), ner, tmp_path, log=lambda *_: None)
    lines = [json.loads(l) for l in open(cache_file(tmp_path, ner))]
    assert all(l["e"] == sorted(l["e"]) for l in lines)
