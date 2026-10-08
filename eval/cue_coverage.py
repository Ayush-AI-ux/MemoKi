"""How many queries carry usable cues? Run: python -m eval.cue_coverage"""
import collections, spacy
from pipeline.config import load_all, path
from pipeline.loaders import load_longmemeval

cfg = load_all()
_, queries = load_longmemeval(path(cfg, "longmemeval_oracle"))
nlp = spacy.load("en_core_web_sm")
NAMED = {"PERSON","ORG","GPE","LOC","FAC","PRODUCT","EVENT","WORK_OF_ART","NORP"}
TIME = {"DATE","TIME"}
tot = collections.Counter(); by_type = collections.defaultdict(collections.Counter)
samples = []
for q, doc in zip(queries, nlp.pipe([q.text for q in queries], batch_size=64)):
    named = [e.text for e in doc.ents if e.label_ in NAMED]
    timed = [e.text for e in doc.ents if e.label_ in TIME]
    flags = {"n": 1, "named": bool(named), "time": bool(timed), "any": bool(named or timed)}
    for key, v in flags.items():
        tot[key] += v; by_type[q.qtype][key] += v
    if len(samples) < 12 and (named or timed): samples.append((q.text, named, timed))
n = tot["n"]
print(f"{n} queries | named entity: {tot['named']/n:.0%} | time expression: {tot['time']/n:.0%} | any cue: {tot['any']/n:.0%}")
print(f"{'type':28} {'n':>4} {'named':>7} {'time':>6} {'any':>6}")
for t, c in sorted(by_type.items()):
    print(f"{t:28} {c['n']:>4} {c['named']/c['n']:>7.0%} {c['time']/c['n']:>6.0%} {c['any']/c['n']:>6.0%}")
print("\nSamples:")
for text, named, timed in samples: print(" -", text[:90], "| named:", named, "| time:", timed)
