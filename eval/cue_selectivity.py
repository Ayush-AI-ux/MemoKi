"""Do named-entity cues find the right conversation, and how much do they prune?
Run: python -m eval.cue_selectivity"""
import re, random, time
import numpy as np, spacy
from pipeline.config import load_all, path
from pipeline.loaders import load_longmemeval
from pipeline.store import load_splits, select_queries

NAMED = {"PERSON","ORG","GPE","LOC","FAC","PRODUCT","EVENT","WORK_OF_ART","NORP"}
QUOTES = re.compile(r"[\"'\u201c\u201d\u2018\u2019]")

def norm(s):
    s = QUOTES.sub("", s.lower().strip())
    s = re.sub(r"^(the|a|an)\s+", "", s)
    return re.sub(r"\s+", " ", s).strip()

cfg = load_all(); d = cfg["data"]
pool, queries = load_longmemeval(path(cfg, "longmemeval"))
splits = load_splits(path(cfg, "splits"))
qs = select_queries(queries, splits["validation"], set(pool), 100, d["max_evidence_per_query"], d["query_seed"])
ev = sorted({r for q in qs for r in q.relevant_ids})
others = sorted(i for i in pool if i not in set(ev))
random.Random(0).shuffle(others)
distract = others[:400]
ids = ev + distract
print(f"{len(qs)} queries | {len(ev)} evidence sessions | {len(distract)} distractor sessions")

nlp = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer", "tagger", "attribute_ruler"])
nlp.max_length = 200000
t0 = time.perf_counter(); ents = {}
for i, doc in zip(ids, nlp.pipe([pool[i].text[:20000] for i in ids], batch_size=16)):
    ents[i] = {norm(e.text) for e in doc.ents if e.label_ in NAMED} - {""}
dt = time.perf_counter() - t0
print(f"spaCy: {len(ids)/dt:.1f} sessions/sec -> ~{10000/(len(ids)/dt)/60:.0f} min for 10,000 sessions")
print(f"entities per session: mean {np.mean([len(v) for v in ents.values()]):.1f}, median {np.median([len(v) for v in ents.values()]):.0f}")

qdocs = list(nlp.pipe([q.text for q in qs]))
n_with = n_hit = n_text = 0; sel = []
for q, doc in zip(qs, qdocs):
    qe = {norm(e.text) for e in doc.ents if e.label_ in NAMED} - {""}
    if not qe:
        continue
    n_with += 1
    if qe & set().union(*(ents[r] for r in q.relevant_ids)):
        n_hit += 1
    low = " ".join(QUOTES.sub("", pool[r].text.lower()) for r in q.relevant_ids)
    if any(e in low for e in qe):
        n_text += 1
    sel.append(np.mean([bool(ents[i] & qe) for i in distract]))
print(f"\nqueries with a named entity: {n_with}/{len(qs)} ({n_with/len(qs):.0%})")
print(f"cue recall (entity found in evidence's entity set): {n_hit}/{n_with} = {n_hit/n_with:.0%}")
print(f"loose recall (entity text appears anywhere in evidence): {n_text}/{n_with} = {n_text/n_with:.0%}")
print(f"selectivity (share of distractors sharing an entity): mean {np.mean(sel):.1%}, median {np.median(sel):.1%}")
print(f"usable coverage (query has entity AND evidence matches): {n_hit/len(qs):.0%} of all queries")
