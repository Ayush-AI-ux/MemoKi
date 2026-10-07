"""Deterministic splits, fixed query sets, and nested stores of any size."""
import json, random
from core.types import Conversation, Query

def make_splits(queries: list[Query], val_frac: float, seed: int) -> dict:
    ids = sorted(q.id for q in queries)
    random.Random(seed).shuffle(ids)
    n_val = int(round(len(ids) * val_frac))
    return {"validation": sorted(ids[:n_val]), "test": sorted(ids[n_val:])}

def save_splits(splits, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(splits, open(path, "w"), indent=1)

def load_splits(path) -> dict:
    return json.load(open(path))

def select_queries(queries, split_ids, pool_ids, n, max_evidence, seed) -> list[Query]:
    """Fixed query set: answerable queries with few evidence sessions, picked with a
    constant seed so EVERY store size and EVERY method sees the same queries."""
    allowed = set(split_ids)
    ok = sorted((q for q in queries if q.id in allowed and 1 <= len(q.relevant_ids) <= max_evidence
                 and q.relevant_ids <= pool_ids), key=lambda q: q.id)
    random.Random(seed).shuffle(ok)
    return ok[:n]

def build_store(pool: dict, queries: list[Query], size: int, seed: int) -> list[Conversation]:
    """Store = all evidence for `queries` + seeded distractors, shuffled.
    Distractors come from a fixed seeded order, so store(100) is a subset of store(500)
    is a subset of store(1000)... (nested stores make scaling curves comparable)."""
    evidence = sorted({r for q in queries for r in q.relevant_ids})
    if size < len(evidence):
        raise ValueError(f"size {size} < {len(evidence)} evidence sessions; lower n_queries")
    ev = set(evidence)
    distractors = sorted(i for i in pool if i not in ev)
    random.Random(seed).shuffle(distractors)
    need = size - len(evidence)
    if need > len(distractors):
        raise ValueError(f"pool has only {len(distractors)} distractors, need {need}. "
                         f"Use the full haystack file or add an extra source (WildChat/ShareGPT).")
    ids = evidence + distractors[:need]
    random.Random(seed + 1).shuffle(ids)          # evidence must not sit at the front
    return [pool[i] for i in ids]
