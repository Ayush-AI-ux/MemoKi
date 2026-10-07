"""Brute-force baseline on real data at every store size.
Run: python -m eval.sanity_check"""
import numpy as np
from core.config import ROOT
from pipeline.config import load_all, path
from pipeline.loaders import load_longmemeval
from pipeline.store import load_splits, select_queries, build_store
from pipeline.embed import Embedder, embed_conversations, embed_queries
from baselines.brute_force import BruteForce
from eval.metrics import recall_at_k

cfg = load_all(); d = cfg["data"]
pool, queries = load_longmemeval(path(cfg, "longmemeval"))
splits = load_splits(path(cfg, "splits"))
qs = select_queries(queries, splits["test"], set(pool), d["n_queries"],
                    d["max_evidence_per_query"], d["query_seed"])
emb = Embedder(cfg["embedding_model"], d["max_seq_length"])
embed_queries(qs, emb.encode)
cache = ROOT / d["paths"]["cache_dir"]
seed, k = cfg["seeds"][0], cfg["top_k"]

print(f"{len(qs)} queries | seed {seed} | k={k}")
print(f"{'size':>6} {'Recall@5':>9} {'comparisons':>12} {'median ms':>10}")
for size in cfg["store_sizes"]:
    store = build_store(pool, qs, size, seed)
    embed_conversations(store, emb.encode, cache, cfg["embedding_model"], d["max_seq_length"])
    bf = BruteForce(); bf.build(store)
    rec, comps, lat = [], [], []
    for q in qs:
        res, c = bf.retrieve(q.text, q.embedding, k)
        rec.append(recall_at_k([r.conv_id for r in res], q.relevant_ids, k))
        comps.append(c.embedding_comparisons); lat.append(c.latency_ms)
    print(f"{size:>6} {np.mean(rec):>9.3f} {np.mean(comps):>12.0f} {np.median(lat):>10.3f}")
