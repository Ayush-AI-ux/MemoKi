"""Compare embedding truncation length. Run: python -m eval.seqlen_check"""
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
cache = ROOT / d["paths"]["cache_dir"]
seed, k, size = cfg["seeds"][0], cfg["top_k"], 1000
store = build_store(pool, qs, size, seed)
print(f"size {size}, {len(qs)} queries")
for max_len in (256, 512):
    emb = Embedder(cfg["embedding_model"], max_len)
    embed_queries(qs, emb.encode)
    embed_conversations(store, emb.encode, cache, cfg["embedding_model"], max_len)
    bf = BruteForce(); bf.build(store)
    rec = [recall_at_k([r.conv_id for r in bf.retrieve(q.text, q.embedding, k)[0]], q.relevant_ids, k) for q in qs]
    print(f"max_seq_length={max_len}: Recall@5 = {np.mean(rec):.3f}")
