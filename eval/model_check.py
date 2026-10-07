"""Compare embedding models at size 1000. Run: python -m eval.model_check"""
import numpy as np
from core.config import ROOT
from pipeline.config import load_all, path
from pipeline.loaders import load_longmemeval
from pipeline.store import load_splits, select_queries, build_store
from pipeline.embed import Embedder, embed_conversations
from baselines.brute_force import BruteForce
from eval.metrics import recall_at_k

cfg = load_all(); d = cfg["data"]
pool, queries = load_longmemeval(path(cfg, "longmemeval"))
splits = load_splits(path(cfg, "splits"))
qs = select_queries(queries, splits["test"], set(pool), 100,
                    d["max_evidence_per_query"], d["query_seed"])
cache = ROOT / d["paths"]["cache_dir"]
seed, k, size, L = cfg["seeds"][0], cfg["top_k"], 1000, 256
store = build_store(pool, qs, size, seed)
print(f"size {size}, {len(qs)} queries, max_seq_length={L}")
BGE = "Represent this sentence for searching relevant passages: "
models = [("sentence-transformers/all-MiniLM-L6-v2", ""),
          ("BAAI/bge-small-en-v1.5", BGE),
          ("sentence-transformers/all-mpnet-base-v2", "")]
for name, prefix in models:
    emb = Embedder(name, L)
    qv = emb.encode([prefix + q.text for q in qs])
    for q, v in zip(qs, qv): q.embedding = v
    embed_conversations(store, emb.encode, cache, name, L)
    bf = BruteForce(); bf.build(store)
    rec = [recall_at_k([r.conv_id for r in bf.retrieve(q.text, q.embedding, k)[0]], q.relevant_ids, k) for q in qs]
    print(f"{name}: Recall@5 = {np.mean(rec):.3f}")
