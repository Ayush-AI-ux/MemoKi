"""One command to inspect data and pre-compute embeddings.
   python -m pipeline.prepare            # stats + splits
   python -m pipeline.prepare --embed    # also embed the largest feasible store (slow, cached)
   python -m pipeline.prepare --oracle   # use the small evidence-only file (dev)"""
import argparse
from collections import Counter
from .config import load_all, path
from .loaders import load_longmemeval, load_locomo
from .store import make_splits, save_splits, select_queries, build_store
from .embed import Embedder, embed_conversations, embed_queries

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embed", action="store_true")
    ap.add_argument("--oracle", action="store_true")
    a = ap.parse_args()
    cfg = load_all(); d = cfg["data"]

    src = path(cfg, "longmemeval_oracle" if a.oracle else "longmemeval")
    pool, queries = load_longmemeval(src)
    print(f"LongMemEval: {len(pool)} unique sessions, {len(queries)} answerable queries")
    print("  query types:", dict(Counter(q.qtype for q in queries)))

    splits = make_splits(queries, d["val_frac"], d["split_seed"])
    save_splits(splits, path(cfg, "splits"))
    print(f"splits: validation={len(splits['validation'])} test={len(splits['test'])} (saved)")

    qs = select_queries(queries, splits["test"], set(pool), d["n_queries"],
                        d["max_evidence_per_query"], d["query_seed"])
    ev = {r for q in qs for r in q.relevant_ids}
    print(f"test query set: {len(qs)} queries, {len(ev)} evidence sessions")

    feasible = []
    for size in cfg["store_sizes"]:
        try:
            build_store(pool, qs, size, seed=cfg["seeds"][0]); feasible.append(size); print(f"  size {size:>6}: OK")
        except ValueError as e:
            print(f"  size {size:>6}: NOT feasible -> {e}")

    lc, lq, _ = load_locomo(path(cfg, "locomo"))
    print(f"LoCoMo: {len(lc)} sessions, {len(lq)} queries")

    if a.embed and feasible:
        biggest = build_store(pool, qs, max(feasible), seed=cfg["seeds"][0])
        emb = Embedder(cfg["embedding_model"], d["max_seq_length"])
        cache = path(cfg, "cache_dir") if "cache_dir" in d["paths"] else None
        from core.config import ROOT
        embed_conversations(biggest, emb.encode, ROOT / d["paths"]["cache_dir"], cfg["embedding_model"], d["max_seq_length"])
        embed_queries(qs, emb.encode)
        print("embeddings ready and cached")

if __name__ == "__main__":
    main()
