"""Tree STRUCTURE sweep on the VALIDATION split (no cues yet).
Decision rule fixed BEFORE running: pick the variant with the highest 'kept' at beam 10 on the
largest size; differences under 5 points count as ties and are broken toward fewer comparisons
and smaller height. The chosen structure is then frozen and shared by the no-cue tree and the
cue tree, so the ablation stays fair.
Run: python -m eval.tree_sweep   (options: --sizes 1000 10000 --n-queries 100)"""
import argparse
import numpy as np
from core.config import ROOT
from pipeline.config import load_all, path
from pipeline.loaders import load_longmemeval
from pipeline.store import load_splits, select_queries, build_store
from pipeline.embed import Embedder, embed_conversations, embed_queries
from baselines.brute_force import BruteForce
from baselines.centroid_tree import CentroidTree
from eval.metrics import recall_at_k

VARIANTS = [(8, 0.0), (8, 0.35), (16, 0.0), (16, 0.35), (32, 0.35)]   # (max_children, min_fill)
BEAMS = (3, 10)

def run(method, qs, k):
    hits, comps, lat = [], [], []
    for q in qs:
        res, c = method.retrieve(q.text, q.embedding, k)
        hits.append(recall_at_k([r.conv_id for r in res], q.relevant_ids, k))
        comps.append(c.embedding_comparisons); lat.append(c.latency_ms)
    return np.array(hits), np.mean(comps), np.median(lat)

def main(encode_fn=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[1000, 10000])
    ap.add_argument("--n-queries", type=int, default=100)
    a = ap.parse_args()
    cfg = load_all(); d = cfg["data"]
    pool, queries = load_longmemeval(path(cfg, "longmemeval"))
    splits = load_splits(path(cfg, "splits"))
    qs = select_queries(queries, splits["validation"], set(pool), a.n_queries, d["max_evidence_per_query"], d["query_seed"])
    enc = encode_fn or Embedder(cfg["embedding_model"], d["max_seq_length"]).encode
    embed_queries(qs, enc)
    cache, k, seed = ROOT / d["paths"]["cache_dir"], cfg["top_k"], cfg["seeds"][0]
    print(f"VALIDATION | {len(qs)} queries | seed {seed} | 'kept' = share of brute-force hits the tree also hits")
    for size in a.sizes:
        try:
            store = build_store(pool, qs, size, seed)
        except ValueError as e:
            print(f"size {size}: skipped ({e})"); continue
        embed_conversations(store, enc, cache, cfg["embedding_model"], d["max_seq_length"])
        bf = BruteForce(); bf.build(store)
        bh, bc, bl = run(bf, qs, k)
        print(f"\nsize {size}: brute force Recall@5 {bh.mean():.3f}, {bc:.0f} comparisons, {bl:.3f} ms")
        print(f"  {'B':>3} {'min_fill':>8} {'height':>6} {'children':>8} {'beam':>5} {'Recall@5':>9} {'kept':>6} {'comps':>6} {'% brute':>8} {'ms':>6}")
        for B, mf in VARIANTS:
            for beam in BEAMS:
                t = CentroidTree(B, beam, mf); t.build(store)
                h, c, l = run(t, qs, k)
                st = t.tree.stats()
                kept = (h * bh).sum() / max(bh.sum(), 1)
                print(f"  {B:>3} {mf:>8} {st['height']:>6} {st['mean_children']:>8} {beam:>5} {h.mean():>9.3f} {kept:>6.0%} {c:>6.0f} {c / bc:>8.1%} {l:>6.3f}")

if __name__ == "__main__":
    main()
