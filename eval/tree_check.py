"""Step 2.3 check: tree WITHOUT cues vs brute force, on the VALIDATION split.
Run after:  python -m pipeline.embed_pool      then:  python -m eval.tree_check
Options:    --n-queries 100   --beams 1 3 5 10   --seed-index 0"""
import argparse, time
import numpy as np
from core.config import ROOT
from pipeline.config import load_all, path
from pipeline.loaders import load_longmemeval
from pipeline.store import load_splits, select_queries, build_store
from pipeline.embed import Embedder, embed_conversations, embed_queries
from baselines.brute_force import BruteForce
from baselines.centroid_tree import CentroidTree
from eval.metrics import recall_at_k

def evaluate(method, qs, k):
    rec, comps, vis, lat = [], [], [], []
    for q in qs:
        res, c = method.retrieve(q.text, q.embedding, k)
        rec.append(recall_at_k([r.conv_id for r in res], q.relevant_ids, k))
        comps.append(c.embedding_comparisons); vis.append(c.nodes_visited); lat.append(c.latency_ms)
    return np.mean(rec), np.mean(comps), np.median(lat)

def main(encode_fn=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-queries", type=int, default=100)
    ap.add_argument("--beams", type=int, nargs="+", default=[1, 3, 5, 10])
    ap.add_argument("--seed-index", type=int, default=0)
    a = ap.parse_args()
    cfg = load_all(); d = cfg["data"]
    pool, queries = load_longmemeval(path(cfg, "longmemeval"))
    splits = load_splits(path(cfg, "splits"))
    qs = select_queries(queries, splits["validation"], set(pool), a.n_queries,
                        d["max_evidence_per_query"], d["query_seed"])
    enc = encode_fn or Embedder(cfg["embedding_model"], d["max_seq_length"]).encode
    embed_queries(qs, enc)
    cache, k, seed, B = ROOT / d["paths"]["cache_dir"], cfg["top_k"], cfg["seeds"][a.seed_index], cfg["tree"]["max_children"]
    MF = cfg["tree"].get("min_fill", 0.0)
    n_ev = len({r for q in qs for r in q.relevant_ids})
    print(f"VALIDATION split | {len(qs)} queries ({n_ev} evidence sessions) | seed {seed} | k={k} | max_children={B} | min_fill={MF}")
    print(f"{'size':>6}  {'method':<14}{'Recall@5':>9}{'comparisons':>12}{'% of brute':>11}{'median ms':>10}")
    shapes = []
    for size in cfg["store_sizes"]:
        if size < n_ev:
            print(f"{size:>6}  skipped (needs at least {n_ev} sessions for the evidence; use --n-queries smaller)")
            continue
        try:
            store = build_store(pool, qs, size, seed)
        except ValueError as e:
            print(f"{size:>6}  skipped ({e})")
            continue
        embed_conversations(store, enc, cache, cfg["embedding_model"], d["max_seq_length"])
        bf = BruteForce(); bf.build(store)
        r0, c0, l0 = evaluate(bf, qs, k)
        print(f"{size:>6}  {'brute force':<14}{r0:>9.3f}{c0:>12.0f}{'100%':>11}{l0:>10.3f}")
        for beam in a.beams:
            t = CentroidTree(B, beam, MF)
            t0 = time.perf_counter(); t.build(store); bt = time.perf_counter() - t0
            r, c, l = evaluate(t, qs, k)
            print(f"{'':>6}  {'tree beam=' + str(beam):<14}{r:>9.3f}{c:>12.0f}{c / c0:>11.1%}{l:>10.3f}")
        shapes.append((size, t.tree.stats(), bt))
    print("\ntree shape (balanced = every leaf at the same depth):")
    for size, s, bt in shapes:
        print(f"  {size:>6}: height {s['height']}, internal nodes {s['internal_nodes']}, children per node "
              f"mean {s['mean_children']} (min {s['min_children']}, max {s['max_children']}), "
              f"uniform depth {s['uniform_depth']}, build {bt:.1f} s")

if __name__ == "__main__":
    main()
