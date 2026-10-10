"""Step 3.0: every method side by side on the VALIDATION split (same queries, same stores).
Run: python -m eval.compare_check            (sizes 1000 5000 10000)
     python -m eval.compare_check --sizes 10000
Rows: brute force | FAISS flat | FAISS HNSW (3 efSearch values) | flat + the SAME cue filter |
      tree without cues | cue tree.   'e2e' = search + query-cue preparation (ms, median).
At the largest size a paired exact (McNemar sign) test compares the cue tree with each opponent."""
import argparse
from math import comb
import numpy as np
from core.config import ROOT
from core.cues import SpacyNER
from pipeline.config import load_all, path
from pipeline.loaders import load_longmemeval
from pipeline.store import load_splits, select_queries, build_store
from pipeline.embed import Embedder, embed_conversations, embed_queries
from pipeline.cue_cache import extract_entities_cached, cache_file, load_entity_cache
from baselines.brute_force import BruteForce
from baselines.centroid_tree import CentroidTree
from baselines.cue_tree import CueTree
from baselines.flat_cue import FlatCueFilter, HybridCueHNSW
from baselines.faiss_index import FaissFlat, FaissHNSW
from eval.cue_tree_check import run, mean_or_nan

def sign_test(a_hits, b_hits):
    """Exact two-sided sign test on discordant pairs. Returns (A only, B only, p)."""
    a_only = int(((a_hits == 1) & (b_hits == 0)).sum()); b_only = int(((a_hits == 0) & (b_hits == 1)).sum())
    n = a_only + b_only
    if n == 0:
        return a_only, b_only, 1.0
    p = min(1.0, 2 * sum(comb(n, i) for i in range(min(a_only, b_only) + 1)) / 2 ** n)
    return a_only, b_only, p

def show(label, r, cue_mask, size=None, rows=None):
    e2e = np.median(r["ms"] + r["prep"])
    if rows is not None:
        rows.append({"size": size, "method": label, "recall_all": r["hit"].mean(),
                     "recall_cue": mean_or_nan(r["hit"], cue_mask), "recall_oth": mean_or_nan(r["hit"], ~cue_mask),
                     "comps_all": r["comps"].mean(), "comps_cue": mean_or_nan(r["comps"], cue_mask),
                     "comps_oth": mean_or_nan(r["comps"], ~cue_mask), "cue_checks": r["chk"].mean(),
                     "prep_ms": r["prep"].mean(), "search_ms": float(np.median(r["ms"])), "e2e_ms": float(e2e),
                     "n_cue_queries": int(cue_mask.sum())})
    print(f"  {label:<24}{r['hit'].mean():>7.3f}{mean_or_nan(r['hit'], cue_mask):>8.3f}{mean_or_nan(r['hit'], ~cue_mask):>8.3f}"
          f"{r['comps'].mean():>8.0f}{mean_or_nan(r['comps'], cue_mask):>8.0f}{mean_or_nan(r['comps'], ~cue_mask):>8.0f}"
          f"{r['chk'].mean():>8.0f}{r['prep'].mean():>7.2f}{np.median(r['ms']):>8.3f}{e2e:>8.3f}")

def main(encode_fn=None, ner=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[1000, 5000, 10000])
    ap.add_argument("--n-queries", type=int, default=100)
    ap.add_argument("--efs", type=int, nargs="+", default=[16, 64, 128])
    ap.add_argument("--beams", type=int, nargs="+", default=[3, 10, 20])
    ap.add_argument("--csv", default=None, help="also save every row to this CSV file")
    ap.add_argument("--audit-margin", type=float, default=None, help="also evaluate trees after one offline audit/re-filing pass with this margin")
    a = ap.parse_args()
    cfg = load_all(); d = cfg["data"]
    pool, queries = load_longmemeval(path(cfg, "longmemeval"))
    splits = load_splits(path(cfg, "splits"))
    qs = select_queries(queries, splits["validation"], set(pool), a.n_queries, d["max_evidence_per_query"], d["query_seed"])
    enc = encode_fn or Embedder(cfg["embedding_model"], d["max_seq_length"]).encode
    embed_queries(qs, enc)
    cache, k, seed = ROOT / d["paths"]["cache_dir"], cfg["top_k"], cfg["seeds"][0]
    B, MF = cfg["tree"]["max_children"], cfg["tree"].get("min_fill", 0.0)
    slack = cfg.get("cues", {}).get("time_slack_days", 7.0)
    own_ner = ner is None
    ner = ner or SpacyNER(); ner(["warm up"])
    print(f"VALIDATION | {len(qs)} queries | seed {seed} | tree B={B} min_fill={MF} | slack {slack} d")
    print("R@5 = Recall@5 (all | queries WITH a usable cue | WITHOUT); comps = mean embedding comparisons (all | cue q | other q);")
    rows = []
    print("cue-chk = cheap cue checks; prep = query-cue extraction ms (first time); search = ms excluding prep; e2e = search + prep")
    for size in a.sizes:
        try:
            store = build_store(pool, qs, size, seed)
        except ValueError as e:
            print(f"\nsize {size}: skipped ({e})"); continue
        embed_conversations(store, enc, cache, cfg["embedding_model"], d["max_seq_length"])
        if own_ner:
            done = load_entity_cache(cache_file(cache, ner))
            if sum(c.id not in done for c in store) > 300:
                print("entity cues missing; run: python -m pipeline.extract_cues --all --procs 3"); return
        extract_entities_cached(store, ner, cache, log=lambda *_: None)
        bf, ff, hn = BruteForce(), FaissFlat(), FaissHNSW(32, 64, 64)
        flat = FlatCueFilter(True, True, slack, ner)
        nocue = CentroidTree(B, 10, MF)
        cue = CueTree(B, 3, MF, True, True, False, 0.35, slack, ner)
        hyb = HybridCueHNSW(True, True, slack, ner, 32, 64, 64)
        for m in (bf, ff, hn, flat, hyb, nocue, cue): m.build(store)
        cue_a = nocue_a = None
        if a.audit_margin is not None:
            cue_a = CueTree(B, 3, MF, True, True, False, 0.35, slack, ner); cue_a.build(store)
            nocue_a = CentroidTree(B, 10, MF); nocue_a.build(store)
            for t in (cue_a, nocue_a):
                st = t.tree.audit_refile(a.audit_margin, 1)
            print(f"  (audit margin {a.audit_margin}: re-filed {len(st['moved'])} of {len(store)} leaves, offline)")
        active = np.array([cue.prepare(q.text, q.timestamp)[0].active for q in qs])
        print(f"\nsize {size} | queries with a usable cue: {active.sum()}/{len(qs)}")
        print(f"  {'method':<24}{'R@5':>7}{'cue q':>8}{'oth q':>8}{'comps':>8}{'cue q':>8}{'oth q':>8}{'cue-chk':>8}{'prep':>7}{'search':>8}{'e2e':>8}")
        res = {}
        res["brute force"] = run(bf, qs, k); res["faiss flat"] = run(ff, qs, k)
        for ef in a.efs:
            hn.ef_search = ef; res[f"hnsw ef={ef}"] = run(hn, qs, k)
        res["flat + cue filter"] = run(flat, qs, k)
        for ef in a.efs:
            hyb.ef_search = ef; res[f"filter + hnsw ef={ef}"] = run(hyb, qs, k)
        for beam in a.beams:
            nocue.beam = beam; res[f"tree no cues b={beam}"] = run(nocue, qs, k)
        for beam in a.beams:
            cue.beam = beam; res[f"CUE TREE b={beam}"] = run(cue, qs, k)
        if cue_a is not None:
            for beam in a.beams:
                nocue_a.beam = cue_a.beam = beam
                res[f"tree no cues + audit b={beam}"] = run(nocue_a, qs, k)
                res[f"CUE TREE + audit b={beam}"] = run(cue_a, qs, k)
        for label, r in res.items():
            show(label, r, active, size, rows)
        if size == max(a.sizes):
            def paired(mine_name, opponents):
                if mine_name not in res:
                    return
                mine = res[mine_name]
                for title, mask in (("ALL queries", np.ones(len(qs), dtype=bool)), ("queries WITH a usable cue", active), ("queries WITHOUT a cue", ~active)):
                    print(f"\npaired exact test at size {size}, {title} (n={int(mask.sum())}): {mine_name} vs each opponent")
                    for name in opponents:
                        if name in res:
                            ao, bo, p = sign_test(mine["hit"][mask], res[name]["hit"][mask])
                            print(f"  vs {name:<26} mine only {ao:>3} | opponent only {bo:>3} | p = {p:.3f}")
            b10 = 10 if 10 in a.beams else a.beams[0]
            paired(f"CUE TREE b={b10}", ("brute force", "hnsw ef=64", "flat + cue filter", "filter + hnsw ef=64", "filter + hnsw ef=16", f"tree no cues b={b10}"))
            paired(f"CUE TREE + audit b={b10}", (f"CUE TREE b={b10}", "brute force", "hnsw ef=64", "filter + hnsw ef=16", "filter + hnsw ef=64"))
            paired(f"tree no cues + audit b={b10}", (f"tree no cues b={b10}", "hnsw ef=16", "hnsw ef=64"))

    if a.csv and rows:
        import csv
        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        print(f"\nsaved {len(rows)} rows to {a.csv}")

if __name__ == "__main__":
    main()
