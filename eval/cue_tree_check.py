"""Step 2.4 check on the VALIDATION split: does the cue tree reach the same recall at lower cost?
Needs entity cues for the whole pool first:  python -m pipeline.extract_cues --all --procs 3
Run:  python -m eval.cue_tree_check                       (sizes 1000 5000 10000, beams 3 10 20)
      python -m eval.cue_tree_check --taus 0.30 0.40      (also test the flat-search fallback)
Pre-registered rule for choosing the fallback threshold tau (fixed BEFORE seeing results):
among the tau values tried, at the largest size and beam 10 with names+time, take the one with the
lowest mean comparisons whose Recall@5 is within 2 points of brute force; if none qualifies, take
the one with the highest Recall@5. Switch settings and slack are NOT tuned here."""
import argparse
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
from eval.metrics import recall_at_k

def run(method, qs, k):
    out = {"hit": [], "comps": [], "chk": [], "prep": [], "ms": [], "fb": []}
    for q in qs:
        res, c = method.retrieve(q.text, q.embedding, k, query_ts=q.timestamp)
        out["hit"].append(recall_at_k([r.conv_id for r in res], q.relevant_ids, k))
        out["comps"].append(c.embedding_comparisons); out["chk"].append(c.cue_checks)
        out["prep"].append(c.prep_ms); out["ms"].append(c.latency_ms); out["fb"].append(c.fallback_triggered)
    return {key: np.array(v) for key, v in out.items()}

def mean_or_nan(a, m):
    return float(a[m].mean()) if m.any() else float("nan")

def row(label, beam, r, cue_mask, show_fb):
    other = ~cue_mask
    fb = f"{r['fb'].mean():>5.0%}" if show_fb else "    -"
    print(f"  {label:<22}{beam:>4}{r['hit'].mean():>8.3f}{mean_or_nan(r['hit'], cue_mask):>9.3f}"
          f"{mean_or_nan(r['hit'], other):>9.3f}{r['comps'].mean():>8.0f}{mean_or_nan(r['comps'], cue_mask):>9.0f}"
          f"{r['chk'].mean():>8.0f}{fb}{r['prep'].mean():>8.2f}{np.median(r['ms']):>8.3f}")

def main(encode_fn=None, ner=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[1000, 5000, 10000])
    ap.add_argument("--beams", type=int, nargs="+", default=[3, 10, 20])
    ap.add_argument("--taus", type=float, nargs="*", default=[])
    ap.add_argument("--n-queries", type=int, default=100)
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
    ner = ner or SpacyNER()
    ner(["warm up"])                      # load the model once so the first query is not charged for it
    print(f"VALIDATION | {len(qs)} queries | seed {seed} | max_children={B} min_fill={MF} | time slack {slack} d")
    print("columns: R@5 = Recall@5 (all | queries WITH a usable cue | queries WITHOUT); comps = embedding comparisons;")
    print("         cue-chk = cheap cue checks; fb = flat-fallback rate; prep = cost of extracting a query's cues (ms, first time only); search = ms excluding prep")
    for size in a.sizes:
        try:
            store = build_store(pool, qs, size, seed)
        except ValueError as e:
            print(f"\nsize {size}: skipped ({e})"); continue
        embed_conversations(store, enc, cache, cfg["embedding_model"], d["max_seq_length"])
        if own_ner:
            done = load_entity_cache(cache_file(cache, ner))
            missing = sum(c.id not in done for c in store)
            if missing > 300:
                print(f"\nsize {size}: {missing} sessions have no entity cues yet. Run:\n"
                      f"  python -m pipeline.extract_cues --all --procs 3"); return
        extract_entities_cached(store, ner, cache, log=lambda *_: None)
        cue = CueTree(B, 3, MF, True, True, False, 0.35, slack, ner); cue.build(store)
        nocue = CentroidTree(B, 3, MF); nocue.build(store)
        bf = BruteForce(); bf.build(store)
        active = np.array([cue.prepare(q.text, q.timestamp)[0].active for q in qs])
        ent_only = np.array([bool(cue.prepare(q.text, q.timestamp)[0].entities) for q in qs])
        rb = run(bf, qs, k)
        print(f"\nsize {size} | queries with a usable cue: {active.sum()}/{len(qs)} (entity: {ent_only.sum()})")
        print(f"  {'variant':<22}{'beam':>4}{'R@5':>8}{'R@5 cue':>9}{'R@5 oth':>9}{'comps':>8}{'comps c':>9}{'cue-chk':>8}{'  fb':>5}{'prep':>8}{'search':>8}")
        row("brute force", "-", rb, active, False)
        for beam in a.beams:
            nocue.beam = cue.beam = beam
            row("tree, no cues", beam, run(nocue, qs, k), active, False)
            for label, un, ut in (("cues: names", True, False), ("cues: names + time", True, True)):
                cue.use_names, cue.use_time, cue.use_fallback = un, ut, False
                row(label, beam, run(cue, qs, k), active, False)
            for tau in a.taus:
                cue.use_names, cue.use_time, cue.use_fallback, cue.tau = True, True, True, tau
                row(f"names+time+fb t={tau}", beam, run(cue, qs, k), active, True)

if __name__ == "__main__":
    main()
