"""Gap 3 on the VALIDATION split: does an LLM-free recency rule reduce stale retrieval?
Run: python -m eval.recency_check --csv results/recency_5000.csv     (size 5000, 'top' and 'local' rule variants)
Update pairs: LongMemEval knowledge-update queries (older evidence session vs newer one).
Rule variants: 'top' = bonus for candidates close to the BEST score; 'local' = bonus among candidates close to
EACH OTHER (synopsis wording). Bonus = lambda * relative recency (0 oldest .. 1 newest) inside the close group.
Metrics: stale = older above newer (or older present while the newer is absent); fresh@1; newer found.
ANALYSIS 1 (original): top 5, rule applied to the top 10 candidates.
ANALYSIS 2 (added after analysis 1 was underpowered, so POST-HOC): rule applied to the top --window candidates;
  among pairs where both sessions are in that list, the share with the older one ranked above the newer one.
DIAGNOSTIC (all pairs, no retrieval): query-to-older minus query-to-newer cosine; near-ties are the only cases a
  bonus of at most lambda can flip.
Pre-registered selection, for every analysis and method: among (variant, delta, lambda) whose standard Recall@5 is
at most 1 point below the no-rule baseline, take the lowest stale rate; ties go to 'top', then the smaller lambda,
then the smaller delta. If none lowers the stale rate, the result is negative. Pilot: few pairs."""
import argparse
import numpy as np
from core.config import ROOT
from core.cues import SpacyNER
from core.recency import pair_outcome, pairwise_outcome
from pipeline.config import load_all, path
from pipeline.loaders import load_longmemeval
from pipeline.store import load_splits, select_queries, build_store
from pipeline.embed import Embedder, embed_conversations, embed_queries
from pipeline.cue_cache import extract_entities_cached, cache_file, load_entity_cache
from pipeline.update_pairs import build_update_pairs
from baselines.brute_force import BruteForce
from baselines.cue_tree import CueTree
from baselines.recency_wrapper import RecencyWrapper
from eval.compare_check import sign_test
from eval.metrics import recall_at_k

def std_recall(w, std, k):
    return float(np.mean([recall_at_k([r.conv_id for r in w.retrieve(q.text, q.embedding, k, query_ts=q.timestamp)[0]],
                                      q.relevant_ids, k) for q in std]))

def eval_top5(w, mode, delta, lam, std, pairs, qmap, k):
    w.pool, w.mode, w.delta, w.lam = 10, mode, delta, lam
    outs = []
    for p in pairs:
        q = qmap[p["qid"]]
        ids = [r.conv_id for r in w.retrieve(q.text, q.embedding, k, query_ts=q.timestamp)[0]]
        outs.append(pair_outcome(ids, p["older"], p["newer"]))
    f = lambda key: np.array([o[key] for o in outs], dtype=float)
    return {"recall_std": std_recall(w, std, k), "stale": f("stale"), "fresh1": f("fresh_first"), "found": f("newer_found"), "neither": f("neither")}

def eval_pairwise(w, mode, delta, lam, std, pairs, qmap, k, window):
    w.pool, w.mode, w.delta, w.lam = window, mode, delta, lam
    outs = []
    for p in pairs:
        q = qmap[p["qid"]]
        ids = [r.conv_id for r in w.retrieve(q.text, q.embedding, window, query_ts=q.timestamp)[0]]
        outs.append(pairwise_outcome(ids, p["older"], p["newer"]))
    return {"recall_std": std_recall(w, std, k), "both": np.array([o["both"] for o in outs], dtype=float),
            "stale": np.array([o["stale_pair"] for o in outs], dtype=float)}

def grid(label, w, modes, deltas, lambdas, base_stale, base_recall, run, rows, analysis, meta):
    """Prints one compact matrix per mode and returns the pre-registered selection."""
    best = None
    for mode in modes:
        print(f"  [{label}, rule variant '{mode}']   stale rate (change in standard Recall@5; 'x' = costs more than 1 point)")
        print("   delta\\lambda" + "".join(f"{lm:>16.3f}" for lm in lambdas))
        for dl in deltas:
            cells = []
            for lm in lambdas:
                r = run(mode, dl, lm)
                drop = r["recall_std"] - base_recall; ok = drop >= -0.01 - 1e-9
                cells.append(f"{r['stale_rate']:>9.0%} {drop:+.3f}{'' if ok else 'x'}".rjust(16))
                rows.append({"method": label, "analysis": analysis, "mode": mode, "delta": dl, "lambda": lm,
                             "stale": r["stale_rate"], "recall_std": r["recall_std"], "recall_change": drop,
                             "eligible": ok, "baseline_stale": base_stale, "baseline_recall_std": base_recall, **meta})
                if ok and r["stale_rate"] < base_stale:
                    key = (r["stale_rate"], 0 if mode == "top" else 1, lm, dl)
                    if best is None or key < best[0]:
                        best = (key, mode, dl, lm, r)
            print(f"   {dl:>10.3f}" + "".join(cells))
    return best

def main(encode_fn=None, ner=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=5000)
    ap.add_argument("--n-queries", type=int, default=100)
    ap.add_argument("--beam", type=int, default=10)
    ap.add_argument("--deltas", type=float, nargs="+", default=[0.01, 0.02, 0.05, 0.10])
    ap.add_argument("--lambdas", type=float, nargs="+", default=[0.005, 0.01, 0.02, 0.05])
    ap.add_argument("--modes", nargs="+", default=["top", "local"])
    ap.add_argument("--window", type=int, default=50)
    ap.add_argument("--csv", default=None)
    a = ap.parse_args()
    cfg = load_all(); d = cfg["data"]
    pool, queries = load_longmemeval(path(cfg, "longmemeval"))
    splits = load_splits(path(cfg, "splits")); val = set(splits["validation"])
    std = select_queries([q for q in queries if q.qtype != "knowledge-update"], splits["validation"], set(pool),
                         a.n_queries, d["max_evidence_per_query"], d["query_seed"])
    pairs, st = build_update_pairs(queries, pool, allowed_ids=val)
    print(f"VALIDATION | standard (non-update) queries {len(std)} | knowledge-update queries {st['knowledge_update_queries']} "
          f"-> usable update pairs {len(pairs)} (same timestamp dropped: {st['dropped_same_timestamp']}; "
          f"more than two evidence sessions: {st['pairs_with_more_than_two_evidence']})")
    if len(pairs) < 5:
        print("too few update pairs on the validation split for even a pilot: send me this line"); return
    qmap = {q.id: q for q in queries}
    ku = [qmap[p["qid"]] for p in pairs]
    enc = encode_fn or Embedder(cfg["embedding_model"], d["max_seq_length"]).encode
    embed_queries(std, enc); embed_queries(ku, enc)
    cache, k, seed = ROOT / d["paths"]["cache_dir"], cfg["top_k"], cfg["seeds"][0]
    B, MF = cfg["tree"]["max_children"], cfg["tree"].get("min_fill", 0.0)
    slack = cfg.get("cues", {}).get("time_slack_days", 7.0)
    own_ner = ner is None
    ner = ner or SpacyNER(); ner(["warm up"])
    store = build_store(pool, std + ku, a.size, seed)
    embed_conversations(store, enc, cache, cfg["embedding_model"], d["max_seq_length"])
    if own_ner:
        done = load_entity_cache(cache_file(cache, ner))
        if sum(c.id not in done for c in store) > 300:
            print("entity cues missing; run: python -m pipeline.extract_cues --all --procs 3"); return
    extract_entities_cached(store, ner, cache, log=lambda *_: None)

    emb = {c.id: c.embedding / np.linalg.norm(c.embedding) for c in store}
    gaps = np.array([float(qmap[p["qid"]].embedding / np.linalg.norm(qmap[p["qid"]].embedding) @ (emb[p["older"]] - emb[p["newer"]]))
                     for p in pairs])
    older_higher = gaps > 0
    print(f"\nSIMILARITY-GAP DIAGNOSTIC over all {len(pairs)} pairs (query-to-older minus query-to-newer cosine):")
    print(f"  the OLDER session scores higher in {int(older_higher.sum())} pairs, the newer in {int((~older_higher).sum())}; "
          f"median gap {np.median(gaps):+.3f}")
    print("  among the older-higher pairs, gap within: " + " | ".join(
        f"{t:.2f}: {int(((gaps > 0) & (gaps <= t)).sum())}" for t in (0.01, 0.02, 0.05, 0.10)) +
          f"  (of {int(older_higher.sum())}; only these near-ties can be flipped by a bonus of at most lambda)")

    ts_all = np.sort(np.array([c.timestamp for c in store]))
    mid = lambda t: (np.searchsorted(ts_all, t, "left") + np.searchsorted(ts_all, t, "right")) / 2 / len(ts_all)
    ev_std = {r for q in std for r in q.relevant_ids}
    pct = [mid(c.timestamp) for c in store if c.id in ev_std]
    print(f"\nBIAS CHECK: the evidence sessions of the standard queries sit at mean recency percentile {np.mean(pct):.2f} among the "
          f"{len(store)} stored sessions (0.50 = no timestamp bias; clearly above 0.50 would mean a recency bonus can help on this benchmark by accident)")
    methods = {"brute force": RecencyWrapper(BruteForce(), 0, 0, 10),
               "cue tree": RecencyWrapper(CueTree(B, a.beam, MF, True, True, False, 0.35, slack, ner), 0, 0, 10)}
    for w in methods.values():
        w.build(store)
    rows = []
    print(f"\nstore size {a.size} | cue-tree beam {a.beam} | {len(pairs)} pairs | selection rule fixed before running (see file header)\n")
    print("=== ANALYSIS 1 (original): top 5, rule applied to the top 10 candidates ===")
    for mname, w in methods.items():
        b = eval_top5(w, "top", 0.0, 0.0, std, pairs, qmap, k)
        print(f"\n{mname}: BASELINE stale {b['stale'].mean():.0%} | fresh@1 {b['fresh1'].mean():.0%} | newer found {b['found'].mean():.0%} | "
              f"neither found {b['neither'].mean():.0%} | standard Recall@5 {b['recall_std']:.3f}")
        run = lambda mode, dl, lm: (lambda r: {**r, "stale_rate": r["stale"].mean()})(eval_top5(w, mode, dl, lm, std, pairs, qmap, k))
        best = grid(mname, w, a.modes, a.deltas, a.lambdas, b["stale"].mean(), b["recall_std"], run, rows, "top5", {"pairs": len(pairs)})
        if best is None:
            print(f"  -> no eligible setting lowers the stale rate for {mname}: negative result")
        else:
            _, mode, dl, lm, r = best
            fixed, broken, p = sign_test(1 - r["stale"], 1 - b["stale"])
            print(f"  -> selected: variant '{mode}', delta {dl}, lambda {lm}: stale {b['stale'].mean():.0%} -> {r['stale'].mean():.0%}, "
                  f"fresh@1 {b['fresh1'].mean():.0%} -> {r['fresh1'].mean():.0%}; pairs fixed {fixed}, broken {broken}, sign test p = {p:.3f}")
    print(f"\n=== ANALYSIS 2 (post-hoc): pairwise inside the top {a.window}, pairs where BOTH sessions are retrieved ===")
    for mname, w in methods.items():
        b = eval_pairwise(w, "top", 0.0, 0.0, std, pairs, qmap, k, a.window)
        both = b["both"] == 1
        if both.sum() < 3:
            print(f"\n{mname}: only {int(both.sum())} pairs have both sessions in the top {a.window}: too few"); continue
        print(f"\n{mname}: both sessions in the top {a.window} for {int(both.sum())} of {len(pairs)} pairs | BASELINE older-above-newer "
              f"{b['stale'][both].mean():.0%} | standard Recall@5 {b['recall_std']:.3f}")
        run = lambda mode, dl, lm: (lambda r: {**r, "stale_rate": r["stale"][both].mean()})(eval_pairwise(w, mode, dl, lm, std, pairs, qmap, k, a.window))
        best = grid(mname, w, a.modes, a.deltas, a.lambdas, b["stale"][both].mean(), b["recall_std"], run, rows, "pairwise",
                    {"window": a.window, "pairs": int(both.sum())})
        if best is None:
            print(f"  -> no eligible setting lowers the pairwise stale rate for {mname}: negative result")
        else:
            _, mode, dl, lm, r = best
            fixed, broken, p = sign_test(1 - r["stale"][both], 1 - b["stale"][both])
            print(f"  -> selected: variant '{mode}', delta {dl}, lambda {lm}: older-above-newer {b['stale'][both].mean():.0%} -> "
                  f"{r['stale'][both].mean():.0%}, standard Recall@5 {b['recall_std']:.3f} -> {r['recall_std']:.3f}; "
                  f"pairs fixed {fixed}, broken {broken}, sign test p = {p:.3f} ({int(both.sum())} pairs)")
    if a.csv and rows:
        import csv
        keys = sorted({kk for r in rows for kk in r})
        with open(a.csv, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=keys); wr.writeheader(); wr.writerows(rows)
        print(f"\nsaved {len(rows)} rows to {a.csv}")

if __name__ == "__main__":
    main()
