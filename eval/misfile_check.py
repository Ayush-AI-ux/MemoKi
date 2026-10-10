"""Gap 2 on the VALIDATION split: how much does misfiling hurt, and what recovers it at what cost?
Run: python -m eval.misfile_check                         (size 5000; fractions 5/10/20 %; 5 repeats)
     python -m eval.misfile_check --sizes 5000 10000 --repeats 5 --csv results/misfile.csv
Definitions (fixed before running):
  affected  = a query with at least one relevant conversation that was deliberately moved
  lost      = affected AND the clean tree found it AND the misfiled tree (no recovery) does not
  recovery  = share of lost queries that a recovery tier turns back into hits
  harmed    = queries the misfiled tree already got right that the tier turns into misses
  extra %   = (mean comparisons with the tier - mean comparisons without) / mean comparisons without
Recovery tiers (each fires only when the best similarity is below tau):
  wider x4 / x8 = rerun the SAME cue-guided search with a 4x / 8x wider beam
  entity lookup = flat entity index plus the date window, exact scoring of those candidates only
  full flat     = the synopsis fallback: brute force over everything
Pre-registered rule: at 10 % misfiling, take the (tier, tau) with the lowest mean comparisons whose
recovery is >= 80 %; if none reaches 80 %, take the one with the highest recovery. The synopsis
criterion is recovery > 80 % AND extra < 25 %; the PASS column applies it."""
import argparse
import numpy as np
from core.config import ROOT
from core.counters import Counters
from core.cues import SpacyNER
from core.misfile import pick_misfiled, plan_moves
from pipeline.config import load_all, path
from pipeline.loaders import load_longmemeval
from pipeline.store import load_splits, select_queries, build_store
from pipeline.embed import Embedder, embed_conversations, embed_queries
from pipeline.cue_cache import extract_entities_cached, cache_file, load_entity_cache
from baselines.brute_force import BruteForce
from baselines.cue_tree import CueTree
from eval.metrics import recall_at_k

TIERS = ("wider x4", "wider x8", "entity lookup", "full flat")

def safe_mean(a):
    return float(a.mean()) if len(a) else float('nan')

def hit_of(hits, rel, k):
    return float(recall_at_k([i for i, _ in hits], rel, k))

def tier_outcomes(cue, q, qc, hits0, c0, beam, k):
    """For one query: (hit, extra_comparisons) of every tier IF it fires."""
    out = {}
    hit0 = hit_of(hits0, q.relevant_ids, k)
    for name, mult in (("wider x4", 4), ("wider x8", 8)):
        c = Counters(); h = cue.tree.search(q.embedding, k, beam * mult, c, qc)
        out[name] = (hit_of(h, q.relevant_ids, k), c.embedding_comparisons)
    c = Counters(); h = cue.tree.entity_flat_search(q.embedding, k, c, qc)
    out["entity lookup"] = (hit_of(h, q.relevant_ids, k) if h else hit0, c.embedding_comparisons)
    c = Counters(); h = cue.tree.flat_search(q.embedding, k, c)
    out["full flat"] = (hit_of(h, q.relevant_ids, k), c.embedding_comparisons)
    return out

def main(encode_fn=None, ner=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[5000])
    ap.add_argument("--fractions", type=float, nargs="+", default=[0.05, 0.10, 0.20])
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--beam", type=int, default=10)
    ap.add_argument("--taus", type=float, nargs="+", default=[0.30, 0.40, 0.50])
    ap.add_argument("--n-queries", type=int, default=100)
    ap.add_argument("--csv", default=None)
    a = ap.parse_args()
    cfg = load_all(); d = cfg["data"]
    pool, queries = load_longmemeval(path(cfg, "longmemeval"))
    splits = load_splits(path(cfg, "splits"))
    qs = select_queries(queries, splits["validation"], set(pool), a.n_queries, d["max_evidence_per_query"], d["query_seed"])
    enc = encode_fn or Embedder(cfg["embedding_model"], d["max_seq_length"]).encode
    embed_queries(qs, enc)
    cache, k, base_seed = ROOT / d["paths"]["cache_dir"], cfg["top_k"], cfg["seeds"][0]
    B, MF = cfg["tree"]["max_children"], cfg["tree"].get("min_fill", 0.0)
    slack = cfg.get("cues", {}).get("time_slack_days", 7.0)
    own_ner = ner is None
    ner = ner or SpacyNER(); ner(["warm up"])
    taus = list(a.taus) + [float("inf")]                    # inf = 'always fire' = upper bound on recovery
    rows = []
    print(f"VALIDATION | {len(qs)} queries | beam {a.beam} | tree B={B} min_fill={MF} | repeats {a.repeats}")
    for size in a.sizes:
        try:
            store = build_store(pool, qs, size, base_seed)
        except ValueError as e:
            print(f"size {size}: skipped ({e})"); continue
        embed_conversations(store, enc, cache, cfg["embedding_model"], d["max_seq_length"])
        if own_ner:
            done = load_entity_cache(cache_file(cache, ner))
            if sum(c.id not in done for c in store) > 300:
                print("entity cues missing; run: python -m pipeline.extract_cues --all --procs 3"); return
        extract_entities_cached(store, ner, cache, log=lambda *_: None)
        mk = lambda: CueTree(B, a.beam, MF, True, True, False, 0.35, slack, ner)
        clean = mk(); clean.build(store)
        bf = BruteForce(); bf.build(store)
        evid = sorted({r for q in qs for r in q.relevant_ids})
        others = [c.id for c in store if c.id not in set(evid)]
        active = np.array([clean.prepare(q.text, q.timestamp)[0].active for q in qs])
        hit_clean, hit_bf = [], []
        for q in qs:
            qc, _ = clean.prepare(q.text, q.timestamp)
            hit_clean.append(hit_of(clean.tree.search(q.embedding, k, a.beam, Counters(), qc), q.relevant_ids, k))
            hit_bf.append(hit_of([(r.conv_id, r.score) for r in bf.retrieve(q.text, q.embedding, k)[0]], q.relevant_ids, k))
        hit_clean, hit_bf = np.array(hit_clean), np.array(hit_bf)
        print(f"\n=== size {size} | clean cue tree Recall@5 {hit_clean.mean():.3f} (brute force {hit_bf.mean():.3f}) | "
              f"{len(evid)} evidence leaves, {int(active.sum())} cue queries ===")
        for frac in a.fractions:
            ev = []                                         # one record per (repeat, query)
            for rep in range(a.repeats):
                seed = base_seed * 1000 + rep
                m = mk(); m.build(store)
                moved = set(pick_misfiled(evid, others, frac, seed))
                m.tree.move_leaves(plan_moves(m.tree, sorted(moved), seed))
                shape = m.tree.stats()
                for i, q in enumerate(qs):
                    qc, _ = m.prepare(q.text, q.timestamp)
                    c0 = Counters(); hits0 = m.tree.search(q.embedding, k, a.beam, c0, qc)
                    ev.append({"i": i, "hit0": hit_of(hits0, q.relevant_ids, k), "comps0": c0.embedding_comparisons,
                               "best": hits0[0][1] if hits0 else -1.0, "affected": bool(q.relevant_ids & moved),
                               "cue": bool(active[i]), "tiers": tier_outcomes(m, q, qc, hits0, c0, a.beam, k)})
            hit0 = np.array([e["hit0"] for e in ev]); clean_h = np.array([hit_clean[e["i"]] for e in ev])
            aff = np.array([e["affected"] for e in ev]); cuem = np.array([e["cue"] for e in ev])
            lost = aff & (clean_h == 1) & (hit0 == 0)
            base_comps = np.mean([e["comps0"] for e in ev])
            print(f"\nmisfiling {frac:.0%} | max children after moves: {shape['max_children']} (limit {B}) | "
                  f"query-events {len(ev)} (affected {int(aff.sum())})")
            print(f"  Recall@5 clean -> misfiled: all {clean_h.mean():.3f} -> {hit0.mean():.3f} | "
                  f"cue q {safe_mean(clean_h[cuem]):.3f} -> {safe_mean(hit0[cuem]):.3f} | other q {safe_mean(clean_h[~cuem]):.3f} -> {safe_mean(hit0[~cuem]):.3f}")
            print(f"  lost to misfiling: {int(lost.sum())}  (cue queries {int((lost & cuem).sum())}, queries without cues {int((lost & ~cuem).sum())})")
            if lost.sum() == 0:
                print("  nothing was lost, so there is nothing to recover at this fraction"); continue
            print(f"  {'tier':<14}{'tau':>5}{'fires':>7}{'recovered':>11}{'recovery':>9}{'harmed':>8}{'mean comps':>11}{'extra %':>9}  criterion")
            print(f"  {'no recovery':<14}{'-':>5}{'-':>7}{'-':>11}{'-':>9}{'-':>8}{base_comps:>11.0f}{0.0:>9.1f}")
            for tier in TIERS:
                for tau in taus:
                    fire = np.array([(e["best"] < tau) for e in ev])
                    th = np.array([e["tiers"][tier][0] for e in ev]); ex = np.array([e["tiers"][tier][1] for e in ev])
                    after = np.where(fire, th, hit0)
                    rec = int((lost & fire & (th == 1)).sum()); harm = int(((hit0 == 1) & fire & (th == 0)).sum())
                    comps = base_comps + float((fire * ex).mean()); extra = (comps - base_comps) / base_comps * 100
                    pct = rec / lost.sum()
                    ok = "PASS" if (pct > 0.8 and extra < 25) else ("recovers" if pct > 0.8 else "")
                    tl = "always" if tau == float("inf") else f"{tau:.2f}"
                    print(f"  {tier:<14}{tl:>5}{fire.mean():>7.0%}{rec:>7}/{int(lost.sum()):<3}{pct:>9.0%}{harm:>8}{comps:>11.0f}{extra:>9.1f}  {ok}")
                    rows.append({"size": size, "fraction": frac, "tier": tier, "tau": tl, "fires": fire.mean(), "lost": int(lost.sum()),
                                 "recovered": rec, "recovery": pct, "harmed": harm, "mean_comps": comps, "extra_pct": extra,
                                 "recall_clean": clean_h.mean(), "recall_misfiled": hit0.mean(), "recall_after": after.mean()})
    if a.csv and rows:
        import csv
        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        print(f"\nsaved {len(rows)} rows to {a.csv}")

if __name__ == "__main__":
    main()
