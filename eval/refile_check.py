"""Gap 2, repair by OFFLINE AUDIT: does re-filing misplaced conversations repair misfiling, at zero
query-time cost, without harming a healthy tree?  VALIDATION split.
Run: python -m eval.refile_check --sizes 5000 --csv results/refile_5000.csv
Same definitions as eval.misfile_check (lost = clean tree hit, misfiled tree miss, query has a moved leaf).
Per margin we report: audit moves (misfiled leaves caught / healthy leaves moved by mistake), lost queries
recovered, queries harmed, recall before and after, mean comparisons per query after (the audit does not
change the search procedure, but it changes the tree), and the effect of the audit on a CLEAN tree.
Pre-registered rule (fixed before running): take the smallest margin whose recovery at 10 % misfiling is
>= 80 % and whose audit on the CLEAN tree lowers Recall@5 by at most 1 point; if none qualifies, the margin
with the highest recovery. The audit costs (leaves x leaf-groups) comparisons ONCE, offline."""
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
from baselines.cue_tree import CueTree
from eval.misfile_check import hit_of, safe_mean

def evaluate(m, qs, k, beam):
    hits, comps = [], []
    for q in qs:
        qc, _ = m.prepare(q.text, q.timestamp)
        c = Counters(); h = m.tree.search(q.embedding, k, beam, c, qc)
        hits.append(hit_of(h, q.relevant_ids, k)); comps.append(c.embedding_comparisons)
    return np.array(hits), np.array(comps)

def main(encode_fn=None, ner=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[5000])
    ap.add_argument("--fractions", type=float, nargs="+", default=[0.05, 0.10, 0.20])
    ap.add_argument("--margins", type=float, nargs="+", default=[0.0, 0.02, 0.05, 0.10])
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--beam", type=int, default=10)
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
    rows = []
    print(f"VALIDATION | {len(qs)} queries | beam {a.beam} | audit rounds {a.rounds} | repeats {a.repeats}")
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
        hit_clean, comps_clean = evaluate(clean, qs, k, a.beam)
        active = np.array([clean.prepare(q.text, q.timestamp)[0].active for q in qs])
        evid = sorted({r for q in qs for r in q.relevant_ids}); evset = set(evid)
        others = [c.id for c in store if c.id not in evset]
        n_groups = len(clean.tree.leaf_parents())
        print(f"\n=== size {size} | clean cue tree Recall@5 {hit_clean.mean():.3f}, {comps_clean.mean():.0f} comps/query | "
              f"{len(store)} leaves in {n_groups} leaf-groups: one audit pass = {len(store) * n_groups:,} comparisons, offline ===")
        print("audit on the CLEAN tree (does it harm a healthy tree?)")
        print(f"  {'margin':>7}{'leaves moved':>14}{'Recall@5':>10}{'change':>8}{'comps/query':>13}")
        for mg in a.margins:
            m = mk(); m.build(store); st = m.tree.audit_refile(mg, a.rounds)
            h, c = evaluate(m, qs, k, a.beam)
            print(f"  {mg:>7.2f}{len(st['moved']):>14}{h.mean():>10.3f}{h.mean() - hit_clean.mean():>+8.3f}{c.mean():>13.0f}")
            rows.append({"size": size, "fraction": 0.0, "margin": mg, "moved": len(st["moved"]), "recall_after": h.mean(),
                         "recall_clean": hit_clean.mean(), "comps_after": c.mean()})
        for frac in a.fractions:
            agg = {mg: {"rec": 0, "harm": 0, "caught": 0, "false": 0, "n_mis": 0, "hit_after": [], "comps": []} for mg in a.margins}
            lost_total, hit0_all, cl_all, cue_all = 0, [], [], []
            for rep in range(a.repeats):
                seed = base_seed * 1000 + rep
                moved = set(pick_misfiled(evid, others, frac, seed))
                base = mk(); base.build(store); base.tree.move_leaves(plan_moves(base.tree, sorted(moved), seed))
                hit0, _ = evaluate(base, qs, k, a.beam)
                aff = np.array([bool(q.relevant_ids & moved) for q in qs])
                lost = aff & (hit_clean == 1) & (hit0 == 0)
                lost_total += int(lost.sum()); hit0_all.append(hit0)
                for mg in a.margins:
                    m = mk(); m.build(store); m.tree.move_leaves(plan_moves(m.tree, sorted(moved), seed))
                    st = m.tree.audit_refile(mg, a.rounds); mv = set(st["moved"])
                    h, c = evaluate(m, qs, k, a.beam)
                    g = agg[mg]; g["rec"] += int((lost & (h == 1)).sum()); g["harm"] += int(((hit0 == 1) & (h == 0)).sum())
                    g["caught"] += len(mv & moved); g["false"] += len(mv - moved); g["n_mis"] += len(moved)
                    g["hit_after"].append(h); g["comps"].append(c)
            hit0_m = np.mean(np.concatenate(hit0_all))
            print(f"\nmisfiling {frac:.0%} | lost to misfiling over {a.repeats} repeats: {lost_total} | Recall@5 clean {hit_clean.mean():.3f} -> misfiled {hit0_m:.3f}")
            if lost_total == 0:
                print("  nothing was lost at this fraction"); continue
            print(f"  {'margin':>7}{'caught':>14}{'moved by mistake':>18}{'recovered':>11}{'recovery':>9}{'harmed':>8}{'Recall@5 after':>16}{'comps/query':>13}")
            for mg in a.margins:
                g = agg[mg]; ha = np.mean(np.concatenate(g["hit_after"])); cc = np.mean(np.concatenate(g["comps"]))
                print(f"  {mg:>7.2f}{g['caught'] / max(g['n_mis'], 1):>14.0%}{g['false'] / a.repeats:>18.0f}"
                      f"{g['rec']:>7}/{lost_total:<3}{g['rec'] / lost_total:>9.0%}{g['harm']:>8}{ha:>16.3f}{cc:>13.0f}")
                rows.append({"size": size, "fraction": frac, "margin": mg, "lost": lost_total, "recovered": g["rec"],
                             "recovery": g["rec"] / lost_total, "harmed": g["harm"], "caught": g["caught"] / max(g["n_mis"], 1),
                             "moved_by_mistake_per_repeat": g["false"] / a.repeats, "recall_after": ha, "comps_after": cc,
                             "recall_clean": hit_clean.mean(), "recall_misfiled": hit0_m})
    if a.csv and rows:
        import csv
        keys = sorted({kk for r in rows for kk in r})
        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
        print(f"\nsaved {len(rows)} rows to {a.csv}")

if __name__ == "__main__":
    main()
