"""Extract entity cues for the largest store (resumable).
   python -m pipeline.extract_cues --limit 200      # timing test
   python -m pipeline.extract_cues                  # full run (about 1 hour; safe to stop and resume)
   python -m pipeline.extract_cues --procs 3        # faster on a multi-core CPU"""
import argparse
import numpy as np
from core.config import ROOT
from core.cues import SpacyNER
from .config import load_all, path
from .loaders import load_longmemeval
from .store import load_splits, select_queries, build_store
from .cue_cache import extract_entities_cached

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--procs", type=int, default=1)
    ap.add_argument("--size", type=int, default=None)
    ap.add_argument("--all", action="store_true", help="extract for the WHOLE pool (about 19k sessions)")
    a = ap.parse_args()
    cfg = load_all(); d = cfg["data"]
    pool, queries = load_longmemeval(path(cfg, "longmemeval"))
    splits = load_splits(path(cfg, "splits"))
    qs = select_queries(queries, splits["test"], set(pool), d["n_queries"],
                        d["max_evidence_per_query"], d["query_seed"])
    size = a.size or max(cfg["store_sizes"])
    store = [pool[i] for i in sorted(pool)] if a.all else build_store(pool, qs, size, seed=cfg["seeds"][0])
    ner = SpacyNER(n_process=a.procs)
    extract_entities_cached(store, ner, ROOT / d["paths"]["cache_dir"], limit=a.limit)
    have = [len(c.cues["entities"]) for c in store if "entities" in c.cues]
    print(f"entities available for {len(have)}/{len(store)} sessions"
          + (f", mean {np.mean(have):.1f} per session" if have else ""))

if __name__ == "__main__":
    main()
