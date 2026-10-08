"""Does the time cue find the right conversation, and how much does it prune?
Uses the VALIDATION split only. Run after the entity extraction has finished
(it loads the full dataset). Run: python -m eval.time_cue_check"""
import collections
import numpy as np
from pipeline.config import load_all, path
from pipeline.loaders import load_longmemeval
from pipeline.store import load_splits
from core.timecue import parse_time_window, DAY

SLACKS = (0, 7, 30)
cfg = load_all()
pool, queries = load_longmemeval(path(cfg, "longmemeval"))
val = set(load_splits(path(cfg, "splits"))["validation"])
vq = [q for q in queries if q.id in val]
all_ts = np.sort(np.array([c.timestamp for c in pool.values()]))
span = (all_ts[-1] - all_ts[0]) / DAY
print(f"pool timestamps: {len(all_ts)} sessions spanning {span:.0f} days; {len(vq)} validation queries")

stats = collections.defaultdict(lambda: {"n": 0, "hit": [0] * len(SLACKS), "sel": [[] for _ in SLACKS]})
samples = {}
for q in vq:
    w = parse_time_window(q.text, q.timestamp)
    if w is None:
        continue
    s = stats[w.kind]; s["n"] += 1
    ev = [pool[r].timestamp for r in q.relevant_ids if r in pool]
    for j, k in enumerate(SLACKS):
        lo, hi = w.start - k * DAY, w.end + k * DAY
        s["hit"][j] += any(lo <= t <= hi for t in ev)
        s["sel"][j].append(np.searchsorted(all_ts, hi, "right") - np.searchsorted(all_ts, lo, "left"))
    samples.setdefault(w.kind, []).append(q.text[:80])

tot = sum(s["n"] for s in stats.values())
print(f"\nqueries with a resolvable time phrase: {tot}/{len(vq)} ({tot/len(vq):.0%})")
head = "".join(f"  rec@{k}d  sel@{k}d" for k in SLACKS)
print(f"{'rule':14}{'n':>4}{head}")
def row(name, n, hits, sels):
    cells = "".join(f"  {h/n:>6.0%}  {np.mean(s)/len(all_ts):>6.1%}" for h, s in zip(hits, sels))
    print(f"{name:14}{n:>4}{cells}")
for kind, s in sorted(stats.items(), key=lambda kv: -kv[1]["n"]):
    row(kind, s["n"], s["hit"], s["sel"])
if tot:
    row("ALL", tot, [sum(s["hit"][j] for s in stats.values()) for j in range(len(SLACKS))],
        [sum((s["sel"][j] for s in stats.values()), []) for j in range(len(SLACKS))])
print("\nrec = evidence session falls inside the window (recall); sel = share of the pool inside the window (lower prunes more)")
print("\nsamples:")
for kind, lst in samples.items():
    print(f"  [{kind}]", lst[0])
