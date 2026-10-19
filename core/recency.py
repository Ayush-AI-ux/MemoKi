"""LLM-free recency rule (Gap 3). Applied to the candidate list of ANY retriever."""
from .types import Result

def apply_recency(results, timestamps, k, delta, lam, mode="top"):
    """results: Result list sorted by score, best first. Returns the top k after re-ranking.
    mode "top":   candidates within `delta` of the BEST score are 'close'; each gets lam * (its recency among them).
    mode "local": every candidate gets lam * (its recency among the candidates within `delta` of ITS OWN score),
                  so near-ties are re-ordered wherever they sit in the ranking (the synopsis wording: 'when
                  similarities are close'). Recency is 0 for the oldest and 1 for the newest of the group.
    Adds no vector comparisons."""
    if not results or lam <= 0 or delta <= 0:
        return list(results[:k])
    ts = [timestamps[r.conv_id] for r in results]
    sc = [r.score for r in results]
    out = []
    if mode == "local":
        for i, r in enumerate(results):
            grp = [ts[j] for j in range(len(results)) if abs(sc[j] - sc[i]) <= delta]
            lo, span = min(grp), max(grp) - min(grp)
            out.append(Result(r.conv_id, r.score + (lam * (ts[i] - lo) / span if span > 0 else 0.0)))
    else:
        best = sc[0]
        grp = [ts[j] for j in range(len(results)) if best - sc[j] <= delta]
        lo, span = min(grp), max(grp) - min(grp)
        for i, r in enumerate(results):
            bonus = lam * (ts[i] - lo) / span if (span > 0 and best - sc[i] <= delta) else 0.0
            out.append(Result(r.conv_id, r.score + bonus))
    out.sort(key=lambda r: (-r.score, r.conv_id))
    return out[:k]

def pair_outcome(top_ids, older_id, newer_id):
    """How one update pair came out in a top-k list.
    stale = the older conversation ranks above the newer one (or appears while the newer one is absent)."""
    io = top_ids.index(older_id) if older_id in top_ids else None
    inw = top_ids.index(newer_id) if newer_id in top_ids else None
    return {"fresh_first": bool(top_ids) and top_ids[0] == newer_id,
            "stale": io is not None and (inw is None or io < inw),
            "newer_found": inw is not None,
            "neither": io is None and inw is None}


def pairwise_outcome(ids, older_id, newer_id):
    """Rank-level comparison inside one candidate list (no top-5 cutoff). `both` = both conversations
    are in the list; `stale_pair` = both are present and the older one ranks above the newer one."""
    io = ids.index(older_id) if older_id in ids else None
    inw = ids.index(newer_id) if newer_id in ids else None
    both = io is not None and inw is not None
    return {"both": both, "stale_pair": both and io < inw, "fresh_first": bool(ids) and ids[0] == newer_id}
