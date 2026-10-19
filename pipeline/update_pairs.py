"""Update pairs for Gap 3: (query, older conversation, newer conversation) from LongMemEval
knowledge-update questions. Assumes the dataset lists both the superseded and the updated session as
evidence; the counts returned show whether that holds."""
def build_update_pairs(queries, pool, allowed_ids=None):
    ku = [q for q in queries if q.qtype == "knowledge-update" and (allowed_ids is None or q.id in allowed_ids)]
    pairs, multi, same_day = [], 0, 0
    for q in sorted(ku, key=lambda q: q.id):
        rel = sorted((r for r in q.relevant_ids if r in pool), key=lambda r: (pool[r].timestamp, r))
        if len(rel) < 2:
            continue
        older, newer = rel[0], rel[-1]
        if pool[older].timestamp == pool[newer].timestamp:
            same_day += 1
            continue
        multi += len(rel) > 2
        pairs.append({"qid": q.id, "older": older, "newer": newer, "n_evidence": len(rel)})
    stats = {"knowledge_update_queries": len(ku), "pairs": len(pairs), "dropped_same_timestamp": same_day,
             "pairs_with_more_than_two_evidence": multi}
    return pairs, stats
