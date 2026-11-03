"""Does LoCoMo retrieval quality depend on HOW sessions are embedded? No LLM anywhere. Each of the 10
conversations is its own memory (like Mem0's per-user store); the questions are those with evidence.
   python -m eval.locomo_variants
Variants: whole session at 256 tokens (current) | whole session at 512 | windows of --window turns scored by
their best window (returning whole sessions) | the same windows returned ALONE (matched passages only).
'hit' = a returned session (or window) belongs to an evidence session. Chance = a random ranking of the sessions."""
import argparse, json, re
from math import comb
import numpy as np
from pipeline.config import load_all, path
from pipeline.embed import Embedder
from spike.common import approx_tokens

KS = (1, 2, 3, 5)

def chance_hit(n, e, k):
    """P(at least one of e evidence sessions is among k random picks out of n sessions)."""
    k = min(k, n)
    return 1.0 - (comb(n - e, k) / comb(n, k) if n - e >= k else 0.0)

def conv_data(entry):
    c = entry["conversation"]
    sessions, n = [], 1
    while f"session_{n}" in c:
        sessions.append([f"{t['speaker']}: {t['text']}" for t in c[f"session_{n}"]]); n += 1
    qs = []
    for qa in entry["qa"]:
        ev = {int(m) - 1 for x in qa.get("evidence", []) for m in re.findall(r"D(\d+):", x)}
        ev = {e for e in ev if 0 <= e < len(sessions)}
        if ev:
            qs.append({"q": qa["question"], "ev": ev})
    return sessions, qs

def main(encode256=None, encode512=None, locomo=None, argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=4, help="turns per window")
    a = ap.parse_args(argv)
    cfg = load_all(); model = cfg["embedding_model"]
    e256 = encode256 or Embedder(model, 256).encode
    e512 = encode512 or Embedder(model, 512).encode
    data = json.load(open(locomo or path(cfg, "locomo"), encoding="utf-8"))
    names = ["whole session, 256 tokens (current)", "whole session, 512 tokens",
             f"{a.window}-turn windows, best window per session", f"{a.window}-turn windows, matched windows only"]
    hit = {n: {k: [] for k in KS} for n in names}; tok = {n: {k: [] for k in KS} for n in names}
    chance = {k: [] for k in KS}; n_q = 0
    for entry in data:
        sessions, qs = conv_data(entry)
        if not qs:
            continue
        texts = ["\n".join(s) for s in sessions]
        stok = [approx_tokens(t) for t in texts]
        S256, S512 = e256(texts), e512(texts)
        wins, owner = [], []
        for si, s in enumerate(sessions):
            for i in range(0, len(s), a.window):
                wins.append("\n".join(s[i:i + a.window])); owner.append(si)
        W = e256(wins); owner = np.array(owner); wtok = np.array([approx_tokens(w) for w in wins])
        Q = e256([x["q"] for x in qs]); Q512 = e512([x["q"] for x in qs])
        for x, q, q5 in zip(qs, Q, Q512):
            n_q += 1
            rank_a = np.argsort(-(S256 @ q), kind="stable"); rank_b = np.argsort(-(S512 @ q5), kind="stable")
            wsim = W @ q
            best = np.array([wsim[owner == si].max() for si in range(len(sessions))])
            rank_c = np.argsort(-best, kind="stable"); rank_w = np.argsort(-wsim, kind="stable")
            for k in KS:
                for nm, rk in ((names[0], rank_a), (names[1], rank_b), (names[2], rank_c)):
                    top = rk[:k]
                    hit[nm][k].append(any(t in x["ev"] for t in top)); tok[nm][k].append(sum(stok[t] for t in top))
                tw = rank_w[:k]
                hit[names[3]][k].append(any(owner[w] in x["ev"] for w in tw)); tok[names[3]][k].append(float(wtok[tw].sum()))
                chance[k].append(chance_hit(len(sessions), len(x["ev"]), k))
    print(f"{n_q} questions with evidence across {len(data)} conversations (each conversation is its own memory)\n")
    print(f"{'variant':<46}" + "".join(f"{'hit@' + str(k):>8}" for k in KS) + f"{'tokens@5':>11}")
    for nm in names:
        print(f"{nm:<46}" + "".join(f"{np.mean(hit[nm][k]):>8.0%}" for k in KS) + f"{np.mean(tok[nm][5]):>11.0f}")
    print(f"{'chance (random ranking of the sessions)':<46}" + "".join(f"{np.mean(chance[k]):>8.0%}" for k in KS))
    return {"names": names, "hit": hit, "tok": tok, "chance": chance, "n_q": n_q}

if __name__ == "__main__":
    main()
