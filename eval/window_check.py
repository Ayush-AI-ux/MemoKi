"""Does embedding WINDOWS of a session beat embedding only its first 256 tokens?  LongMemEval, VALIDATION split,
brute-force search (so only the representation is compared; no tree, no cues, no LLM).
   python -m eval.window_check --sizes 1000            (window vectors are cached per session, so larger sizes reuse them)
Variants:  current = one vector from the first 256 tokens (the cached embeddings every earlier result used)
           mean    = the normalised mean of the vectors of all the session's windows (still ONE vector per session)
           best    = the session's score is its best window (about 15 times more vectors compared)
Windows are --words words long and do not overlap. 'comparisons' = vectors compared per query.
A paired exact sign test compares each variant's Recall@5 with the current one."""
import argparse, time
from math import comb
import numpy as np
from core.config import ROOT
from pipeline.config import load_all, path
from pipeline.loaders import load_longmemeval
from pipeline.store import load_splits, select_queries, build_store
from pipeline.embed import Embedder, embed_conversations, embed_queries
from eval.metrics import recall_at_k

def chunk_words(text, words):
    toks = text.split()
    return [" ".join(toks[i:i + words]) for i in range(0, max(len(toks), 1), words)]

def sign_test(a_hits, b_hits):
    a = int(((a_hits == 1) & (b_hits == 0)).sum()); b = int(((a_hits == 0) & (b_hits == 1)).sum()); n = a + b
    return a, b, (1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(min(a, b) + 1)) / 2 ** n))

def window_vectors(convs, encode, cache_dir, words, tag="m", chunk=250, log=print):
    """dict: session id -> (n_windows x d) array. Cached on disk, resumable, and only missing sessions are encoded."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    f = cache_dir / f"windows_{tag}_{words}.npz"
    store = {}
    if f.exists():
        z = np.load(f, allow_pickle=False); pos = 0
        for sid, n in zip(z["ids"].tolist(), z["counts"].tolist()):
            store[sid] = z["vec"][pos:pos + n]; pos += n
    todo = [c for c in convs if c.id not in store]
    log(f"[windows] {len(convs) - len(todo)} cached, {len(todo)} sessions to encode")
    t0 = time.perf_counter()
    for i in range(0, len(todo), chunk):
        part = todo[i:i + chunk]; wins, cnts = [], []
        for c in part:
            ws = chunk_words(c.text, words); wins += ws; cnts.append(len(ws))
        V = np.asarray(encode(wins), dtype="float32"); pos = 0
        for c, n in zip(part, cnts):
            store[c.id] = V[pos:pos + n]; pos += n
        ids = list(store); np.savez(f, ids=np.array(ids), counts=np.array([len(store[x]) for x in ids]),
                                    vec=np.concatenate([store[x] for x in ids]))
        el = time.perf_counter() - t0; done = min(i + chunk, len(todo))
        log(f"[windows] {done}/{len(todo)} sessions, {el/60:.1f} min elapsed, eta {el / done * (len(todo) - done) / 60:.1f} min")
    return store

def main(encode_fn=None, argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[1000])
    ap.add_argument("--words", type=int, default=150)
    ap.add_argument("--n-queries", type=int, default=100)
    ap.add_argument("--chunk", type=int, default=250)
    a = ap.parse_args(argv)
    cfg = load_all(); d = cfg["data"]
    pool, queries = load_longmemeval(path(cfg, "longmemeval"))
    splits = load_splits(path(cfg, "splits"))
    qs = select_queries(queries, splits["validation"], set(pool), a.n_queries, d["max_evidence_per_query"], d["query_seed"])
    enc = encode_fn or Embedder(cfg["embedding_model"], d["max_seq_length"]).encode
    embed_queries(qs, enc)
    cache, k, seed = ROOT / d["paths"]["cache_dir"], cfg["top_k"], cfg["seeds"][0]
    Q = np.stack([q.embedding / np.linalg.norm(q.embedding) for q in qs])
    print(f"VALIDATION | {len(qs)} queries | seed {seed} | windows of {a.words} words | brute force over each representation")
    out = {}
    for size in a.sizes:
        try:
            store = build_store(pool, qs, size, seed)
        except ValueError as e:
            print(f"size {size}: skipped ({e})"); continue
        embed_conversations(store, enc, cache, cfg["embedding_model"], d["max_seq_length"])
        wv = window_vectors(store, enc, cache, a.words, cfg["embedding_model"].replace("/", "_"), a.chunk)
        MA = np.stack([c.embedding / np.linalg.norm(c.embedding) for c in store])
        MB = np.stack([wv[c.id].mean(0) for c in store]); MB = MB / np.linalg.norm(MB, axis=1, keepdims=True)
        counts = np.array([len(wv[c.id]) for c in store]); offsets = np.concatenate([[0], np.cumsum(counts)[:-1]])
        WA = np.concatenate([wv[c.id] for c in store]); WA = WA / np.linalg.norm(WA, axis=1, keepdims=True)
        ids = [c.id for c in store]
        hit = {v: {1: [], 5: []} for v in ("current", "mean", "best")}
        for q, qv in zip(qs, Q):
            sc = {"current": MA @ qv, "mean": MB @ qv, "best": np.maximum.reduceat(WA @ qv, offsets)}
            for v, s in sc.items():
                top = [ids[i] for i in np.argsort(-s, kind="stable")[:5]]
                for kk in (1, 5):
                    hit[v][kk].append(recall_at_k(top, q.relevant_ids, kk))
        comps = {"current": len(store), "mean": len(store), "best": len(WA)}
        print(f"\nsize {size} | {len(WA)} windows in total, {counts.mean():.1f} per session (median {int(np.median(counts))})")
        print(f"  {'variant':<10}{'Recall@1':>9}{'Recall@5':>9}{'comparisons':>13}{'vs current (Recall@5)':>30}")
        base = np.array(hit["current"][5])
        for v in ("current", "mean", "best"):
            r5 = np.array(hit[v][5]); extra = ""
            if v != "current":
                ao, bo, p = sign_test(r5, base); extra = f"{ao} better, {bo} worse, p = {p:.3f}"
            print(f"  {v:<10}{np.mean(hit[v][1]):>9.3f}{r5.mean():>9.3f}{comps[v]:>13}{extra:>30}")
        out[size] = {"hit": hit, "comps": comps}
    return out

if __name__ == "__main__":
    main()
