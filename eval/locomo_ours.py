"""Our method on the SAME LoCoMo slice as the Mem0 spike: LLM calls, ingestion cost, retrieved tokens and a
hit rate (does a returned session contain the evidence?). Run in the main project venv:
   python -m eval.locomo_ours --sessions 3        (match the --sessions you used for the Mem0 spike)
No LLM is used anywhere. Units differ from Mem0 (we return WHOLE SESSIONS, Mem0 returns short facts), so retrieved
tokens are reported for k = 1, 2, 3 sessions and the difference is stated, not hidden."""
import argparse, time
import numpy as np
from core.cues import SpacyNER
from core.types import Conversation
from pipeline.config import load_all, path
from pipeline.embed import Embedder
from baselines.cue_tree import CueTree
from spike.common import load_slice, approx_tokens, extrapolate, TOTAL_SESSIONS, TOTAL_TURNS

def main(encode_fn=None, ner=None, locomo=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--conv", default="conv-26")
    ap.add_argument("--sessions", type=int, default=3)
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 2, 3])
    a = ap.parse_args()
    cfg = load_all(); d = cfg["data"]
    sl = load_slice(locomo or path(cfg, "locomo"), a.conv, a.sessions)
    turns = sum(len(s["turns"]) for s in sl["sessions"])
    enc = encode_fn or Embedder(cfg["embedding_model"], d["max_seq_length"]).encode
    ner = ner or SpacyNER()
    enc(["warm up"]); ner(["warm up"])
    convs = [Conversation(f"{a.conv}_s{s['n']}", "\n".join(f"{t['speaker']}: {t['text']}" for t in s["turns"]), 0.0)
             for s in sl["sessions"]]
    t0 = time.perf_counter(); vecs = enc([c.text for c in convs]); t_embed = time.perf_counter() - t0
    t0 = time.perf_counter(); ents = ner([c.text for c in convs]); t_ner = time.perf_counter() - t0
    for c, v, e in zip(convs, vecs, ents):
        c.embedding, c.cues["entities"] = v, e
    cue = CueTree(32, 10, 0.35, True, False, False, 0.35, 7.0, ner); cue.build(convs)
    qvec = enc([q["question"] for q in sl["questions"]])
    toks = {c.id: approx_tokens(c.text) for c in convs}
    print(f"slice: {a.conv}, first {len(convs)} sessions, {turns} turns, {len(sl['questions'])} questions answerable from the slice")
    ing = t_embed + t_ner
    print(f"\nMEASURED, no LLM anywhere:")
    print(f"  ingestion: 0 LLM calls, {ing:.2f} s total ({t_embed:.2f} s embedding + {t_ner:.2f} s name extraction), "
          f"{ing/len(convs)*1000:.0f} ms per session")
    for k in a.ks:
        hits, tk, ms, llm = [], [], [], 0
        for q, v in zip(sl["questions"], qvec):
            rel = {f"{a.conv}_s{n}" for n in q["evidence_sessions"]}
            res, c = cue.retrieve(q["question"], v, k)
            ids = [r.conv_id for r in res]
            hits.append(any(i in rel for i in ids)); tk.append(sum(toks[i] for i in ids)); ms.append(c.latency_ms + c.prep_ms); llm += c.llm_calls
        print(f"  search, top {k} session(s): LLM calls {llm}, hit rate {np.mean(hits):.0%} (evidence session returned), "
              f"about {np.mean(tk):.0f} tokens retrieved per query (chars/4), {np.mean(ms):.1f} ms per query including name extraction")
    ext = extrapolate({"seconds": ing}, turns, len(convs))
    print(f"\nEXTRAPOLATED linearly to all of LoCoMo-10 ({TOTAL_SESSIONS} sessions, {TOTAL_TURNS} turns): "
          f"0 LLM calls, about {ext['seconds']:.0f} s of ingestion")

if __name__ == "__main__":
    main()
