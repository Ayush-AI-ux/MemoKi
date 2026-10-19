"""Mem0 feasibility spike on a small LoCoMo slice. Run in a SEPARATE venv with Ollama running:
   python -m spike.mem0_spike --dry-run                                   (no Mem0 needed: shows the plan)
   python -m spike.mem0_spike --sessions 1 --granularity session          (smallest real test)
   python -m spike.mem0_spike --sessions 3 --granularity session
   python -m spike.mem0_spike --sessions 3 --granularity pair             (closer to how Mem0 is usually fed)
Reports LLM calls, tokens, wall time per add and per search, the retrieved text size, and a LINEAR EXTRAPOLATION
to all of LoCoMo-10 (always labelled as extrapolated). Accuracy is NOT measured here."""
import argparse, json, shutil, time
from pathlib import Path
from spike.common import (load_slice, to_messages, approx_tokens, extrapolate, search_compat, result_texts, stored_count,
                          TOTAL_SESSIONS, TOTAL_TURNS)
from spike.meter import Meter

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--locomo", default="data/raw/locomo10.json")
    ap.add_argument("--conv", default="conv-26")
    ap.add_argument("--sessions", type=int, default=1)
    ap.add_argument("--granularity", choices=["session", "pair"], default="session")
    ap.add_argument("--model", default="llama3.2:3b")
    ap.add_argument("--limit", type=int, default=5, help="memories returned per search")
    ap.add_argument("--max-queries", type=int, default=10)
    ap.add_argument("--out", default="results/mem0_spike.json")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-add", action="store_true", help="reuse the memories already stored by a previous run (saves the ingestion time)")
    a = ap.parse_args()
    sl = load_slice(a.locomo, a.conv, a.sessions)
    turns = sum(len(s["turns"]) for s in sl["sessions"])
    batches = [(s, b) for s in sl["sessions"] for b in to_messages(s["turns"], sl["speaker_a"], a.granularity)]
    print(f"slice: {a.conv}, first {len(sl['sessions'])} sessions, {turns} turns, {len(batches)} Mem0 add() calls planned "
          f"({a.granularity} granularity), {len(sl['questions'])} questions answerable from the slice")
    if a.dry_run:
        print("dry run only: nothing was sent to Mem0 or Ollama"); return

    meter = Meter(); meter.install()
    from mem0 import Memory
    qdir = Path("data/cache/mem0_qdrant")
    if not a.skip_add:
        shutil.rmtree(qdir, ignore_errors=True)
    config = {"llm": {"provider": "ollama", "config": {"model": a.model, "temperature": 0.0, "max_tokens": 1024,
                                                       "ollama_base_url": "http://localhost:11434"}},
              "embedder": {"provider": "huggingface", "config": {"model": "sentence-transformers/all-MiniLM-L6-v2"}},
              "vector_store": {"provider": "qdrant", "config": {"collection_name": "mem0_spike", "path": str(qdir),
                                                                "embedding_model_dims": 384}}}
    m = Memory.from_config(config)
    uid = f"locomo_{a.conv}"
    adds, failures = [], 0
    t_all = time.perf_counter()
    for i, (s, msgs) in enumerate([] if a.skip_add else batches, 1):
        mk, t0 = meter.mark(), time.perf_counter()
        try:
            m.add(msgs, user_id=uid, metadata={"session": s["n"], "date": s["date"]})
        except Exception as ex:
            failures += 1; print(f"  add {i} FAILED: {type(ex).__name__}: {str(ex)[:160]}")
        st = meter.since(mk); st["wall"] = time.perf_counter() - t0; st["turns"] = len(msgs); adds.append(st)
        sent = [c for c in meter.calls[mk:]]
        st["chars_sent"] = sum(c["chars"] for c in sent); st["options"] = sent[0]["options"] if sent else {}
        print(f"  add {i}/{len(batches)} (session {s['n']}, {len(msgs)} turns): {st['calls']} LLM calls, "
              f"{st['prompt_tokens']} prompt + {st['output_tokens']} output tokens, {st['wall']:.0f} s")
        print(f"     sent to the model: {st['chars_sent']} characters (about {st['chars_sent']/4:.0f} tokens by chars/4) | options passed: {st['options']}")
    ingest_wall = time.perf_counter() - t_all
    tot = {k: sum(x[k] for x in adds) for k in ("calls", "prompt_tokens", "output_tokens", "wall")}

    def save(extra=None):
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump({"slice": {"conv": a.conv, "sessions": len(sl["sessions"]), "turns": turns, "adds": len(batches), "granularity": a.granularity},
                   "model": a.model, "failed_adds": failures, "adds": adds, "ingest_wall_s": ingest_wall, **(extra or {})},
                  open(a.out, "w"), indent=1)
    n_stored = stored_count(m, uid)
    if n_stored is not None:
        print(f"  memories stored for this user: {n_stored}")
    save({"memories_stored": n_stored})                      # ingestion numbers are safe on disk before any search is attempted
    searches = []
    for q in sl["questions"][:a.max_queries]:
        mk, t0 = meter.mark(), time.perf_counter()
        try:
            res = search_compat(m, q["question"], uid, a.limit)
        except Exception as ex:
            print(f"  search FAILED: {type(ex).__name__}: {str(ex)[:200]}"); break
        wall = time.perf_counter() - t0
        raw = result_texts(res); texts = raw[:a.limit]       # cut to the same k on our side whatever Mem0 returned
        searches.append({"llm_calls": meter.since(mk)["calls"], "wall": wall, "memories": len(texts), "raw_returned": len(raw),
                         "approx_tokens": sum(approx_tokens(t) for t in texts)})
    n = max(len(searches), 1)
    if adds:
        print(f"\nMEASURED ingestion on this slice ({turns} turns, {len(batches)} adds, {failures} failed adds):")
        print(f"  {tot['calls']} LLM calls, {tot['prompt_tokens']} prompt + {tot['output_tokens']} output tokens, {ingest_wall/60:.1f} min wall")
    if searches:
        print(f"MEASURED search ({len(searches)} queries): LLM calls per query {sum(s['llm_calls'] for s in searches)/n:.2f}, "
              f"{sum(s['wall'] for s in searches)/n*1000:.0f} ms per query, {sum(s['memories'] for s in searches)/n:.1f} memories (Mem0 itself returned {sum(s['raw_returned'] for s in searches)/n:.1f}, cut to {a.limit}) and "
              f"about {sum(s['approx_tokens'] for s in searches)/n:.0f} tokens retrieved per query (chars/4, approximate)")
    ext = None
    if adds:
        ext = extrapolate({"calls": tot["calls"], "seconds": tot["wall"], "prompt_tokens": tot["prompt_tokens"],
                           "output_tokens": tot["output_tokens"]}, turns, len(sl["sessions"]))
        print(f"\nEXTRAPOLATED linearly to all of LoCoMo-10 ({TOTAL_SESSIONS} sessions, {TOTAL_TURNS} turns); assumes the slice is typical:")
        print(f"  about {ext['calls']:.0f} LLM calls, {ext['prompt_tokens']/1e6:.2f} M prompt + {ext['output_tokens']/1e6:.2f} M output tokens, "
              f"{ext['seconds']/3600:.1f} hours of ingestion on this laptop")
    save({"searches": searches, "extrapolated": ext})
    print(f"saved {a.out}")

if __name__ == "__main__":
    main()
