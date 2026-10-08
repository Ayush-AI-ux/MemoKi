"""Resumable on-disk cache for entity cues. Extraction is slow (about 1 hour for
10,000 sessions), so results are appended in chunks and an interrupted run simply
continues where it stopped."""
import hashlib, json, time

def cache_file(cache_dir, ner):
    tag = hashlib.md5(ner.tag.encode()).hexdigest()[:8]
    return cache_dir / f"entities_{tag}.jsonl"

def load_entity_cache(f) -> dict:
    done = {}
    if f.exists():
        for line in open(f, encoding="utf-8"):
            try:
                r = json.loads(line)
                done[r["id"]] = frozenset(r["e"])
            except (json.JSONDecodeError, KeyError):
                continue            # half-written last line after an interruption
    return done

def _ensure_trailing_newline(f):
    """A crash can leave a half-written last line; without this the next appended
    record would be glued onto it and silently lost."""
    if f.exists() and f.stat().st_size > 0:
        with open(f, "rb") as fh:
            fh.seek(-1, 2)
            last = fh.read(1)
        if last != b"\n":
            with open(f, "ab") as fh:
                fh.write(b"\n")

def extract_entities_cached(convs, ner, cache_dir, chunk=200, limit=None, log=print):
    """Fills c.cues['entities'] (frozenset) for every conversation, extracting only misses."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    f = cache_file(cache_dir, ner)
    _ensure_trailing_newline(f)
    done = load_entity_cache(f)
    todo = [c for c in convs if c.id not in done]
    if limit is not None:
        todo = todo[:limit]
    log(f"[cues] {len(convs) - len([c for c in convs if c.id not in done])} cached, {len(todo)} to extract")
    t0 = time.perf_counter()
    for i in range(0, len(todo), chunk):
        part = todo[i:i + chunk]
        res = ner([c.text for c in part])
        with open(f, "a", encoding="utf-8") as fh:
            for c, e in zip(part, res):
                fh.write(json.dumps({"id": c.id, "e": sorted(e)}) + "\n")
                done[c.id] = frozenset(e)
        n = i + len(part); el = time.perf_counter() - t0
        log(f"[cues] {n}/{len(todo)}  {n/el:.1f} sessions/s  eta {(len(todo)-n)/(n/el)/60:.1f} min")
    for c in convs:
        if c.id in done:
            c.cues["entities"] = done[c.id]
    return convs
