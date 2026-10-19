"""Pure-Python helpers shared by the Mem0 spike (runs in its own venv) and our own LoCoMo measurement."""
import json, re

TOTAL_SESSIONS, TOTAL_TURNS, TOTAL_QUESTIONS = 272, 5882, 1986     # LoCoMo-10 totals (measured from the file)

def approx_tokens(text):
    """Rough token count: characters / 4. Used only where an exact count is unavailable; labelled approximate."""
    return len(text) / 4.0

def load_slice(path, conv_id, n_sessions):
    """First `n_sessions` sessions of one LoCoMo conversation, plus the questions whose evidence lies fully inside."""
    data = json.load(open(path, encoding="utf-8"))
    e = next(x for x in data if x["sample_id"] == conv_id)
    c = e["conversation"]
    sessions = []
    for n in range(1, n_sessions + 1):
        if f"session_{n}" not in c:
            break
        sessions.append({"n": n, "date": c[f"session_{n}_date_time"], "turns": c[f"session_{n}"]})
    inside = {s["n"] for s in sessions}
    qs = []
    for i, qa in enumerate(e["qa"]):
        ev = {int(m) for x in qa.get("evidence", []) for m in re.findall(r"D(\d+):", x)}
        if ev and ev <= inside:
            qs.append({"id": f"{conv_id}_q{i}", "question": qa["question"], "answer": qa.get("answer"),
                       "evidence_sessions": sorted(ev), "category": qa.get("category")})
    return {"speaker_a": c["speaker_a"], "speaker_b": c["speaker_b"], "sessions": sessions, "questions": qs}

def to_messages(turns, speaker_a, granularity):
    """Turns -> Mem0-style message lists. Speaker A is 'user', B is 'assistant'; the speaker name is kept in the text."""
    msgs = [{"role": "user" if t["speaker"] == speaker_a else "assistant", "content": f"{t['speaker']}: {t['text']}"} for t in turns]
    if granularity == "session":
        return [msgs]
    return [msgs[i:i + 2] for i in range(0, len(msgs), 2)]

def extrapolate(per_turn, measured_turns, measured_sessions):
    """Linear extrapolation of totals measured on a slice to all of LoCoMo-10. per_turn: dict of totals measured."""
    out = {}
    for k, v in per_turn.items():
        out[k] = v / max(measured_turns, 1) * TOTAL_TURNS
    return out


def search_compat(memory, query, user_id, limit):
    """Mem0's search() signature changed across versions: 2.x wants filters={'user_id': ...}; older versions
    want user_id=... directly. Try the new form first, fall back to the old one."""
    last = None
    for kwargs in ({"filters": {"user_id": user_id}, "top_k": limit, "limit": limit},
                   {"filters": {"user_id": user_id}, "limit": limit}, {"user_id": user_id, "limit": limit}):
        try:
            return memory.search(query, **kwargs)
        except (TypeError, ValueError) as ex:
            last = ex
    raise last

def result_texts(res):
    """Memory strings from either result shape: {'results': [...]} or a plain list."""
    items = res.get("results", []) if isinstance(res, dict) else res
    return [(i.get("memory") or i.get("text") or "") if isinstance(i, dict) else str(i) for i in items]


def stored_count(memory, user_id):
    """How many memories Mem0 holds for the user (None if the call is not available in this version)."""
    for kwargs in ({"filters": {"user_id": user_id}}, {"user_id": user_id}):
        try:
            res = memory.get_all(**kwargs)
            return len(res.get("results", []) if isinstance(res, dict) else res)
        except Exception:
            continue
    return None
