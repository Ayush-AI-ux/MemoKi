"""Convert raw datasets into Conversation (a session) and Query records."""
import json, re
from datetime import datetime, timezone
from core.types import Conversation, Query

_LME_DATE = re.compile(r"(\d{4})/(\d{2})/(\d{2}).*?(\d{2}):(\d{2})")

def parse_lme_date(s: str) -> float:
    """'2023/05/20 (Sat) 02:21' -> unix seconds (UTC)."""
    m = _LME_DATE.search(s)
    if not m:
        raise ValueError(f"unrecognised LongMemEval date: {s!r}")
    y, mo, d, h, mi = map(int, m.groups())
    return datetime(y, mo, d, h, mi, tzinfo=timezone.utc).timestamp()

def load_longmemeval(path):
    """Returns (pool: dict[id -> Conversation], queries: list[Query]).
    Every haystack session becomes one Conversation (kept whole, never chunked).
    Abstention questions (no evidence) are skipped."""
    data = json.load(open(path, encoding="utf-8"))
    pool, queries = {}, []
    for e in data:
        for sid, date, turns in zip(e["haystack_session_ids"], e["haystack_dates"], e["haystack_sessions"]):
            if sid not in pool:
                text = "\n".join(f"{t['role']}: {t['content']}" for t in turns)
                pool[sid] = Conversation(id=sid, text=text, timestamp=parse_lme_date(date))
        rel = set(e.get("answer_session_ids") or [])
        if not rel or e["question_id"].endswith("_abs"):
            continue
        queries.append(Query(id=e["question_id"], text=e["question"], relevant_ids=rel,
                             qtype=e.get("question_type", ""),
                             timestamp=parse_lme_date(e["question_date"]) if e.get("question_date") else 0.0))
    return pool, queries

def parse_locomo_date(s: str) -> float:
    """'1:56 pm on 8 May, 2023' -> unix seconds (UTC)."""
    return datetime.strptime(s.strip(), "%I:%M %p on %d %B, %Y").replace(tzinfo=timezone.utc).timestamp()

def load_locomo(path):
    """Returns (conversations: list[Conversation], queries: list[Query], by_sample: dict).
    Query evidence 'D3:7' (dialogue 7 of session 3) maps to that session's id."""
    data = json.load(open(path, encoding="utf-8"))
    convs, queries, by_sample = [], [], {}
    for sample in data:
        sid, c = sample["sample_id"], sample["conversation"]
        ids = []
        n = 1
        while f"session_{n}" in c:
            turns = c[f"session_{n}"]
            text = "\n".join(f"{t['speaker']}: {t['text']}" for t in turns)
            conv = Conversation(id=f"{sid}_s{n}", text=text,
                                timestamp=parse_locomo_date(c[f"session_{n}_date_time"]))
            convs.append(conv); ids.append(conv.id); n += 1
        by_sample[sid] = ids
        for i, qa in enumerate(sample["qa"]):
            rel = {f"{sid}_s{m}" for ev in qa.get("evidence", []) for m in re.findall(r"D(\d+):", ev)}
            rel &= set(ids)
            if rel:
                queries.append(Query(id=f"{sid}_q{i}", text=qa["question"], relevant_ids=rel,
                                     qtype=str(qa.get("category", ""))))
    return convs, queries, by_sample
