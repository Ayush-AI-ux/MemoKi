"""Time cue: rule-based, LLM-free. Conversations already carry a timestamp, so no
extraction is needed on that side. For a query we resolve phrases such as
'last month' or 'in February' to a date window relative to the date the query
was asked. Returns None when the query has no resolvable time phrase."""
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

DAY = 86400.0
_UTC = timezone.utc
_MONTHS = {m: i + 1 for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july",
     "august", "september", "october", "november", "december"])}
_ABBR = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7,
         "aug": 8, "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12}
_WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
_NUM = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
        "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "couple of": 2, "a couple of": 2}
_UNIT_DAYS = {"day": 1, "week": 7, "month": 30, "year": 365}
_MONTH_NAMES = "January|February|March|April|May|June|July|August|September|October|November|December"
_MONTH_ABBR = "Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sept|Sep|Oct|Nov|Dec"
_MONTH_DATE = re.compile(rf"\b({_MONTH_NAMES}|{_MONTH_ABBR})\.?(?![A-Za-z])"
                         r"(?:\s+(\d{1,2})(?:st|nd|rd|th)?(?!\d))?(?:,?\s+((?:19|20)\d{2}))?")
_MAY_CONTEXT = {"in", "on", "of", "since", "during", "by", "early", "late", "mid", "last", "this", "from", "until"}
_YEAR_ONLY = re.compile(r"\b(?:in|during|since|of|from)\s+((?:19|20)\d{2})\b")
_N_AGO = re.compile(r"\b(\d+|a couple of|couple of|an|a|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)"
                    r"\s+(day|week|month|year)s?\s+ago\b", re.I)
_PAST = re.compile(r"\b(?:(?:in|over|within|during|for)\s+the\s+(?:last|past)|the\s+past)\s+(week|month|year)\b", re.I)
_LAST = re.compile(r"\b(?:last|previous)\s+(week|month|year)\b(?![\'\u2019]s)", re.I)  # not "previous year's time"
_THIS = re.compile(r"\bthis\s+(week|month|year)\b", re.I)
# only a dated reference ("last Sunday", "on Sunday"), not a habit ("my Sunday group ride")
_WEEKDAY = re.compile(r"\b(?:last|previous|on|this\s+past|since)\s+(" + "|".join(_WEEKDAYS) + r")\b(?![\'\u2019]s)", re.I)
_YESTERDAY = re.compile(r"\byesterday\b", re.I)

@dataclass(frozen=True)
class TimeWindow:
    start: float          # unix seconds, inclusive
    end: float
    kind: str             # which rule fired (used for per-rule evaluation)

def time_bucket(ts: float) -> str:
    """Coarse bucket for a conversation timestamp, e.g. '2023-05'."""
    return datetime.fromtimestamp(ts, tz=_UTC).strftime("%Y-%m")

def window_overlaps(lo: float, hi: float, w: TimeWindow, slack_days: float = 0.0) -> bool:
    """One cue check: does the time range [lo, hi] (a leaf or a node) overlap the window?"""
    s = slack_days * DAY
    return lo <= w.end + s and hi >= w.start - s

def _dt(y, m=1, d=1): return datetime(y, m, d, tzinfo=_UTC)
def _day(dt): return dt.replace(hour=0, minute=0, second=0, microsecond=0)
def _month_end(y, m): return _dt(y + (m == 12), m % 12 + 1).timestamp() - 1
def _w(a, b, kind): return TimeWindow(float(a), float(b), kind)

def _month_rule(text, ref):
    for m in _MONTH_DATE.finditer(text):
        name = m.group(1)
        low = name.lower()
        month = _MONTHS.get(low) or _ABBR.get(low)
        if month is None:
            continue
        if low == "may" and name == "May" and not m.group(2):
            prev = text[:m.start()].split()
            if not prev or prev[-1].lower().strip(",.") not in _MAY_CONTEXT:
                continue                                  # the verb 'may', not the month
        year = int(m.group(3)) if m.group(3) else None
        day = int(m.group(2)) if m.group(2) else None
        if year is None:
            year = ref.year if month <= ref.month else ref.year - 1
        if day is not None:
            try:
                d0 = _dt(year, month, day)
                if m.group(3) is None and d0 > ref:
                    d0 = _dt(year - 1, month, day)
                return _w(d0.timestamp(), d0.timestamp() + DAY - 1, "date")
            except ValueError:
                pass
        return _w(_dt(year, month).timestamp(), _month_end(year, month), "month")
    return None

def parse_time_window(text: str, ref_ts: float) -> "TimeWindow | None":
    """ref_ts = when the query was asked. Rules run in a fixed priority order."""
    ref = datetime.fromtimestamp(ref_ts, tz=_UTC)
    r = _month_rule(text, ref)
    if r:
        return r
    m = _YEAR_ONLY.search(text)
    if m:
        y = int(m.group(1))
        return _w(_dt(y).timestamp(), _dt(y + 1).timestamp() - 1, "year")
    m = _N_AGO.search(text)
    if m:
        n = int(m.group(1)) if m.group(1).isdigit() else _NUM[m.group(1).lower()]
        u = _UNIT_DAYS[m.group(2).lower()] * DAY
        return _w(ref_ts - (n + 0.5) * u, min(ref_ts, ref_ts - (n - 0.5) * u), "n_ago")
    m = _PAST.search(text)
    if m:
        return _w(ref_ts - _UNIT_DAYS[m.group(1).lower()] * DAY, ref_ts, "past_period")
    m = _LAST.search(text)
    if m:
        unit = m.group(1).lower()
        if unit == "week":
            this_mon = _day(ref) - timedelta(days=ref.weekday())
            return _w((this_mon - timedelta(days=7)).timestamp(), this_mon.timestamp() - 1, "last_period")
        if unit == "month":
            y, mo = (ref.year, ref.month - 1) if ref.month > 1 else (ref.year - 1, 12)
            return _w(_dt(y, mo).timestamp(), _month_end(y, mo), "last_period")
        return _w(_dt(ref.year - 1).timestamp(), _dt(ref.year).timestamp() - 1, "last_period")
    m = _THIS.search(text)
    if m:
        unit = m.group(1).lower()
        start = {"week": _day(ref) - timedelta(days=ref.weekday()),
                 "month": _dt(ref.year, ref.month), "year": _dt(ref.year)}[unit]
        return _w(start.timestamp(), ref_ts, "this_period")
    m = _WEEKDAY.search(text)
    if m:
        back = (ref.weekday() - _WEEKDAYS.index(m.group(1).lower())) % 7 or 7
        d0 = _day(ref) - timedelta(days=back)
        return _w(d0.timestamp(), d0.timestamp() + DAY - 1, "weekday")
    if _YESTERDAY.search(text):
        d0 = _day(ref) - timedelta(days=1)
        return _w(d0.timestamp(), d0.timestamp() + DAY - 1, "yesterday")
    return None
