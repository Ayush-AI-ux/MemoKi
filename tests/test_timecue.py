from datetime import datetime, timezone
from core.timecue import parse_time_window, window_overlaps, time_bucket, DAY

REF = datetime(2023, 6, 15, 12, 0, tzinfo=timezone.utc).timestamp()   # a Thursday
def ts(y, m, d, h=0, mi=0): return datetime(y, m, d, h, mi, tzinfo=timezone.utc).timestamp()
def win(text): return parse_time_window(text, REF)

def test_no_time_phrase():
    assert win("What degree did I graduate with?") is None
    assert win("I may have forgotten the Market research notes") is None   # verb 'may', 'Mar' inside 'Market'

def test_last_month_is_previous_calendar_month():
    w = win("Which pair of shoes did I clean last month?")
    assert w.kind == "last_period" and w.start == ts(2023, 5, 1) and w.end == ts(2023, 6, 1) - 1

def test_in_month_infers_most_recent_past_year():
    assert win("Which vehicle did I take care of first in February?").start == ts(2023, 2, 1)
    assert win("What did I do in November?").start == ts(2022, 11, 1)     # later than June -> previous year
    assert win("What happened in May?").start == ts(2023, 5, 1)           # 'May' with context is the month

def test_explicit_date_and_year():
    w = win("the party on February 5th")
    assert w.kind == "date" and w.start == ts(2023, 2, 5) and w.end == ts(2023, 2, 6) - 1
    assert win("a trip in March 2022").start == ts(2022, 3, 1)
    assert win("since 2021").kind == "year"

def test_n_ago_window_contains_expected_day():
    w = win("I bought it two weeks ago")
    assert w.kind == "n_ago" and w.start <= REF - 14 * DAY <= w.end
    assert win("3 days ago").start <= REF - 3 * DAY <= win("3 days ago").end

def test_weekday_and_yesterday():
    w = win("the mass last Sunday")                                       # Thursday 15 Jun -> Sunday 11 Jun
    assert w.kind == "weekday" and w.start == ts(2023, 6, 11)
    assert win("what did I eat yesterday").start == ts(2023, 6, 14)

def test_last_week_and_past_period():
    assert win("last week").start == ts(2023, 6, 5) and win("last week").end == ts(2023, 6, 12) - 1
    assert win("in the past month").kind == "past_period"
    assert win("this year").start == ts(2023, 1, 1)

def test_overlap_and_bucket():
    w = win("last month")
    assert window_overlaps(ts(2023, 5, 10), ts(2023, 5, 10), w)
    assert not window_overlaps(ts(2023, 6, 10), ts(2023, 6, 10), w)
    assert window_overlaps(ts(2023, 6, 2), ts(2023, 6, 2), w, slack_days=7)   # slack widens the window
    assert window_overlaps(ts(2023, 4, 1), ts(2023, 5, 3), w)                 # node ranges overlap
    assert time_bucket(ts(2023, 5, 20)) == "2023-05"

def test_every_n_ago_phrase_the_pattern_accepts_resolves():
    # regression: 'a couple of' was matched by the regex but missing from the lookup table
    for n in ["a couple of", "couple of", "a", "an", "one", "two", "three", "four", "five", "six",
              "seven", "eight", "nine", "ten", "eleven", "twelve", "5", "12"]:
        for unit in ["day", "week", "month", "year"]:
            for suffix in ["", "s"]:
                w = win(f"I did it {n} {unit}{suffix} ago")
                assert w is not None and w.kind == "n_ago", (n, unit, suffix)

def test_habits_and_comparisons_are_not_dates():
    assert win("during my Sunday group ride") is None                       # habit, not a date
    assert win("the Sunday mass at St. Mary's Church") is None
    assert win("compared to my previous year's time") is None               # comparison, not 'when'
    assert win("the mass on Sunday").kind == "weekday"                      # real references still work
    assert win("since Monday").kind == "weekday"
    assert win("what I did last Sunday").kind == "weekday"
    assert win("how did I do last year").kind == "last_period"
