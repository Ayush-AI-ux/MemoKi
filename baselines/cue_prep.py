import time
from core.cues import QueryCues, SpacyNER
from core.timecue import parse_time_window

class CuePrep:
    """Query-cue preparation shared by every cue-using method, so they all see identical cues and
    pay the same extraction cost. Entities and dates are cached INDEPENDENTLY of the switches
    (toggling a switch later never reuses a stale value); the reported cost counts only enabled cues."""
    def _init_prep(self, use_names, use_time, slack_days, ner):
        self.use_names, self.use_time, self.slack_days = use_names, use_time, slack_days
        self.ner = ner if ner is not None else SpacyNER()
        self._ent_cache, self._win_cache = {}, {}

    def prepare(self, query_text, query_ts):
        if query_text not in self._ent_cache:
            t0 = time.perf_counter()
            ents = self.ner([query_text])[0]
            self._ent_cache[query_text] = (ents, (time.perf_counter() - t0) * 1000)
        key = (query_text, query_ts)
        if key not in self._win_cache:
            t0 = time.perf_counter()
            w = parse_time_window(query_text, query_ts) if query_ts else None
            self._win_cache[key] = (w, (time.perf_counter() - t0) * 1000)
        ents, ent_ms = self._ent_cache[query_text]
        window, win_ms = self._win_cache[key]
        qc = QueryCues(ents if self.use_names else frozenset(),
                       window if self.use_time else None, self.slack_days)
        return qc, (ent_ms if self.use_names else 0.0) + (win_ms if self.use_time else 0.0)
