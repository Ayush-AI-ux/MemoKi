import time
from core.cues import QueryCues, SpacyNER
from core.retriever import Retriever
from core.timecue import parse_time_window
from core.tree import MemoryTree
from core.types import Result

class CueTree(Retriever):
    """The proposed method: the same tree, plus entity and time summaries on every node.
    Switches (Gap 4 ablations): use_names, use_time, use_fallback. Zero LLM calls."""
    name = "cue_tree"

    def __init__(self, max_children=32, beam=3, min_fill=0.35, use_names=True, use_time=True,
                 use_fallback=False, tau=0.35, slack_days=7.0, ner=None):
        self.max_children, self.beam, self.min_fill = max_children, beam, min_fill
        self.use_names, self.use_time, self.use_fallback = use_names, use_time, use_fallback
        self.tau, self.slack_days = tau, slack_days
        self.ner = ner if ner is not None else SpacyNER()
        self.tree, self._ent_cache, self._win_cache = None, {}, {}

    def build(self, conversations):
        if any("entities" not in c.cues for c in conversations):
            raise ValueError("entity cues missing: run extract_entities_cached(...) before build()")
        self.tree = MemoryTree(self.max_children, self.min_fill)
        for c in conversations:                         # same insertion order as the no-cue tree
            self.tree.insert(c.id, c.embedding, c.cues["entities"], c.timestamp)

    def prepare(self, query_text, query_ts):
        """Query cues. Entities and dates are cached INDEPENDENTLY of the switches (so toggling a
        switch later never reuses a stale empty value). The reported prep cost counts only the
        cues that are switched on."""
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

    def _retrieve(self, query_text, query_emb, k, counters, query_ts=None):
        t0 = time.perf_counter()
        qc, counters.prep_ms = self.prepare(query_text, query_ts)
        counters.prep_spent_ms = (time.perf_counter() - t0) * 1000
        hits = self.tree.search(query_emb, k, self.beam, counters, qc)
        if self.use_fallback and (not hits or hits[0][1] < self.tau):
            counters.fallback_triggered = True          # weak or empty result: pay for a full flat search
            hits = self.tree.flat_search(query_emb, k, counters)
        return [Result(i, s) for i, s in hits]
