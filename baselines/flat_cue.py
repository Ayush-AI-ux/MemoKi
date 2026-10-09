import time
import numpy as np
from core.retriever import Retriever
from core.types import Result
from core.timecue import DAY
from .cue_prep import CuePrep

class FlatCueFilter(Retriever, CuePrep):
    """Fair opponent for the cue tree: a FLAT store with the SAME cue filter, via an inverted index
    (entity -> leaves) and a timestamp array. No tree. Same cues, same semantics (any query entity
    matches; date window with slack; AND when both are present; if nothing matches, search everything)."""
    name = "flat_cue_filter"

    def __init__(self, use_names=True, use_time=True, slack_days=7.0, ner=None):
        self._init_prep(use_names, use_time, slack_days, ner)

    def build(self, conversations):
        if any("entities" not in c.cues for c in conversations):
            raise ValueError("entity cues missing: run extract_entities_cached(...) before build()")
        self.ids = [c.id for c in conversations]
        m = np.stack([c.embedding for c in conversations]).astype("float32")
        self.matrix = m / np.linalg.norm(m, axis=1, keepdims=True)
        self.ts = np.array([c.timestamp for c in conversations], dtype=np.float64)
        inv = {}
        for i, c in enumerate(conversations):
            for e in c.cues["entities"]:
                inv.setdefault(e, []).append(i)
        self.index = {e: np.array(v, dtype=np.int64) for e, v in inv.items()}

    def _candidates(self, qc, counters):
        """Leaf indices passing the cue filter, or None when there is nothing to filter on
        (no cues, or the cues matched nothing, in which case 'search everything' applies)."""
        n = len(self.ids)
        cand = None
        if qc.entities:
            counters.cue_checks += len(qc.entities)                  # one index lookup per query entity
            parts = [self.index[e] for e in qc.entities if e in self.index]
            cand = np.unique(np.concatenate(parts)) if parts else np.empty(0, dtype=np.int64)
        if qc.window is not None:
            pool = np.arange(n) if cand is None else cand
            counters.cue_checks += len(pool)                         # one date test per candidate scanned
            s = qc.slack_days * DAY
            keep = (self.ts[pool] >= qc.window.start - s) & (self.ts[pool] <= qc.window.end + s)
            cand = pool[keep]
        if cand is None or len(cand) == 0:
            if qc.active:
                counters.levels_relaxed += 1
            return None
        return cand

    def _search_all(self, query_emb, k, counters):
        """No usable cues: exact search over everything (the plain flat behaviour)."""
        q = query_emb / np.linalg.norm(query_emb)
        sims = self.matrix @ q                                       # no copy of the matrix
        counters.embedding_comparisons += len(self.ids)
        counters.nodes_visited += len(self.ids)
        top = np.argsort(-sims, kind="stable")[:k]
        return [Result(self.ids[i], float(sims[i])) for i in top]

    def _retrieve(self, query_text, query_emb, k, counters, query_ts=None):
        t0 = time.perf_counter()
        qc, counters.prep_ms = self.prepare(query_text, query_ts)
        counters.prep_spent_ms = (time.perf_counter() - t0) * 1000
        cand = self._candidates(qc, counters)
        if cand is None:
            return self._search_all(query_emb, k, counters)
        q = query_emb / np.linalg.norm(query_emb)
        sims = self.matrix[cand] @ q
        counters.embedding_comparisons += len(cand)
        counters.nodes_visited += len(cand)
        top = np.argsort(-sims, kind="stable")[:k]
        return [Result(self.ids[cand[i]], float(sims[i])) for i in top]

class HybridCueHNSW(FlatCueFilter):
    """The strongest flat opponent: cue filter + exact scoring when the query has usable cues,
    FAISS HNSW when it has none."""
    name = "filter_plus_hnsw"

    def __init__(self, use_names=True, use_time=True, slack_days=7.0, ner=None, M=32, ef_construction=64, ef_search=64):
        super().__init__(use_names, use_time, slack_days, ner)
        self.M, self.ef_construction, self.ef_search = M, ef_construction, ef_search

    def build(self, conversations):
        from .faiss_index import FaissHNSW
        super().build(conversations)
        self.hnsw = FaissHNSW(self.M, self.ef_construction, self.ef_search)
        self.hnsw.build(conversations)

    def _search_all(self, query_emb, k, counters):
        self.hnsw.ef_search = self.ef_search
        return self.hnsw._retrieve("", query_emb, k, counters)
