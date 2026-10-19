from core.recency import apply_recency
from core.retriever import Retriever

class RecencyWrapper(Retriever):
    """Adds the recency rule on top of any retriever: ask it for `pool` candidates, re-rank the close
    ones by recency, return k. The wrapped retriever's cost counters are unchanged apart from k."""
    name = "recency_wrapper"

    def __init__(self, inner, delta=0.02, lam=0.01, pool=10, mode="top"):
        self.inner, self.delta, self.lam, self.pool, self.mode = inner, delta, lam, pool, mode
        self.timestamps = {}

    def build(self, conversations):
        self.inner.build(conversations)
        self.timestamps = {c.id: c.timestamp for c in conversations}

    def _retrieve(self, query_text, query_emb, k, counters, query_ts=None):
        cands = self.inner._retrieve(query_text, query_emb, max(self.pool, k), counters, query_ts)
        return apply_recency(cands, self.timestamps, k, self.delta, self.lam, self.mode)
