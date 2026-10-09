import time
from core.retriever import Retriever
from core.tree import MemoryTree
from core.types import Result
from .cue_prep import CuePrep

class CueTree(Retriever, CuePrep):
    """The proposed method: the same tree, plus entity and time summaries on every node.
    Switches (Gap 4 ablations): use_names, use_time, use_fallback. Zero LLM calls."""
    name = "cue_tree"

    def __init__(self, max_children=32, beam=3, min_fill=0.35, use_names=True, use_time=True,
                 use_fallback=False, tau=0.35, slack_days=7.0, ner=None):
        self.max_children, self.beam, self.min_fill = max_children, beam, min_fill
        self.use_fallback, self.tau = use_fallback, tau
        self._init_prep(use_names, use_time, slack_days, ner)
        self.tree = None

    def build(self, conversations):
        if any("entities" not in c.cues for c in conversations):
            raise ValueError("entity cues missing: run extract_entities_cached(...) before build()")
        self.tree = MemoryTree(self.max_children, self.min_fill)
        for c in conversations:                         # same insertion order as the no-cue tree
            self.tree.insert(c.id, c.embedding, c.cues["entities"], c.timestamp)

    def _retrieve(self, query_text, query_emb, k, counters, query_ts=None):
        t0 = time.perf_counter()
        qc, counters.prep_ms = self.prepare(query_text, query_ts)
        counters.prep_spent_ms = (time.perf_counter() - t0) * 1000
        hits = self.tree.search(query_emb, k, self.beam, counters, qc)
        if self.use_fallback and (not hits or hits[0][1] < self.tau):
            counters.fallback_triggered = True          # weak or empty result: pay for a full flat search
            hits = self.tree.flat_search(query_emb, k, counters)
        return [Result(i, s) for i, s in hits]
