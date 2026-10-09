from core.retriever import Retriever
from core.tree import MemoryTree
from core.types import Result

class CentroidTree(Retriever):
    """Baseline #3: the tree WITHOUT cues. Isolates what the structure alone achieves."""
    name = "tree_no_cues"

    def __init__(self, max_children: int = 8, beam: int = 3, min_fill: float = 0.0):
        self.max_children, self.beam, self.min_fill, self.tree = max_children, beam, min_fill, None

    def build(self, conversations):
        self.tree = MemoryTree(self.max_children, self.min_fill)
        for c in conversations:                 # insertion order = store order (seeded, deterministic)
            self.tree.insert(c.id, c.embedding)

    def _retrieve(self, query_text, query_emb, k, counters, query_ts=None):
        return [Result(i, s) for i, s in self.tree.search(query_emb, k, self.beam, counters)]
