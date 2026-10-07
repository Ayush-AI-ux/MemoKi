import numpy as np
from core.retriever import Retriever
from core.types import Result

class BruteForce(Retriever):
    name = "flat_brute_force"

    def build(self, conversations):
        self.ids = [c.id for c in conversations]
        m = np.stack([c.embedding for c in conversations]).astype("float32")
        self.matrix = m / np.linalg.norm(m, axis=1, keepdims=True)

    def _retrieve(self, query_text, query_emb, k, counters):
        q = query_emb / np.linalg.norm(query_emb)
        sims = self.matrix @ q
        counters.embedding_comparisons += len(self.ids)   # every item compared
        counters.nodes_visited += len(self.ids)
        top = np.argsort(-sims)[:k]
        return [Result(self.ids[i], float(sims[i])) for i in top]
