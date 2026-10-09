"""FAISS baselines: exact flat index and approximate HNSW. FAISS is imported lazily so the rest of the
project works without it. HNSW cost is the number of distance computations FAISS reports."""
import numpy as np
from core.retriever import Retriever
from core.types import Result

def _unit_matrix(convs):
    m = np.stack([c.embedding for c in convs]).astype("float32")
    return np.ascontiguousarray(m / np.linalg.norm(m, axis=1, keepdims=True))

class FaissFlat(Retriever):
    name = "faiss_flat"

    def build(self, conversations):
        import faiss
        self.ids = [c.id for c in conversations]
        self.index = faiss.IndexFlatIP(conversations[0].embedding.shape[0])
        self.index.add(_unit_matrix(conversations))

    def _retrieve(self, query_text, query_emb, k, counters, query_ts=None):
        q = (query_emb / np.linalg.norm(query_emb)).astype("float32")[None, :]
        D, I = self.index.search(q, k)
        counters.embedding_comparisons += len(self.ids)               # exact: every vector compared
        counters.nodes_visited += len(self.ids)
        return [Result(self.ids[i], float(s)) for s, i in zip(D[0], I[0]) if i >= 0]

class FaissHNSW(Retriever):
    name = "faiss_hnsw"

    def __init__(self, M=32, ef_construction=64, ef_search=64):
        self.M, self.ef_construction, self.ef_search = M, ef_construction, ef_search

    def build(self, conversations):
        import faiss
        faiss.omp_set_num_threads(1)                                  # single thread: deterministic graph
        self.ids = [c.id for c in conversations]
        self.index = faiss.IndexHNSWFlat(conversations[0].embedding.shape[0], self.M, faiss.METRIC_INNER_PRODUCT)
        self.index.hnsw.efConstruction = self.ef_construction
        self.index.add(_unit_matrix(conversations))                   # insertion order = store order

    def _retrieve(self, query_text, query_emb, k, counters, query_ts=None):
        import faiss
        self.index.hnsw.efSearch = self.ef_search
        faiss.cvar.hnsw_stats.reset()
        q = (query_emb / np.linalg.norm(query_emb)).astype("float32")[None, :]
        D, I = self.index.search(q, k)
        n = int(faiss.cvar.hnsw_stats.ndis)                           # distance computations FAISS performed
        counters.embedding_comparisons += n
        counters.nodes_visited += n
        return [Result(self.ids[i], float(s)) for s, i in zip(D[0], I[0]) if i >= 0]
