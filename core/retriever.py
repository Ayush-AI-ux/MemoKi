from abc import ABC, abstractmethod
import time
import numpy as np
from .counters import Counters
from .types import Conversation, Result

class Retriever(ABC):
    """One interface for all methods: brute force, FAISS/HNSW, tree, cue-tree."""
    name = "base"

    @abstractmethod
    def build(self, conversations: list[Conversation]) -> None: ...

    @abstractmethod
    def _retrieve(self, query_text: str, query_emb: np.ndarray,
                  k: int, counters: Counters) -> list[Result]: ...

    def retrieve(self, query_text: str, query_emb: np.ndarray, k: int = 5):
        counters = Counters()
        t0 = time.perf_counter()
        results = self._retrieve(query_text, query_emb, k, counters)
        counters.latency_ms = (time.perf_counter() - t0) * 1000
        return results, counters
