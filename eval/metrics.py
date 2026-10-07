def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int = 5) -> float:
    """1.0 if any relevant conversation is in the top-k, else 0.0 (per query)."""
    return float(any(r in relevant_ids for r in retrieved_ids[:k]))
