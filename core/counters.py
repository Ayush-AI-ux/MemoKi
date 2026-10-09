from dataclasses import dataclass, asdict

@dataclass
class Counters:
    """Every retriever MUST report cost through this object.
    Cue checks (cheap) and embedding comparisons (expensive) are kept separate."""
    cue_checks: int = 0
    embedding_comparisons: int = 0
    nodes_visited: int = 0
    llm_calls: int = 0          # must stay 0 for the proposed method
    fallback_triggered: bool = False
    levels_relaxed: int = 0     # levels where no child matched the cues, so cue pruning was skipped
    prep_ms: float = 0.0        # cost of query preparation (entity extraction, date parsing), reported separately
    prep_spent_ms: float = 0.0  # time actually spent preparing in THIS call (0 on a cache hit); subtracted from latency
    latency_ms: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)
