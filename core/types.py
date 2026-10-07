from dataclasses import dataclass, field
from typing import Optional
import numpy as np

@dataclass
class Conversation:
    id: str
    text: str
    timestamp: float                      # unix seconds
    embedding: Optional[np.ndarray] = None
    cues: dict = field(default_factory=dict)   # filled in Phase 2

@dataclass
class Result:
    conv_id: str
    score: float


@dataclass
class Query:
    id: str
    text: str
    relevant_ids: set            # ids of conversations that answer this query
    qtype: str = ""
    timestamp: float = 0.0       # when the query was asked (used by recency rule)
    embedding: Optional[np.ndarray] = None
