"""Cue extraction. Cues are plain data, kept separate from the tree so that
ablations (Gap 4) can switch each cue on/off without touching tree code."""
import re
from dataclasses import dataclass
from typing import Optional
from .timecue import TimeWindow

NORM_VERSION = "v1"   # bump if normalisation changes: invalidates cached entities
NAMED_LABELS = frozenset({"PERSON", "ORG", "GPE", "LOC", "FAC", "PRODUCT",
                          "EVENT", "WORK_OF_ART", "NORP"})
_QUOTES = re.compile(r"[\"'\u201c\u201d\u2018\u2019]")

def normalize_entity(s: str) -> str:
    """lowercase, strip quotes and a leading article, collapse spaces.
    Identical to the version validated in eval/cue_selectivity.py
    (95% cue recall, 1.2% mean selectivity)."""
    s = _QUOTES.sub("", s.lower().strip())
    s = re.sub(r"^(the|a|an)\s+", "", s)
    return re.sub(r"\s+", " ", s).strip()

class SpacyNER:
    """Callable: list[str] -> list[frozenset[str]] of normalised named entities.
    Any object with the same call signature and a `tag` attribute can replace it
    (tests use a fake, so they need neither spaCy nor its model)."""
    def __init__(self, model: str = "en_core_web_sm", max_chars: int = 20000,
                 n_process: int = 1, batch_size: int = 16):
        self.model, self.max_chars = model, max_chars
        self.n_process, self.batch_size = n_process, batch_size
        self._nlp = None

    @property
    def tag(self) -> str:                    # identifies the cache this extractor writes
        return f"{self.model}|{NORM_VERSION}|{self.max_chars}"

    def _load(self):
        if self._nlp is None:
            import spacy
            self._nlp = spacy.load(self.model, disable=["parser", "lemmatizer", "tagger", "attribute_ruler"])
            self._nlp.max_length = 200000
        return self._nlp

    def __call__(self, texts):
        nlp = self._load()
        docs = nlp.pipe([t[: self.max_chars] for t in texts],
                        batch_size=self.batch_size, n_process=self.n_process)
        out = []
        for d in docs:
            ents = {normalize_entity(e.text) for e in d.ents if e.label_ in NAMED_LABELS}
            out.append(frozenset(ents - {""}))
        return out


@dataclass(frozen=True)
class QueryCues:
    """Cues found in ONE query. entities: normalised names. window: resolved date window or None."""
    entities: frozenset = frozenset()
    window: Optional[TimeWindow] = None
    slack_days: float = 7.0

    @property
    def active(self) -> bool:
        return bool(self.entities) or self.window is not None
