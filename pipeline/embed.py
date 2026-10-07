"""Embeddings with an on-disk cache so the (slow) encoding happens once."""
import hashlib
import numpy as np

class Embedder:
    def __init__(self, model_name: str, max_seq_length: int = 256, batch_size: int = 32):
        self.model_name, self.max_seq_length, self.batch_size = model_name, max_seq_length, batch_size
        self._model = None

    def encode(self, texts: list[str]) -> np.ndarray:
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
            self._model.max_seq_length = self.max_seq_length
        v = self._model.encode(texts, batch_size=self.batch_size, show_progress_bar=len(texts) > 64,
                               convert_to_numpy=True, normalize_embeddings=True)
        return v.astype("float32")

def _cache_file(cache_dir, model_name, max_len):
    tag = hashlib.md5(f"{model_name}|{max_len}".encode()).hexdigest()[:8]
    return cache_dir / f"emb_{tag}.npz"

def embed_conversations(convs, encode_fn, cache_dir, model_name="m", max_len=256):
    """Fills c.embedding for every conversation, encoding only cache misses."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    f = _cache_file(cache_dir, model_name, max_len)
    store = {}
    if f.exists():
        z = np.load(f, allow_pickle=False)
        store = dict(zip(z["ids"].tolist(), z["mat"]))
    missing = [c for c in convs if c.id not in store]
    if missing:
        print(f"[embed] encoding {len(missing)} new conversations ({len(convs)-len(missing)} cached)")
        mat = encode_fn([c.text for c in missing])
        for c, v in zip(missing, mat):
            store[c.id] = v
        ids = np.array(list(store.keys()))
        np.savez(f, ids=ids, mat=np.stack([store[i] for i in ids]))
    for c in convs:
        c.embedding = store[c.id]
    return convs

def embed_queries(queries, encode_fn):
    mat = encode_fn([q.text for q in queries])
    for q, v in zip(queries, mat):
        q.embedding = v
    return queries
