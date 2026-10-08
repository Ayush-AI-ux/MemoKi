"""Embed the ENTIRE LongMemEval pool once (about 19k sessions, cached, resumable in chunks).
After this, any split / seed / store size reuses the cache.   python -m pipeline.embed_pool"""
from core.config import ROOT
from .config import load_all, path
from .loaders import load_longmemeval
from .embed import Embedder, embed_conversations

def main(encode_fn=None, chunk=2000):
    cfg = load_all(); d = cfg["data"]
    pool, _ = load_longmemeval(path(cfg, "longmemeval"))
    convs = [pool[i] for i in sorted(pool)]
    enc = encode_fn or Embedder(cfg["embedding_model"], d["max_seq_length"]).encode
    cache = ROOT / d["paths"]["cache_dir"]
    for i in range(0, len(convs), chunk):
        embed_conversations(convs[i:i + chunk], enc, cache, cfg["embedding_model"], d["max_seq_length"])
        print(f"[pool] {min(i + chunk, len(convs))}/{len(convs)} embedded")

if __name__ == "__main__":
    main()
