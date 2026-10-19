"""Counts every LLM call made through the Ollama Python client, with the exact token counts Ollama reports."""
import time

def _g(r, key):
    v = getattr(r, key, None)
    if v is None:
        try:
            v = r[key]
        except Exception:
            v = None
    return v

def _chars(messages):
    try:
        return sum(len(str(m.get("content", "")) if isinstance(m, dict) else str(getattr(m, "content", ""))) for m in messages)
    except Exception:
        return 0

def _opts(options):
    try:
        return dict(options) if options else {}
    except Exception:
        return {"raw": str(options)}

def summarize(calls):
    return {"calls": len(calls), "seconds": sum(c["sec"] for c in calls),
            "prompt_tokens": sum(c["prompt_tokens"] for c in calls), "output_tokens": sum(c["output_tokens"] for c in calls)}

class Meter:
    def __init__(self):
        self.calls = []

    def install(self, ollama_module=None):
        """Patch ollama.Client.chat (the path Mem0's Ollama provider uses) so every call is recorded."""
        if ollama_module is None:
            import ollama as ollama_module
        orig, meter = ollama_module.Client.chat, self
        def chat(self_, *a, **k):
            t0 = time.perf_counter()
            r = orig(self_, *a, **k)
            msgs = k.get("messages") if k.get("messages") is not None else (a[1] if len(a) > 1 else None)
            meter.calls.append({"sec": time.perf_counter() - t0, "chars": _chars(msgs or []), "options": _opts(k.get("options")),
                                "prompt_tokens": _g(r, "prompt_eval_count") or 0,
                                "output_tokens": _g(r, "eval_count") or 0})
            return r
        ollama_module.Client.chat = chat

    def mark(self):
        return len(self.calls)

    def since(self, mark):
        return summarize(self.calls[mark:])
