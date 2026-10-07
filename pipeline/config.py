from core.config import ROOT, load_config

def load_all() -> dict:
    cfg = load_config()
    cfg["data"] = load_config(ROOT / "configs" / "data.yaml")
    return cfg

def path(cfg: dict, key: str):
    return ROOT / cfg["data"]["paths"][key]
