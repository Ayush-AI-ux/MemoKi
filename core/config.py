from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parent.parent

def load_config(path: str | Path = ROOT / "configs" / "global.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)
