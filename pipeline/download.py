"""Download raw datasets. Run: python -m pipeline.download [--full]"""
import argparse, urllib.request
from .config import load_all, path

HF = "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/"
GH = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"

def _fetch(url, dest):
    if dest.exists():
        print(f"[skip] {dest.name} already exists"); return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[get ] {url}")
    def hook(b, bs, total):
        if total > 0 and b % 200 == 0:
            print(f"       {min(100, b*bs*100//total)}%", end="\r")
    urllib.request.urlretrieve(url, dest, hook)
    print(f"[done] {dest} ({dest.stat().st_size/1e6:.1f} MB)")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="also download the large haystack file (~270 MB)")
    a = ap.parse_args()
    cfg = load_all()
    _fetch(GH, path(cfg, "locomo"))
    _fetch(HF + "longmemeval_oracle.json", path(cfg, "longmemeval_oracle"))
    if a.full:
        _fetch(HF + "longmemeval_s_cleaned.json", path(cfg, "longmemeval"))

if __name__ == "__main__":
    main()
