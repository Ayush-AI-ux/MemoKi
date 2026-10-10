"""Hand-labelling sheet to measure how accurate the entity cues are (Gap 2).
   python -m eval.label_sample --n 100       -> results/label_template.csv   (open in Excel / VS Code)
For each conversation excerpt (first 1200 characters) the sheet shows the entities the extractor found.
Fill two columns, separating items with ';':
   wrong   = items in 'predicted' that are NOT real names, places, organisations, products or events
   missing = real names, places, organisations, products or events in the excerpt that were NOT predicted
Leave a cell empty when there is nothing to add. Then run:  python -m eval.score_labels"""
import argparse, csv, random
from core.cues import SpacyNER
from pipeline.config import load_all, path
from pipeline.loaders import load_longmemeval

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=100); ap.add_argument("--chars", type=int, default=1200)
    ap.add_argument("--out", default="results/label_template.csv"); a = ap.parse_args()
    cfg = load_all()
    pool, _ = load_longmemeval(path(cfg, "longmemeval"))
    ids = sorted(pool); random.Random(2026).shuffle(ids); ids = ids[:a.n]
    ner = SpacyNER(max_chars=a.chars)
    excerpts = [pool[i].text[:a.chars] for i in ids]
    preds = ner(excerpts)
    with open(a.out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(["id", "excerpt", "predicted", "wrong", "missing"])
        for i, ex, p in zip(ids, excerpts, preds):
            w.writerow([i, ex.replace("\n", " | "), "; ".join(sorted(p)), "", ""])
    print(f"wrote {len(ids)} rows to {a.out}")

if __name__ == "__main__":
    main()
