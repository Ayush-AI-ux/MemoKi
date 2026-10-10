"""Score the filled-in label sheet.  python -m eval.score_labels [results/label_template.csv]"""
import csv, sys
from core.cues import normalize_entity

def items(cell):
    return {normalize_entity(x) for x in cell.split(";") if x.strip()} - {""}

def main(p="results/label_template.csv"):
    tp = fp = fn = rows = 0; warn = 0
    for r in csv.DictReader(open(p, encoding="utf-8-sig")):
        pred, wrong, miss = items(r["predicted"]), items(r["wrong"]), items(r["missing"])
        if wrong - pred:
            warn += 1
        wrong &= pred
        tp += len(pred) - len(wrong); fp += len(wrong); fn += len(miss); rows += 1
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    print(f"{rows} excerpts | predicted entities {tp + fp} | correct {tp} | wrong {fp} | missed {fn}")
    print(f"entity precision {prec:.1%} | recall {rec:.1%} | F1 {2 * prec * rec / max(prec + rec, 1e-9):.1%}")
    if warn:
        print(f"note: {warn} rows listed 'wrong' items that were not in 'predicted' (ignored)")
    print("limitation: measured on the first 1200 characters of each conversation, not the whole text")

if __name__ == "__main__":
    main(*sys.argv[1:2])
