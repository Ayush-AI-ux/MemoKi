"""Recall vs cost curves from the CSV written by eval.compare_check.
   python -m eval.compare_check --sizes 1000 5000 10000 --efs 5 8 16 32 64 128 --beams 1 2 3 5 10 20 --csv results/pareto.csv
   python -m eval.plot_pareto results/pareto.csv          -> results/pareto_<size>.png"""
import csv, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FAMILIES = [("brute force", "brute force", "k", "s"), ("hnsw ef", "HNSW", "tab:blue", "o"),
            ("filter + hnsw", "metadata filter + HNSW", "tab:green", "^"), ("flat + cue filter", "flat + cue filter", "tab:olive", "D"),
            ("tree no cues", "tree, no cues", "tab:gray", "v"), ("CUE TREE", "CUE TREE (proposed)", "tab:red", "*")]

def family(name):
    if name == "faiss flat":
        return None
    for key, label, color, marker in FAMILIES:
        if name.startswith(key):
            return label, color, marker
    return None

def main(path):
    rows = list(csv.DictReader(open(path)))
    for size in sorted({int(r["size"]) for r in rows}):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=False)
        for ax, col, title in ((axes[0], "all", "all queries"), (axes[1], "cue", "queries WITH a usable cue")):
            seen = set()
            for fam in FAMILIES:
                pts = sorted([(float(r["comps_" + col]), float(r["recall_" + col])) for r in rows
                              if int(r["size"]) == size and (family(r["method"]) or ("",))[0] == fam[1]])
                if not pts:
                    continue
                ax.plot(*zip(*pts), color=fam[2], marker=fam[3], ms=9 if fam[3] == "*" else 6, lw=1.4, label=fam[1])
            ax.set_xscale("log"); ax.set_xlabel("embedding comparisons per query (log scale)")
            ax.set_ylabel("Recall@5"); ax.set_title(f"size {size}: {title}"); ax.grid(alpha=.3)
        axes[0].legend(fontsize=8, loc="lower right")
        fig.suptitle("Recall vs cost, validation split (up and to the left is better)", fontsize=10)
        fig.tight_layout(); out = f"results/pareto_{size}.png"; fig.savefig(out, dpi=150); plt.close(fig)
        print("wrote", out)

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/pareto.csv")
