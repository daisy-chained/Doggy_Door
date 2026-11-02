"""usage: train/.venv/bin/python train/plot_levels.py <results.jsonl> --axis visibility_m|light_level [--out levels.png]
mAP against smoke visibility or room light level (from train/score.py), one line per run, labelled with its GFLOPs."""
import argparse, json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from doordet import score

LABELS = {"visibility_m": "smoke visibility (m)", "light_level": "room light level"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results")
    ap.add_argument("--axis", required=True, choices=LABELS)
    ap.add_argument("--out", help="PNG (default: next to the results file)")
    args = ap.parse_args()
    rows = [json.loads(l) for l in pathlib.Path(args.results).read_text().splitlines() if l.strip()]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for run in score.table(rows, args.axis):
        x, y = zip(*run["points"])
        g = "" if run["gflops"] is None else f" ({run['gflops']:.1f} GFLOPs)"
        ax.plot(x, y, marker="o", label=run["run"] + g)
    ax.set_xscale("log")
    ax.set_xlabel(LABELS[args.axis]); ax.set_ylabel("mAP")
    ax.set_ylim(0, 1); ax.grid(alpha=0.3); ax.legend()
    out = pathlib.Path(args.out) if args.out else pathlib.Path(args.results).with_suffix(f".{args.axis}.png")
    fig.tight_layout(); fig.savefig(out, dpi=150)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
