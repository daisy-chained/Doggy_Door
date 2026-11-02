"""usage: train/.venv/bin/python train/score.py <run dir> <export dir>... [--split test] [--out results.jsonl]
Scores the run's best checkpoint on each export (made at a fixed smoke / light level) and appends one JSON line per export."""
import argparse, json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from doordet import score


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("exports", nargs="+")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", help="results file, one JSON line per scored export (appended)")
    args = ap.parse_args()
    for e in args.exports:
        r = score.score_run(args.run, e, args.split)
        if args.out:
            with open(args.out, "a") as f:
                f.write(json.dumps(r) + "\n")
        lv = {k: v for k, v in r["levels"].items() if v is not None}
        print(f"{r['run']}  {pathlib.Path(e).name}  {lv}  {r['split']} map {r['metrics']['map']:.4f}  ({r['images']} images)", flush=True)


if __name__ == "__main__":
    main()
