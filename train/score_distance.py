"""usage: train/.venv/bin/python train/score_distance.py <run dir> <export dir> [--split test] [--bins 0 1 2 3 4] [--score-min 0.3] [--device cpu]
Detector recall per camera distance (m) for doors and handles; writes <run dir>/distance_<export name>.json."""
import argparse, json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from doordet import evaluate, score


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("export")
    ap.add_argument("--split", default="test")
    ap.add_argument("--bins", type=float, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--score-min", type=float, default=0.3)
    ap.add_argument("--device")
    args = ap.parse_args()
    ds, preds, _ = score.predict_run(args.run, args.export, args.split, args.device)
    res = {c: evaluate.recall_by_distance(ds.gt_path, preds, c, args.bins, args.score_min) for c in ("door", "handle")}
    for c, rows in res.items():
        print(c)
        for b, r in rows.items():
            print(f"  {b:>7} m  n {r['n']:4d}  found {r['found']:4d}  recall {'-' if r['recall'] is None else format(r['recall'], '.3f')}")
    out = pathlib.Path(args.run) / f"distance_{pathlib.Path(args.export).name}.json"
    out.write_text(json.dumps({"export": args.export, "split": args.split, "score_min": args.score_min, "recall": res}, indent=2) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
