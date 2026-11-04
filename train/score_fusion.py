"""usage: train/.venv/bin/python train/score_fusion.py <run A> <run B> <export dir> [--split test] [--weights 1 1] [--iou 0.55] [--out fusion.json]
Late fusion (weighted box fusion) of two runs, e.g. RGB + depth; each run predicts with its own input type. Prints COCO metrics for A, B and fused."""
import argparse, json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from doordet import evaluate, fusion, score


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_a")
    ap.add_argument("run_b")
    ap.add_argument("export")
    ap.add_argument("--split", default="test")
    ap.add_argument("--weights", type=float, nargs=2, default=[1, 1])
    ap.add_argument("--iou", type=float, default=0.55)
    ap.add_argument("--out")
    args = ap.parse_args()
    (ds, pa, _), (_, pb, _) = (score.predict_run(r, args.export, args.split) for r in (args.run_a, args.run_b))
    fused = fusion.fuse_runs([pa, pb], args.weights, iou=args.iou)
    res = {name: evaluate.coco_metrics(ds.gt_path, p) for name, p in (("A", pa), ("B", pb), ("fused", fused))}
    keys = [k for k in res["A"] if k == "map" or k.startswith("ap_")]
    print(f"{'':10}" + "".join(f"{n:>9}" for n in res))
    for k in keys:
        print(f"{k:10}" + "".join(f"{res[n][k]:9.3f}" for n in res))
    if args.out:
        pathlib.Path(args.out).write_text(json.dumps({"runs": [args.run_a, args.run_b], "export": args.export, "split": args.split,
                                                      "weights": args.weights, "iou": args.iou, "metrics": res}, indent=2) + "\n")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
