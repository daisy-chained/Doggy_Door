"""usage: train/.venv/bin/python train/score_fusion.py <run> [<run> ...] <export dir> [--split test] [--weights 1 1 ...] [--iou 0.55] [--out fusion.json]
Late fusion (weighted box fusion) of N runs, e.g. RGB + depth + thermal; each run predicts with its own input type. Prints COCO metrics per run and fused.
--weights takes one weight per run (default: equal)."""
import argparse, json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))


def parse(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", metavar="run... export")
    ap.add_argument("--split", default="test")
    ap.add_argument("--weights", type=float, nargs="+", default=None)
    ap.add_argument("--iou", type=float, default=0.55)
    ap.add_argument("--out")
    args = ap.parse_args(argv)
    if len(args.paths) < 3:
        ap.error("need at least two runs and an export dir")
    args.runs, args.export = args.paths[:-1], args.paths[-1]
    if args.weights is not None and len(args.weights) != len(args.runs):
        ap.error(f"--weights needs one value per run ({len(args.runs)}), got {len(args.weights)}")
    return args


def main():
    args = parse(sys.argv[1:])
    from doordet import evaluate, fusion, score
    outs = [score.predict_run(r, args.export, args.split) for r in args.runs]
    ds, preds = outs[0][0], [o[1] for o in outs]
    fused = fusion.fuse_runs(preds, args.weights, iou=args.iou)
    names = [pathlib.Path(r.rstrip("/")).name for r in args.runs]
    res = {n: evaluate.coco_metrics(ds.gt_path, p) for n, p in (*zip(names, preds), ("fused", fused))}
    keys = [k for k in res["fused"] if k == "map" or k.startswith("ap_")]
    print(f"{'':10}" + "".join(f"{n[-24:]:>25}" for n in res))
    for k in keys:
        print(f"{k:10}" + "".join(f"{res[n][k]:25.3f}" for n in res))
    if args.out:
        pathlib.Path(args.out).write_text(json.dumps({"runs": args.runs, "export": args.export, "split": args.split,
                                                      "weights": args.weights or [1.0] * len(args.runs), "iou": args.iou, "metrics": res}, indent=2) + "\n")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
