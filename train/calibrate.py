"""usage: train/.venv/bin/python train/calibrate.py <run> [<run> ...] --fit <export> --eval <export> [<export> ...] --out DIR [--fit-split val] [--split test] [--score-min 0.1] [--iou 0.5] [--adaptive]
Detector confidence calibration (2d, Madelief 2026-11-06): per-class temperature scaling (Guo et al. 2017) and box-aware logistic calibration
(Kuppers et al. 2020), fitted on the --fit export's --fit-split (e.g. mixed val), tested on every --eval export's --split, by D-ECE and mAP.
Several runs are fused first (weighted box fusion; --adaptive = per-image quality weights, like score_fusion.py) and the fused output is calibrated.
Writes DIR/calibrators.json and DIR/<export name>.json per eval export."""
import argparse, json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

EDGES = [0, 0.05, 0.1, 0.2, 0.4, 1.0]


def parse(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--fit", required=True)
    ap.add_argument("--fit-split", default="val")
    ap.add_argument("--eval", nargs="+", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--score-min", type=float, default=0.1)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--adaptive", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    if args.adaptive and len(args.runs) < 2:
        ap.error("--adaptive needs at least two runs")
    return args


def predictions(args, export, split):
    """(gt json path, predictions) of the runs on one export split; fused when there are several runs."""
    from doordet import fusion, score
    outs = [score.predict_run(r, export, split) for r in args.runs]
    if len(outs) == 1:
        return outs[0][0].gt_path, outs[0][1]
    iw = None
    if args.adaptive:
        from doordet import quality
        iw, _ = quality.adaptive_weights(args.runs, export, split)
    return outs[0][0].gt_path, fusion.fuse_runs([o[1] for o in outs], image_weights=iw)


def main():
    args = parse(sys.argv[1:])
    from doordet import calibrate, evaluate
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    gt, preds = predictions(args, args.fit, args.fit_split)
    fit_rows = calibrate.rows(gt, preds, args.score_min, args.iou)
    cals = {k: calibrate.fit(fit_rows, k) for k in ("temperature", "box")}
    (out / "calibrators.json").write_text(json.dumps({"fit": args.fit, "fit_split": args.fit_split, "runs": args.runs, "adaptive": args.adaptive,
                                                      "n_rows": len(fit_rows), **{k: calibrate.to_json(c) for k, c in cals.items()}}, indent=2) + "\n")
    print(f"{'export':28}{'n':>7}{'raw':>8}{'temp':>8}{'box':>8}{'mAP raw':>10}{'mAP box':>9}")
    for export in args.eval:
        gt, preds = predictions(args, export, args.split)
        rs = calibrate.rows(gt, preds, args.score_min, args.iou)
        res = {}
        for name, cal in (("raw", None), *cals.items()):
            r = rs if cal is None else calibrate.apply_rows(cal, rs)
            p = preds if cal is None else calibrate.apply_preds(cal, preds, gt)
            res[name] = {"dece": calibrate.dece(r), "dece_by_size": calibrate.dece_by_size(r, EDGES), "metrics": evaluate.coco_metrics(gt, p)}
        name = pathlib.Path(export.rstrip("/")).name
        (out / f"{name}.json").write_text(json.dumps({"export": export, "split": args.split, "fit": args.fit, "fit_split": args.fit_split, "runs": args.runs,
                                                      "adaptive": args.adaptive, "n_rows": len(rs), **res}, indent=2) + "\n")
        d = lambda k: res[k]["dece"]["all"]
        f = lambda v: f"{v:8.3f}" if v is not None else f"{'-':>8}"
        print(f"{name[-27:]:28}{len(rs):7}{f(d('raw'))}{f(d('temperature'))}{f(d('box'))}{res['raw']['metrics']['map']:10.3f}{res['box']['metrics']['map']:9.3f}")


if __name__ == "__main__":
    main()
