"""usage: train/.venv/bin/python train/score_cascade.py <door run> <handle run> <detector run> <export dir> [--split test] [--score-min 0.3] [--type-run RUN] [--hinges] [--device cpu] [--out FILE]
Door property cascade (door type first, then hinged-only questions) on the true boxes next to the detector's own boxes, with the flat door
classifier's numbers; writes <door run>/cascade_<export name>.json."""
import argparse, json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from doordet import cascade, score


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("door_run")
    ap.add_argument("handle_run")
    ap.add_argument("detector_run")
    ap.add_argument("export")
    ap.add_argument("--split", default="test")
    ap.add_argument("--score-min", type=float, default=0.3)
    ap.add_argument("--type-run", help="props run with a door_type head (default: P(slide) of the door run's opening head)")
    ap.add_argument("--hinges", action="store_true", help="step B: use hinge boxes (class 3; GT or detector) for opening / hinge side")
    ap.add_argument("--device", help="detector device (default: from its config)")
    ap.add_argument("--out", help="default: <door run>/cascade_<export name>.json")
    args = ap.parse_args()
    _, preds, _ = score.predict_run(args.detector_run, args.export, args.split, args.device)
    kw = dict(split=args.split, type_run=args.type_run, hinges=args.hinges)
    true = cascade.score(args.door_run, args.handle_run, args.export, preds=None, **kw)
    det = cascade.score(args.door_run, args.handle_run, args.export, preds=preds, score_min=args.score_min, **kw)
    print(f"{true['doors']} doors on true boxes, {det['doors']} matched detector boxes (score >= {args.score_min})")
    for k in true:
        if k != "doors":
            print(f"  {k:<26} true boxes {true[k]:.3f}   detector boxes {det.get(k, float('nan')):.3f}")
    out = pathlib.Path(args.out) if args.out else pathlib.Path(args.door_run) / f"cascade_{pathlib.Path(args.export).name}.json"
    out.write_text(json.dumps({"door_run": args.door_run, "handle_run": args.handle_run, "type_run": args.type_run, "detector": args.detector_run,
                               "export": args.export, "split": args.split, "score_min": args.score_min,
                               "true_boxes": true, "detector_boxes": det}, indent=2) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
