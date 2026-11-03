"""usage: train/.venv/bin/python train/score_props.py <props run> <detector run> <export dir> [--split test] [--score-min 0.3] [--device cpu]
Props classifier accuracy on the true boxes next to its accuracy on the detector's own boxes; writes <props run>/detector_boxes_<export name>.json."""
import argparse, json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from doordet import props, score


def true_box_acc(run, export, split):
    """Accuracy per head on the GT boxes (the dataset the run was trained with, no jitter)."""
    from torch.utils.data import DataLoader
    model, meta = props.load(run, "cpu")
    ds = props.CropDataset(export, split, meta["crop"], meta["imgsz"], meta["margin"], meta["min_px"])
    pr = props.predict(model, DataLoader(ds, batch_size=32), "cpu", list(meta["heads"]))
    return {f"acc_{h}": (lg.argmax(1) == lb).float().mean().item() for h, (lg, lb) in pr.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("props_run")
    ap.add_argument("detector_run")
    ap.add_argument("export")
    ap.add_argument("--split", default="test")
    ap.add_argument("--score-min", type=float, default=0.3)
    ap.add_argument("--device", help="detector device (default: from its config)")
    args = ap.parse_args()
    _, preds, _ = score.predict_run(args.detector_run, args.export, args.split, args.device)
    det = props.score_on_boxes(args.props_run, args.export, args.split, preds, args.score_min)
    true = true_box_acc(args.props_run, args.export, args.split)
    print(f"{det['crops']} GT crops: {det['matched']} matched, {det['missed']} missed, {det['false']} false detections (score >= {args.score_min})")
    for k, v in true.items():
        print(f"  {k:<22} true boxes {v:.3f}   detector boxes {det.get(k, float('nan')):.3f}")
    out = pathlib.Path(args.props_run) / f"detector_boxes_{pathlib.Path(args.export).name}.json"
    out.write_text(json.dumps({"export": args.export, "detector": args.detector_run, "split": args.split, "score_min": args.score_min,
                               "true_boxes": true, "detector_boxes": det}, indent=2) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
