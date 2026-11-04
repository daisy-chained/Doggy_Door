"""Score a trained run on another export (e.g. fixed smoke / light levels) and tabulate scores per level."""
import json, pathlib
from torch.utils.data import DataLoader

from . import adapters, evaluate, loop
from .data import CocoDetection


def levels(export_dir):
    """{"visibility_m", "light_level", "headlight"} an export was made at (from export.json); None when missing."""
    src = (json.loads((pathlib.Path(export_dir) / "export.json").read_text()).get("source")) or {}
    fx, light = src.get("effects") or {}, src.get("lighting") or {}
    return {"visibility_m": fx.get("visibility_m"), "light_level": light.get("level"), "headlight": light.get("headlight")}


def predict_run(run_dir, export_dir, split="test", device=None):
    """Best checkpoint of a run on one split of an export -> (CocoDetection, predictions {image_id: boxes xyxy, scores, labels}, run config).
    device overrides the run's config."""
    import torch
    run_dir, export_dir = pathlib.Path(run_dir), pathlib.Path(export_dir)
    cfg = json.loads((run_dir / "config.json").read_text())
    use = {**cfg, "model": {**cfg["model"], "checkpoint": str(run_dir / "best")}, "export": str(export_dir)}
    ds = CocoDetection(export_dir, split, input=cfg.get("input", "rgb"))
    adapter = adapters.get(cfg["model"]["family"])(use, ds.id2label)
    device = torch.device(device or use["device"] or ("cuda" if torch.cuda.is_available() else "cpu"))
    adapter.model.to(device)
    loader = DataLoader(ds, batch_size=use["batch"], shuffle=False, num_workers=use["workers"], collate_fn=adapter.collate)
    preds, _ = loop.predict_all(adapter, loader, device, use)
    return ds, preds, cfg


def score_run(run_dir, export_dir, split="test"):
    """Best checkpoint of a run on one split of an export -> {run, export, split, images, model, imgsz, levels, metrics, gflops}."""
    run_dir, export_dir = pathlib.Path(run_dir), pathlib.Path(export_dir)
    ds, preds, cfg = predict_run(run_dir, export_dir, split)
    model = {k: v for k, v in cfg["model"].items() if v is not None}
    return {"run": run_dir.name, "export": str(export_dir), "split": split, "images": len(ds), "model": model,
            "imgsz": cfg["imgsz"], "levels": levels(export_dir), "metrics": evaluate.coco_metrics(ds.gt_path, preds),
            "gflops": cfg.get("gflops")}


def table(rows, axis):
    """Score rows -> one dict per run, sorted by GFLOPs (unknown last) then name:
    {"run", "gflops", "points": sorted [(level on `axis`, mAP)]}; rows without that level are skipped."""
    runs = {}
    for r in rows:
        level = r["levels"].get(axis)
        if level is None:
            continue
        run = runs.setdefault(r["run"], {"run": r["run"], "gflops": r.get("gflops"), "points": []})
        run["points"].append((level, r["metrics"]["map"]))
    for run in runs.values():
        run["points"].sort()
    return sorted(runs.values(), key=lambda r: (r["gflops"] is None, r["gflops"] or 0, r["run"]))
