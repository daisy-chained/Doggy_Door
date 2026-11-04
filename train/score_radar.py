"""usage: train/.venv/bin/python train/score_radar.py <run> <radar export dir> [--split test] [--thr-split val] [--out radar.json]
Scores a doorway detector trained on a radar BEV export (sim/doorsim/radar_export.py) the way the geometric finders are scored
(radar_doorways): predicted boxes -> doorway centres in world metres (radar_heatmap.bev_to_world), matched to the open/closed doorway
truth of the target frame. The score threshold maximises F1 on --thr-split, then applies to --split. Also COCO mAP on --split."""
import argparse, json, pathlib, sys
import numpy as np
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "sim"))
GRID = np.round(np.arange(0.05, 0.901, 0.05), 2)


def rows_for(ds, preds, thr, info, src, K):
    """Score rows (radar_doorways.score + in_view per row) of every image at one threshold."""
    from doorsim import effects, radar_doorways as rd, radar_heatmap as rh
    rows = []
    for im in ds.images:
        meta = json.loads((src / "frames" / im["frame"] / "meta.json").read_text())
        cam = meta["camera"]
        p = preds.get(im["id"]) or {"boxes": [], "scores": []}
        boxes, scores = np.asarray(p["boxes"], float).reshape(-1, 4), np.asarray(p["scores"], float)
        found = [{"center": rh.bev_to_world((b[0] + b[2]) / 2, (b[1] + b[3]) / 2, cam, info["size_m"], info["res"]),
                  "width": float(abs(b[2] - b[0]) * info["res"])} for b, s in zip(boxes, scores) if s >= thr]
        gt = rd.ground_truth(meta)
        depth = effects.depth_m(src / "frames" / im["frame"] / "radar_depth.png")
        r = rd.score(found, gt, cam["world_position"])
        for x in r:
            if x["door"] is not None:
                x["in_view"] = bool(rd.in_view(gt[x["door"]], cam, K, depth))
            x["frame"] = im["frame"]
        rows += r
    return rows


def metrics(rows):
    from doorsim import radar_doorways as rd
    m = rd.detection_metrics(rows)
    err = [r["center_m"] for r in rows if r.get("door") is not None and r["found"] and str(r["state"]) == "open" and r.get("in_view")]
    m["centre_err_median_m"] = float(np.median(err)) if err else None
    m["centre_err_n"] = len(err)
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("export")
    ap.add_argument("--split", default="test")
    ap.add_argument("--thr-split", default="val")
    ap.add_argument("--out")
    a = ap.parse_args()
    from doordet import evaluate, score
    from doorsim import radar_doorways as rd
    info = json.loads((pathlib.Path(a.export) / "export.json").read_text())
    src = pathlib.Path(info["source"]["path"])
    K = rd._camera_K(src)
    ds_v, pv, _ = score.predict_run(a.run, a.export, a.thr_split)
    sweep = {float(t): metrics(rows_for(ds_v, pv, float(t), info, src, K)) for t in GRID}
    thr = max(sweep, key=lambda t: (sweep[t]["f1"], -t))  # ties: the lower threshold
    ds, preds, _ = score.predict_run(a.run, a.export, a.split)
    m = metrics(rows_for(ds, preds, thr, info, src, K))
    coco = evaluate.coco_metrics(ds.gt_path, preds)
    res = {"run": a.run, "export": a.export, "model": info["model"], "views": info["views"], "split": a.split, "thr_split": a.thr_split,
           "threshold": thr, "images": len(ds), "metrics": m, "coco": coco, "f1_by_threshold": {str(t): s["f1"] for t, s in sweep.items()}}
    print(f"{pathlib.Path(a.run.rstrip('/')).name} on {a.split} ({info['model']}, {info['views']} views), threshold {thr:.2f} (best F1 on {a.thr_split})")
    print(f"  open doors in view: recall={m['recall']:.2f} precision={m['precision']:.2f} F1={m['f1']:.2f} (tp={m['tp']} fp={m['fp']} fn={m['fn']})")
    err = m["centre_err_median_m"]
    print(f"  centre error median: {'-' if err is None else f'{err:.2f} m'} (n={m['centre_err_n']})")
    print(f"  COCO map={coco['map']:.3f} map_50={coco.get('map_50', float('nan')):.3f}")
    if a.out:
        pathlib.Path(a.out).write_text(json.dumps(res, indent=2) + "\n")
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
