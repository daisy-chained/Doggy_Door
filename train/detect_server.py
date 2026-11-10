"""Detector as a long-running helper for the robot tour (hinge look, option A, 2026-11-10).

The tour runs under the ROS python (no torch), so it starts this once under train/.venv and talks over stdin / stdout:
one image path per line in, one JSON line out: {"boxes": [[x1, y1, x2, y2], ...], "scores": [...], "labels": [names]}
(score >= --score-min). Prints "READY" once the model is loaded.

  train/.venv/bin/python train/detect_server.py runs/dfine_n960_v7_hw8 [--score-min 0.1] [--device cuda]
"""
import argparse, json, pathlib, sys
import torch
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from doordet import adapters


def named(result, id2label, score_min):
    """Post-processed HF result (tensors / arrays) -> plain dict with label names, scores >= score_min."""
    keep = [i for i, s in enumerate(result["scores"]) if float(s) >= score_min]
    return {"boxes": [[float(v) for v in result["boxes"][i]] for i in keep],
            "scores": [float(result["scores"][i]) for i in keep],
            "labels": [id2label[int(result["labels"][i])] for i in keep]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--score-min", type=float, default=0.1)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    run = pathlib.Path(a.run)
    cfg = json.loads((run / "config.json").read_text())
    cats = json.loads((pathlib.Path(cfg["export"]) / "coco" / "test.json").read_text())["categories"]
    id2label = {c["id"] - 1: c["name"] for c in cats}
    ad = adapters.get(cfg["model"]["family"])({**cfg, "model": {**cfg["model"], "checkpoint": str(run / "best")}}, id2label)
    dev = torch.device(a.device)
    ad.model.to(dev).eval()
    print("READY", flush=True)
    for line in sys.stdin:
        path = line.strip()
        if not path:
            continue
        try:
            img = Image.open(path).convert("RGB")
            enc = ad.processor(images=[img], return_tensors="pt").to(dev)
            with torch.no_grad():
                out = ad.model(**enc)
            r = ad.processor.post_process_object_detection(out, threshold=a.score_min, target_sizes=torch.tensor([img.size[::-1]]))[0]
            print(json.dumps(named({k: v.cpu().numpy() for k, v in r.items()}, id2label, a.score_min)), flush=True)
        except Exception as e:   # keep serving; the tour logs it and treats the look as "nothing seen"
            print(json.dumps({"error": repr(e), "boxes": [], "scores": [], "labels": []}), flush=True)


if __name__ == "__main__":
    main()
