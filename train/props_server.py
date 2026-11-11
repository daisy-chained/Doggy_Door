"""Door-crop property classifier as a long-running helper for the robot tour (reactive door step, 2026-11-11).

Starts once under train/.venv (the tour's ROS python has no torch); one JSON line in, one JSON line out. Prints "READY" once loaded.
  in:  {"rgb": path, "depth": path (uint16 png), "box": [x1, y1, x2, y2], "K": [9 floats], "camera": {"world_position": [x, y, z], "world_rpy": [r, p, y]}}
  out: {"opening": {"push": p, ...}, "hinge_side": {...}, "glass": {...}} (softmax of logits / temperature), or {"skip": "small"}
       when max(box w, h) < meta["min_px"]; {"error": ...} on failure.

  train/.venv/bin/python train/props_server.py runs/props_door_v7_dha_448 [--device cuda]
"""
import argparse, json, pathlib, sys
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from doordet import props, data


def classify(model, meta, heads, T, dev, req):
    x1, y1, x2, y2 = req["box"]
    if max(x2 - x1, y2 - y1) < meta["min_px"]:
        return {"skip": "small"}
    crop = props.crop_box((x1, y1, x2 - x1, y2 - y1), meta["margin"], None, None)
    parts = []
    if meta["input"] in ("rgb", "rgb_dha"):
        parts.append(props.crop_tensor(Image.open(req["rgb"]).convert("RGB"), crop, meta["imgsz"]))
    if meta["input"] in ("dha", "rgb_dha"):
        dha = Image.fromarray(data.encode_depth(np.asarray(Image.open(req["depth"])), req["K"], req["camera"], "dha"))
        parts.append(props.crop_tensor(dha, crop, meta["imgsz"], props.DHA_MEAN, props.DHA_STD))
    xin = torch.cat(parts)[None].to(dev)
    with torch.no_grad():
        logits = model(xin)
    out = {}
    for h in heads:
        lg = logits[h] if isinstance(logits, dict) else logits[heads.index(h)]
        t = T.get(h, 1.0) if isinstance(T, dict) else 1.0
        p = torch.softmax(lg[0] / t, -1).cpu().tolist()
        out[h] = dict(zip(meta["heads"][h], p))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    model, meta = props.load(a.run, a.device)
    heads = list(meta["heads"])
    T = meta.get("temperature") or {}
    print("READY", flush=True)
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            print(json.dumps(classify(model, meta, heads, T, a.device, json.loads(line))), flush=True)
        except Exception as e:   # keep serving
            print(json.dumps({"error": repr(e)}), flush=True)


if __name__ == "__main__":
    main()
