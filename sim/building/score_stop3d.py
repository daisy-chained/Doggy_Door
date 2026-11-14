"""DoD number "door centre 3D error at the 1 m stop" (target <= 5 cm median) from saved patrol tours, no Gazebo.

    python3 sim/building/score_stop3d.py runs/tours/<run> [...] [--json out.json]

Per patrol candidate that matched a real door (patrol.match = [door, side]): its 1.0 m stop frame. The door's state at that moment =
its start state in building.json, or open once an earlier door step opened it. Box: the detector's door box (dets.json, greedy best
score first, IoU >= 0.5 with the true panel box, label 10 * door + 1), and the true panel box as a check (geometry error alone).
Estimate = doorsim.door3d.fit(depth, K, box) on the saved depth, as in the door-room number (<= 3 m, ~11-14 cm).
Truth (same definition as door3d.gt_doorway: the doorway centre on the wall face of the camera's side, at half the door height) from
Gazebo's exact 3D panel box in the camera frame: the closed panel's local x offset (doors._layout xc: hinged 0.03 * swing_x with
a door stop, sliding sgn * (0.09 + proud)) and its centre height are undone along the panel's own axes. So drift / SLAM errors in the
robot's map pose don't enter: this is the perception error at the stop. Closed doors only for the DoD row; open doors are listed apart
(the open panel is not in the doorway, so the truth above does not hold for them)."""
import argparse, json, math, pathlib, re, sys
import numpy as np
from PIL import Image

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "sim"))
from doorsim import door3d, doors, frames, radar, robot

K = radar.camera_K(robot.IMG_W, robot.IMG_H, robot.FOV)


def _qrot(q, v):
    x, y, z, w = q
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    return R @ np.asarray(v, float)


def truth(door, side, box3d):
    """Doorway centre on the camera-side wall face + normal toward the camera, optical frame; from the closed panel's 3D box."""
    c, q = frames.gl_box_to_optical(box3d["center"], box3d["orientation_xyzw"])   # q is x, y, z, w
    s = door["spec"]
    g = doors._layout(s, s["slot"])                     # the same numbers the world file was built from
    w, h = s["width"], s["height"]
    xc, zc = g["xc"] - doors.WALL_X, g["zc"]
    face = -0.05 if side == "front" else 0.05
    ax, az = _qrot(q, [1, 0, 0]), _qrot(q, [0, 0, 1])
    p = np.asarray(c) + ax * (face - xc) + az * (h / 2 - zc)
    return {"center": p.tolist(), "normal": (-ax if side == "front" else ax).tolist(), "width": w, "height": h}


def _iou(a, b):
    ix, iy = max(0, min(a[2], b[2]) - max(a[0], b[0])), max(0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    return ix * iy / u if u > 0 else 0.0


def score(run):
    t = json.loads((run / "tour.json").read_text())
    pl = t.get("patrol")
    if pl is None:
        return []
    split, bid = re.match(r"bld_([a-z]+)_(\d{3})", run.name).groups()
    b = json.loads((REPO / f"sim/building/out/bld_{split}_{bid}.building.json").read_text())
    bd = {d["id"]: d for d in b["doors"]}
    opened, rows = set(), []
    for v in t["visits"]:
        k = v.get("cand")
        m = pl["candidates"][k]["match"] if k is not None and k < len(pl["candidates"]) else None
        if m:
            did, side = m
            d = bd[did]
            state = "open" if (d["state"] != "closed" or did in opened) else "closed"
            for s in v["stops"]:
                if s.get("dist") != 1.0 or not s.get("frame"):
                    continue
                f = run / "stops" / s["frame"]
                meta = json.loads((f / "meta.json").read_text())
                p2 = next((x for x in meta["boxes"] if x["label"] == 10 * did + 1), None)
                p3 = next((x for x in meta["boxes_3d"] if x["label"] == 10 * did + 1), None)
                row = {"run": run.name, "frame": s["frame"], "door": did, "side": side, "state": state,
                       "glass": d["spec"]["glass"], "type": d["spec"]["door_type"], "in_view": p2 is not None}
                if p2 is None or p3 is None:
                    rows.append(row); continue
                gt = truth(d, side, p3)
                depth = np.asarray(Image.open(f / "depth.png"), np.float64) / 1000.0
                gbox = [p2["x_min"], p2["y_min"], p2["x_max"] + 1, p2["y_max"] + 1]
                dets = json.loads((f / "dets.json").read_text()) if (f / "dets.json").exists() else {"boxes": [], "scores": [], "labels": []}
                best = None
                for i in sorted(range(len(dets["boxes"])), key=lambda i: -dets["scores"][i]):
                    if dets["labels"][i] == "door" and _iou(dets["boxes"][i], gbox) >= 0.5:
                        best = dets["boxes"][i]; break
                for name, bx in (("det", best), ("gt", gbox)):
                    if bx is None:
                        row[name] = None; continue
                    H, W = depth.shape
                    box = {"x_min": max(0, int(math.floor(bx[0]))), "y_min": max(0, int(math.floor(bx[1]))),
                           "x_max": min(W - 1, int(math.ceil(bx[2])) - 1), "y_max": min(H - 1, int(math.ceil(bx[3])) - 1)}
                    est = door3d.fit(depth, K, box)
                    if est is None:
                        row[name] = None; continue
                    dv = np.asarray(est["center"]) - np.asarray(gt["center"])
                    # floor plane = optical x (sideways) + z (ahead); the camera is level, so optical y is height (Claude 2026-11-14:
                    # at 1 m the low camera never sees the door top, so the full 3D centre is off by the cut-off height)
                    row[name] = {**door3d.errors(est, gt), "floor_m": float(math.hypot(dv[0], dv[2])), "height_off_m": float(dv[1]), "side_m": float(dv[0]), "ahead_m": float(dv[2]),
                                 "source": est["source"], "clipped": door3d.clipped(box, depth.shape)}
                rows.append(row)
            if (v.get("door_step") or {}).get("opened"):
                opened.add(did)
    return rows


def summary(rows):
    out = []
    for label, sel in (("closed", [r for r in rows if r["state"] == "closed"]), ("open", [r for r in rows if r["state"] == "open"])):
        for key in ("det", "gt"):
            e = [r[key]["center_m"] for r in sel if r.get(key)]
            fl = [r[key]["floor_m"] for r in sel if r.get(key)]
            n_dist = [r[key]["distance_m"] for r in sel if r.get(key)]
            miss = sum(1 for r in sel if r["in_view"] and not r.get(key))
            out.append(f"{label:6} {key:3} n={len(e):3d} (in view {sum(r['in_view'] for r in sel)}, no box/fit {miss}, not in view "
                       f"{sum(not r['in_view'] for r in sel)})  centre error median {np.median(e) * 100 if e else float('nan'):.1f} cm, "
                       f"p90 {np.percentile(e, 90) * 100 if e else float('nan'):.1f} cm; floor plane median {np.median(fl) * 100 if fl else float('nan'):.1f} cm, "
                       f"p90 {np.percentile(fl, 90) * 100 if fl else float('nan'):.1f} cm; distance median {np.median(n_dist) if n_dist else float('nan'):.2f} m")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    rows = [r for run in a.runs for r in score(pathlib.Path(run))]
    print(summary(rows))
    if a.json:
        pathlib.Path(a.json).write_text(json.dumps(rows, indent=1) + "\n")


if __name__ == "__main__":
    main()
