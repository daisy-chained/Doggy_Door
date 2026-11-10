"""Score tour runs with hinge looks (option A, 2026-11-10) against the sim's true boxes, recomputed from the saved frames.

    python3 sim/building/score_hinge_look.py runs/tours/<run> [runs/tours/<run> ...] [--quiet]

Per door side: truth = a true hinge box (label 10 * door + 5..7) of that door in any reached frame of the visit. Each look is
re-checked from its saved dets.json + depth.png + robot pose: "col" = hinge at the guessed edge's image column (give or take 0.15 m),
"depth" = also at the edge's depth (median depth in the hinge box within 0.3 m; Madelief 2026-11-10: rules out another door's hinge
further back). Only looks the robot reached count; a door side whose far stop was not reached is left out. Reported for: the far stop
alone ("first", one look as before), stops only (far + 1 m), stops + 20 deg looks, stops + 20 + 40 deg (all)."""
import argparse, json, math, pathlib, sys
import numpy as np
from PIL import Image

SIM = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIM))
from doorsim import hinge_look  # noqa: E402

IMG_W, HFOV, CAM_FORWARD = 640, 1.3, 0.20      # robot camera (doorsim.robot; wheeled_* platforms: camera 0.20 m ahead of the base centre)
FX, CX = (IMG_W / 2) / math.tan(HFOV / 2), IMG_W / 2
SETS = {"first": lambda l, i: i == 0, "stops": lambda l, i: l["kind"] == "stop",
        "+20": lambda l, i: l["kind"] == "stop" or l.get("angle") == 20, "all": lambda l, i: True}


def check(frame_dir, edges):
    """(col, depth, meta): hinge checks of one saved look against its edges."""
    meta = json.loads((frame_dir / "meta.json").read_text())
    dets = json.loads((frame_dir / "dets.json").read_text())
    depth = np.asarray(Image.open(frame_dir / "depth.png"), np.float32) / 1000.0
    x, y, yaw = meta["robot_pose"]
    cx_, cy_ = x + CAM_FORWARD * math.cos(yaw), y + CAM_FORWARD * math.sin(yaw)
    col = dep = False
    for e in edges:
        c, f = hinge_look.edge_column(cx_, cy_, yaw, e, FX, CX), hinge_look.edge_forward(cx_, cy_, yaw, e)
        if c is None or f is None:
            continue
        col |= hinge_look.hinge_seen(dets, c, f, FX)
        dep |= hinge_look.hinge_seen(dets, c, f, FX, depth_img=depth)
    return col, dep, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    visits, guessed = {"col": {k: [] for k in SETS}, "depth": {k: [] for k in SETS}}, 0
    for run in map(pathlib.Path, a.runs):
        tour = json.loads((run / "tour.json").read_text())
        for v in tour["visits"]:
            hl = v.get("hinge_look")
            if not hl:
                continue
            looks = [l for l in hl["looks"] if l["result"] == "SUCCEEDED" and l["frame"]]
            if not looks or looks[0]["kind"] != "stop" or looks[0]["dist"] != v["stops"][0]["dist"]:
                continue
            edges = [tuple(e) for e in hl["edges"]]
            truth, res = False, []
            for l in looks:
                mine = edges if l["kind"] == "stop" else [edges[int(l["frame"].split("_look")[1].split("_")[0])]]
                col, dep, meta = check(run / "stops" / l["frame"], mine)
                truth |= any(b["door"] == v["door"] and b["label"] % 10 in (5, 6, 7) for b in meta["boxes"])
                res.append((l, col, dep))
            guessed += hl["guess"] is not None
            for kind, j in (("col", 1), ("depth", 2)):
                for name, keep in SETS.items():
                    seen = [r[j] for i, r in enumerate(res) if keep(r[0], i)]
                    visits[kind][name].append({"truth": truth, "first": seen[0] if seen else False, "looks": seen})
            if not a.quiet:
                print(f"{run.name[:13]} door {v['door']:>2} {v['side']:<5} truth {'Y' if truth else '-'} guess {hl['guess'] or '—':<5} "
                      f"col {''.join('Y' if r[1] else '.' for r in res):<7} depth {''.join('Y' if r[2] else '.' for r in res)}")
    print(f"\n{len(a.runs)} run(s), {len(visits['col']['all'])} door sides reached, handle guess on {guessed}")
    for kind in ("col", "depth"):
        print(f"\n{'column only' if kind == 'col' else 'column + depth'}:")
        for name in SETS:
            s = hinge_look.score_visits(visits[kind][name])
            print(f"  {name:<6} found {s['found_any']:.2f} (of {s['n_pos']})   false yes {s['false_yes_any']:.2f} (of {s['n_neg']})")


if __name__ == "__main__":
    main()
