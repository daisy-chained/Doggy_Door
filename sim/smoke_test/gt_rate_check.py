"""Do the tours run faster with the ground-truth cameras (boxes, segmentation) at 2 Hz instead of 10, with the same labels?

    source /opt/ros/jazzy/setup.bash && python3 sim/smoke_test/gt_rate_check.py --out runs/scratch/gt_rate_check --apply runs/scratch/gpu_env.sh

Madelief 2026-11-14 ("is there a way to speed up the runs?" -> "you see about the other things"). Two short real tours on
bld_train_005 (wheeled_ring, ground-truth odometry, normal light, --max-visits 3): DOORSIM_GT_HZ unset (10 Hz) and 2. Measured:
Gazebo speed (sim s per real s, from /stats), wall minutes; per stop present in both runs: the labelled objects (boxes >= 200 px)
and the 2D box IoU per label. The robot stops a few cm apart between runs, so IoU is not 1 even if nothing changed.
Passes when: speedup >= 1.10, the same labels at >= 90 % of the shared stops, median IoU >= 0.85. Then (with --apply) appends
`export DOORSIM_GT_HZ=2` to that file, which the queued tour jobs source at their start. Output: OUT/result.json, OUT/report.md."""
import argparse, json, os, signal, subprocess, sys, time
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
from gpu_ram_check import sim_real, rtf, gz_servers   # noqa: E402

PATTERN = "bld_train_005_wheeled_ring_2*"   # no SLAM / drift / conditions in the name


def tour(hz, out, limit=900):
    env = dict(os.environ, SLAM="0", ODOM_NOISE="0")
    env.pop("LIGHT", None); env.pop("SMOKE", None)
    if hz:
        env["DOORSIM_GT_HZ"] = str(hz)
    else:
        env.pop("DOORSIM_GT_HZ", None)
    before = {d.name for d in (REPO / "runs/tours").glob(PATTERN)}
    p = subprocess.Popen(["bash", "sim/building/tour_test.sh", "sim/building/out/bld_train_005", "sim/configs/robots/wheeled_ring.yaml",
                          "1.0", "4.915", "--max-visits", "3"], cwd=REPO, env=env, stdout=open(out / f"tour_{hz or 10}.log", "w"),
                         stderr=subprocess.STDOUT, start_new_session=True)
    part, st, t0, k = f"tour_{p.pid}", [], time.time(), 0
    while p.poll() is None and time.time() - t0 < limit:
        time.sleep(5); k += 1
        if k % 2 == 0 and gz_servers(part):
            x = sim_real(part)
            if x:
                st.append(x)
    timed_out = p.poll() is None
    if timed_out:
        os.killpg(p.pid, signal.SIGINT); time.sleep(10)
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    new = [d for d in (REPO / "runs/tours").glob(PATTERN) if d.name not in before]
    tdir = max(new, key=lambda d: d.stat().st_mtime) if new else None
    return {"gt_hz": hz or 10, "rtf": rtf(st), "rtf_samples": len(st), "minutes": round((time.time() - t0) / 60, 1),
            "exit": p.returncode, "timed_out": timed_out, "tour_dir": str(tdir) if tdir else None}


def boxes(stop):
    m = json.loads((stop / "meta.json").read_text())
    return {b["label"]: (b["x_min"], b["y_min"], b["x_max"], b["y_max"]) for b in m.get("boxes", [])
            if (b["x_max"] - b["x_min"]) * (b["y_max"] - b["y_min"]) >= 200}


def iou(a, b):
    ix, iy = max(0, min(a[2], b[2]) - max(a[0], b[0])), max(0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    return ix * iy / u if u > 0 else 0.0


def compare(d10, d2):
    s10 = {s.name: s for s in (Path(d10) / "stops").iterdir() if (s / "meta.json").exists()}
    s2 = {s.name: s for s in (Path(d2) / "stops").iterdir() if (s / "meta.json").exists()}
    shared = sorted(set(s10) & set(s2))
    same, ious = 0, []
    for n in shared:
        a, b = boxes(s10[n]), boxes(s2[n])
        same += set(a) == set(b)
        ious += [iou(a[l], b[l]) for l in set(a) & set(b)]
    return {"stops_10": len(s10), "stops_2": len(s2), "shared": len(shared), "same_labels": same,
            "same_label_rate": round(same / len(shared), 3) if shared else None,
            "iou_median": round(float(np.median(ious)), 3) if ious else None, "iou_p10": round(float(np.percentile(ious, 10)), 3) if ious else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--apply", default="")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    res = {"started": time.strftime("%Y-%m-%d %H:%M"), "egl": os.environ.get("__EGL_VENDOR_LIBRARY_FILENAMES", "default")}
    res["gt10"] = tour(0, out); print("10 Hz", res["gt10"], flush=True)
    res["gt2"] = tour(2, out); print("2 Hz", res["gt2"], flush=True)
    if res["gt10"]["tour_dir"] and res["gt2"]["tour_dir"]:
        res["labels"] = compare(res["gt10"]["tour_dir"], res["gt2"]["tour_dir"])
    r10, r2 = res["gt10"]["rtf"], res["gt2"]["rtf"]
    res["speedup"] = round(r2 / r10, 2) if r10 and r2 else None
    lab = res.get("labels", {})
    res["passes"] = bool(res["speedup"] and res["speedup"] >= 1.10 and (lab.get("same_label_rate") or 0) >= 0.9
                         and (lab.get("iou_median") or 0) >= 0.85)
    if res["passes"] and a.apply:
        with open(a.apply, "a") as f:
            f.write(f"export DOORSIM_GT_HZ=2   # ground-truth cameras at 2 Hz (gt_rate_check {res['started']}: x{res['speedup']} faster)\n")
    (out / "result.json").write_text(json.dumps(res, indent=1) + "\n")
    rep = [f"# Ground-truth camera rate check {res['started']} (renderer: {res['egl']})", "",
           "| | 10 Hz | 2 Hz |", "|---|---|---|",
           f"| Gazebo speed (sim s per real s) | {r10} | {r2} |",
           f"| tour minutes (3 doors) | {res['gt10']['minutes']} | {res['gt2']['minutes']} |", "",
           f"Labels at shared stops: {lab}", f"Speedup x{res['speedup']}; passes: **{res['passes']}**"
           + (f" -> DOORSIM_GT_HZ=2 added to {a.apply}" if res["passes"] and a.apply else "")]
    (out / "report.md").write_text("\n".join(rep) + "\n")
    print("\n".join(rep))


if __name__ == "__main__":
    main()
