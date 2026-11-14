"""Which GPU does Gazebo render on, and what does it cost in RAM? Default (what the tours use now) vs forced NVIDIA EGL.

    source /opt/ros/jazzy/setup.bash && python3 sim/smoke_test/gpu_ram_check.py --out runs/scratch/gpu_check

Why (2026-11-14): during the dark sweep the tour's Gazebo server held ~2.8 GB of Intel iGPU (i915) buffers in system RAM while
the RTX 4060 was nearly idle (on 2026-10-29 rendering still went to the 4060 by default). Meant to run alone in the queue.

Per mode (default, nvidia = __EGL_VENDOR_LIBRARY_FILENAMES=.../10_nvidia.json, the setting from sim/smoke_test/README.md):
  1. static: rgbd_smoke.sdf (one fixed RGB-D camera), grab RGB frames for 10 s -> frame rate + the last frame
     (both modes see the exact same scene, so the pixel difference says whether the renderer changes the images).
  2. tour: the real thing, short: tour_test.sh on bld_train_005 with the dark-2 % conditions + headlight (same as cond_005_dark2),
     wheeled_ring platform (5 cameras), --max-visits 1. Every 5 s: the gz server's i915 resident memory (drm fdinfo), NVIDIA
     memory used, MemAvailable. Peaks go in the result.
Output: OUT/result.json, OUT/report.md, OUT/rgb_<mode>.png; when NVIDIA helps (i915 near zero, 4060 takes the drawing, tour
reached its stop, images close) also OUT/gpu_env.sh with the export line, and with --apply PATH the same line at PATH: the queued tour jobs source runs/scratch/gpu_env.sh at their start
(Madelief 2026-11-14: "if it helps alter the queued runs"). Delete that file to go back to the default."""
import argparse, json, os, re, signal, subprocess, sys, time
from pathlib import Path

NV = "/usr/share/glvnd/egl_vendor.d/10_nvidia.json"
REPO = Path(__file__).resolve().parents[2]


def env_for(mode, part):
    e = dict(os.environ, GZ_PARTITION=part)
    if mode == "nvidia":
        e["__EGL_VENDOR_LIBRARY_FILENAMES"] = NV
    else:
        e.pop("__EGL_VENDOR_LIBRARY_FILENAMES", None)
    return e


def i915_bytes(pid):
    """Resident i915 (Intel iGPU) memory of one process, from its DRM fdinfo, each client counted once."""
    tot, seen = 0, set()
    try:
        fds = os.listdir(f"/proc/{pid}/fdinfo")
    except OSError:
        return 0
    for fd in fds:
        try:
            s = open(f"/proc/{pid}/fdinfo/{fd}").read()
        except OSError:
            continue
        m = re.search(r"drm-client-id:\s*(\d+)", s)
        if not m or not re.search(r"drm-driver:\s*i915", s) or m.group(1) in seen:
            continue
        seen.add(m.group(1))
        for v, u in re.findall(r"drm-resident-system0:\s*(\d+)\s*(KiB|MiB|GiB)?", s):
            tot += int(v) * {"KiB": 1024, "MiB": 2**20, "GiB": 2**30}.get(u, 1)
    return tot


def nvidia_mb():
    try:
        return int(subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                  capture_output=True, text=True, timeout=10).stdout.split()[0])
    except Exception:
        return -1


def avail_gb():
    for l in open("/proc/meminfo"):
        if l.startswith("MemAvailable"):
            return int(l.split()[1]) / 1e6


def gz_servers(part):
    """PIDs of gz sim servers (ruby) running in this GZ_PARTITION."""
    out = []
    for p in os.listdir("/proc"):
        if not p.isdigit():
            continue
        try:
            if open(f"/proc/{p}/comm").read().strip() != "ruby":
                continue
            env = open(f"/proc/{p}/environ", "rb").read().split(b"\0")
        except OSError:
            continue
        if f"GZ_PARTITION={part}".encode() in env:
            out.append(int(p))
    return out


def uses_nvidia_egl(pid):
    try:
        return "libnvidia-eglcore" in open(f"/proc/{pid}/maps").read()
    except OSError:
        return None


def static(mode, out):
    """Fixed camera, 10 s of frames. Runs in a child process so gz-transport picks up this mode's partition."""
    part = f"gpuchk_{mode}_{os.getpid()}"
    nv0 = nvidia_mb()
    p = subprocess.Popen(["gz", "sim", "-s", "-r", "--headless-rendering", str(REPO / "sim/smoke_test/rgbd_smoke.sdf")],
                         env=env_for(mode, part), stdout=open(out / f"static_{mode}.log", "w"), stderr=subprocess.STDOUT,
                         start_new_session=True)
    try:
        child = subprocess.run([sys.executable, __file__, "--grab", str(out / f"rgb_{mode}.npy"), "--out", str(out)],
                               env=env_for(mode, part), capture_output=True, text=True, timeout=120)
        res = json.loads(child.stdout.strip().splitlines()[-1]) if child.stdout.strip() else {"error": child.stderr[-500:]}
        srv = gz_servers(part)
        res.update(i915_gb=round(sum(i915_bytes(s) for s in srv) / 1e9, 3), nvidia_mb_delta=nvidia_mb() - nv0,
                   nvidia_egl_loaded=any(uses_nvidia_egl(s) for s in srv))
        return res
    finally:
        os.killpg(p.pid, signal.SIGINT); time.sleep(3)
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def grab(npy):
    """Child: count /rgbd/image messages for 10 s after the first one, keep the last frame."""
    import numpy as np
    from gz.transport13 import Node
    from gz.msgs10.image_pb2 import Image
    last, n = {}, [0]
    def cb(m):
        last["m"] = m; n[0] += 1
    node = Node(); node.subscribe(Image, "/rgbd/image", cb)
    t0 = time.time()
    while not last and time.time() - t0 < 90:
        time.sleep(0.2)
    if not last:
        print(json.dumps({"error": "no image in 90 s"})); return
    n[0] = 0; time.sleep(10)
    m = last["m"]
    a = np.frombuffer(m.data, np.uint8).reshape(m.height, m.width, -1)[:, :, :3]
    np.save(npy, a)
    print(json.dumps({"fps": round(n[0] / 10, 1), "shape": list(a.shape)}))


def tour(mode, out, limit=900):
    nv0, a0 = nvidia_mb(), avail_gb()
    env = env_for(mode, "unused")
    env.update(SLAM="0", ODOM_NOISE="0", LIGHT="0.02", HEADLIGHT="0.7")
    p = subprocess.Popen(["bash", "sim/building/tour_test.sh", "sim/building/out/bld_train_005", "sim/configs/robots/wheeled_ring.yaml",
                          "1.0", "4.915", "--max-visits", "1"], cwd=REPO, env=env, stdout=open(out / f"tour_{mode}.log", "w"),
                         stderr=subprocess.STDOUT, start_new_session=True)
    part = f"tour_{p.pid}"                   # tour_test.sh sets GZ_PARTITION=tour_$$
    peak = {"i915_gb": 0.0, "nvidia_mb_delta": 0, "avail_min_gb": a0, "nvidia_egl_loaded": None}
    t0 = time.time()
    while p.poll() is None and time.time() - t0 < limit:
        time.sleep(5)
        srv = gz_servers(part)
        if srv:
            peak["i915_gb"] = max(peak["i915_gb"], round(sum(i915_bytes(s) for s in srv) / 1e9, 3))
            if peak["nvidia_egl_loaded"] is None:
                peak["nvidia_egl_loaded"] = any(uses_nvidia_egl(s) for s in srv)
        peak["nvidia_mb_delta"] = max(peak["nvidia_mb_delta"], nvidia_mb() - nv0)
        peak["avail_min_gb"] = round(min(peak["avail_min_gb"], avail_gb()), 2)
    if p.poll() is None:
        os.killpg(p.pid, signal.SIGINT); time.sleep(10)
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        peak["timed_out"] = True
    dirs = [d for d in (REPO / "runs/tours").glob("bld_train_005_wheeled_ring_l0.02_snone_h0.7_*") if d.stat().st_mtime >= t0 - 5]
    tdir = max(dirs, key=lambda d: d.stat().st_mtime) if dirs else None
    reached = None
    if tdir and (tdir / "tour.json").exists():
        t = json.loads((tdir / "tour.json").read_text())
        reached = sum(1 for v in t.get("visits", []) for s in v.get("stops", []) if s.get("reached"))
    peak.update(minutes=round((time.time() - t0) / 60, 1), exit=p.returncode, tour_dir=str(tdir) if tdir else None, stops_reached=reached,
                avail_start_gb=round(a0, 2))
    return peak


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--grab", default="")
    ap.add_argument("--apply", default="", help="when NVIDIA helps, also write the export line here (the queued tour jobs source it)")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if a.grab:
        grab(a.grab); return
    import numpy as np
    from PIL import Image as PImage
    res = {"started": time.strftime("%Y-%m-%d %H:%M")}
    for mode in ("default", "nvidia"):
        res[mode] = {"static": static(mode, out)}
        print(mode, "static", res[mode]["static"], flush=True)
        res[mode]["tour"] = tour(mode, out)
        print(mode, "tour", res[mode]["tour"], flush=True)
        (out / "result.json").write_text(json.dumps(res, indent=1) + "\n")
    imgs = {}
    for mode in ("default", "nvidia"):
        f = out / f"rgb_{mode}.npy"
        if f.exists():
            imgs[mode] = np.load(f).astype(np.int16)
            PImage.fromarray(imgs[mode].astype(np.uint8)).save(out / f"rgb_{mode}.png")
    if len(imgs) == 2 and imgs["default"].shape == imgs["nvidia"].shape:
        d = np.abs(imgs["default"] - imgs["nvidia"])
        res["image_diff"] = {"mean": round(float(d.mean()), 2), "p99": float(np.percentile(d, 99)), "max": int(d.max()),
                             "pixels_over_10": round(float((d.max(axis=2) > 10).mean()), 4)}
    dn, nn = res["default"]["tour"], res["nvidia"]["tour"]
    helps = (nn.get("nvidia_egl_loaded") and nn["i915_gb"] < 0.3 and dn["i915_gb"] > 1.0 and (nn.get("stops_reached") or 0) >= 1
             and res.get("image_diff", {}).get("mean", 99) < 3.0)
    res["helps"] = bool(helps)
    res["ram_freed_gb"] = round(dn["i915_gb"] - nn["i915_gb"], 2)
    (out / "result.json").write_text(json.dumps(res, indent=1) + "\n")
    if helps:
        line = f"export __EGL_VENDOR_LIBRARY_FILENAMES={NV}   # Gazebo renders on the RTX 4060 (gpu_ram_check {res['started']})\n"
        (out / "gpu_env.sh").write_text(line)
        if a.apply:
            Path(a.apply).write_text(line)
    lines = [f"# GPU / RAM check {res['started']}", "",
             "| | default | nvidia |", "|---|---|---|"]
    for k, lab in (("i915_gb", "Intel iGPU RAM, tour peak (GB)"), ("nvidia_mb_delta", "4060 memory added, tour (MB)"),
                   ("nvidia_egl_loaded", "NVIDIA EGL in gz server"), ("avail_min_gb", "RAM available, lowest (GB)"),
                   ("stops_reached", "tour stops reached"), ("minutes", "tour minutes")):
        lines.append(f"| {lab} | {dn.get(k)} | {nn.get(k)} |")
    lines.append(f"| static camera fps | {res['default']['static'].get('fps')} | {res['nvidia']['static'].get('fps')} |")
    lines += ["", f"Image difference default vs nvidia (same fixed camera): {res.get('image_diff')}",
              f"Helps: **{res['helps']}** (RAM freed ~{res['ram_freed_gb']} GB)" + ((" -> gpu_env.sh written" + (f", applied to {a.apply}" if a.apply else "")) if helps else "")]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
