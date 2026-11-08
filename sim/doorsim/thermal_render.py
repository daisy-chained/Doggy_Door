"""Thermal replay: a thermal image (320 x 240, 16 bit, 0.01 K) for every frame of an existing dataset, in two scenarios. The Gazebo
thermal camera is only 8 bit (2.6 K steps), so the image is a label render (seg) + a per-model temperature lookup, with T^4 area
averaging down to 320 x 240 (thermal.kelvin_from_seg).

usage (system Python with ROS sourced):
  source /opt/ros/jazzy/setup.bash
  python3 sim/doorsim/thermal_render.py datasets/door_room_v4_2k --workers 4 [--limit-worlds K] [--frames-limit N] [--seed N] [--behind]

For every world and scenario (room, fire) temperatures are drawn (seeded per world + scenario), saved to
DATASET/thermal/<world>_<scenario>.json, injected into the world file, and every recorded scene is replayed (same clutter, door
positions, camera poses as meta.json). Output: frames/<frame>/thermal_<scenario>.png (uint16, 0.01 K; thermal.from_png16),
thermal/camera.json (intrinsics). --behind: for frames with glass panes in view also thermal_<scenario>_behind.png, the same image with those panes moved away (for
thermal_fx --glass: what leaks through). Frames that already have their file(s) are skipped, so a run can be resumed. Workers: one Gazebo run
each, own GZ_PARTITION (same scheme as generate.py)."""
import argparse, json, math, multiprocessing, os, pathlib, signal, sys, tempfile, time
import numpy as np
from PIL import Image

SIM = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIM))
from doorsim import effects, thermal

SCENARIOS = ("room", "fire")


def frame_file(ds_dir, frame, scenario):
    return ds_dir / "frames" / frame / f"thermal_{scenario}.png"


def panes_in_view(meta):
    """Sorted ["glass_x", ...] of the panes whose door "door_x" has glass and is (or may be) visible: visible_fraction None or > 0."""
    return sorted("glass_" + n.split("_", 1)[1] for n, o in meta["objects"].items()
                  if n.startswith("door_") and o.get("glass") and (o.get("visible_fraction") is None or o["visible_fraction"] > 0)
                  and "glass_" + n.split("_", 1)[1] in meta["objects"])


def behind_file(ds_dir, frame, scenario):
    return ds_dir / "frames" / frame / f"thermal_{scenario}_behind.png"


def world_temps(ds_dir, name, index, scenario, models, seed):
    """Temperatures of one world + scenario (saved in thermal/, reused when present so a resumed run is consistent)."""
    f = ds_dir / "thermal" / f"{name}_{scenario}.json"
    if f.exists():
        return json.loads(f.read_text())
    t = thermal.temperatures(np.random.default_rng([seed, index, SCENARIOS.index(scenario)]), scenario, models)
    f.parent.mkdir(exist_ok=True)
    f.write_text(json.dumps(t, indent=2) + "\n")
    return t


def camera_info(sdf_text):
    """Pinhole intrinsics of the thermal camera from its SDF field of view."""
    import xml.etree.ElementTree as ET
    cam = ET.fromstring(sdf_text.encode()).find(".//sensor[@type='thermal']/camera")
    w, h = int(cam.find("image/width").text), int(cam.find("image/height").text)
    f = (w / 2) / math.tan(float(cam.find("horizontal_fov").text) / 2)
    return {"width": w, "height": h, "K": [f, 0.0, w / 2, 0.0, f, h / 2, 0.0, 0.0, 1.0]}


def render_world(job):
    """Worker: all scenarios of one world, one Gazebo run per scenario. Returns the number of frames written."""
    partition, ds_dir, ds, name, index, plan, seed, frames_limit, check, behind = job
    os.environ["GZ_PARTITION"] = partition  # before gz.transport is imported
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(1))
    from doorsim import gzsim
    sdf = effects.world_file(ds, {"world": name})
    attrs = json.loads(sdf.with_suffix(".attributes.json").read_text())
    cam_info = json.loads((ds_dir / "thermal" / "camera.json").read_text())
    written, t0, agree = 0, time.time(), []

    def frame_meta(f):
        return json.loads((ds_dir / "frames" / f / "meta.json").read_text())
    for scenario in SCENARIOS:
        def todo(f):
            if not frame_file(ds_dir, f, scenario).exists():
                return True
            return behind and bool(panes_in_view(frame_meta(f))) and not behind_file(ds_dir, f, scenario).exists()
        views = [(p, [(f, c) for f, c in p["views"] if todo(f)][:frames_limit]) for p in plan]
        views = [(p, v) for p, v in views if v]
        if not views:
            continue
        temps = world_temps(ds_dir, name, index, scenario, list(attrs["objects"]) + ["floor", "back_wall", "room_shell", "rig"], seed)
        with tempfile.TemporaryDirectory(prefix="thermal_") as tmp:
            path = pathlib.Path(tmp) / sdf.name
            path.write_text(thermal.inject(sdf.read_text(), {}, thermal=False))  # labels only
            by_id = thermal.temps_by_id(temps, attrs)
            sim = gzsim.Sim(path, name, attrs, ds_dir / "thermal" / f"gz_{name}_{scenario}.log")
            try:
                for p, vs in views:
                    sim.place_clutter(p["clutter"])
                    sim.set_doors(p["doors"])
                    for frame, cam in vs:
                        sim.set_pose("rig", cam["world_position"], cam["world_rpy"])
                        size = (cam_info["width"], cam_info["height"])
                        panes = panes_in_view(frame_meta(frame)) if behind else []
                        if panes and not behind_file(ds_dir, frame, scenario).exists():
                            seg_b = sim.seg_without_panes(panes)
                            k = thermal.kelvin_from_seg(seg_b, by_id, temps["air"], size)
                            Image.fromarray(thermal.to_png16(k)).save(behind_file(ds_dir, frame, scenario))
                        if frame_file(ds_dir, frame, scenario).exists():
                            written += 1
                            continue
                        seg = gzsim.image_array(sim.render(["seg"])["seg"], "seg")
                        if check:
                            ref = np.array(Image.open(ds_dir / "frames" / frame / "seg.png"))
                            lab = ref > 0
                            agree.append(((seg == ref)[lab].mean(), np.isin(seg[~lab], [0, *thermal.STRUCT_IDS.values()]).mean()))
                        k = thermal.kelvin_from_seg(seg, by_id, temps["air"], size)
                        Image.fromarray(thermal.to_png16(k)).save(frame_file(ds_dir, frame, scenario))
                        written += 1
                    print(f"[{name}/{scenario}] scene {p['scene']}: {written} frames, {time.time() - t0:.0f} s", flush=True)
            finally:
                sim.stop()
    if agree:
        a = np.array(agree)
        print(f"[{name}] seg agreement with the dataset's seg.png: labelled pixels {100 * a[:, 0].mean():.2f} % (min {100 * a[:, 0].min():.2f} %), "
              f"background = 0 or structure {100 * a[:, 1].mean():.2f} %", flush=True)
    return written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--limit-worlds", type=int)
    ap.add_argument("--frames-limit", type=int, help="at most N new frames per scene and scenario (for quick checks)")
    ap.add_argument("--check-seg", action="store_true", help="compare each replayed seg with the dataset's seg.png and print the agreement")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--behind", action="store_true", help="also thermal_<scenario>_behind.png (panes moved away) for frames with panes in view")
    a = ap.parse_args()
    ds_dir = pathlib.Path(a.dataset).resolve()
    ds = json.loads((ds_dir / "dataset.json").read_text())
    metas = {d.name: json.loads((d / "meta.json").read_text()) for d in sorted((ds_dir / "frames").iterdir())}
    plan = thermal.replay_plan(metas)
    names = list(ds["worlds"])[:a.limit_worlds]
    (ds_dir / "thermal").mkdir(exist_ok=True)
    first = effects.world_file(ds, {"world": names[0]})
    (ds_dir / "thermal" / "camera.json").write_text(json.dumps(camera_info(thermal.inject(first.read_text(), {})), indent=2) + "\n")
    jobs = [(f"doorsim_{os.getpid()}_{i}", ds_dir, ds, n, list(ds["worlds"]).index(n),
             [p for p in plan if p["world"] == n], a.seed, a.frames_limit, a.check_seg, a.behind) for i, n in enumerate(names)]
    jobs = [j for j in jobs if j[5]]
    t = time.time()
    pool = multiprocessing.get_context("spawn").Pool(max(1, min(a.workers, len(jobs))), maxtasksperchild=1)  # fresh process per world: own gz-transport partition
    try:
        total = sum(pool.imap_unordered(render_world, jobs))
        pool.close()
    except BaseException:
        pool.terminate(); raise
    finally:
        pool.join()
    print(f"{total} frames in {time.time() - t:.0f} s")


if __name__ == "__main__":
    main()
