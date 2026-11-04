"""Thermal replay: render a thermal camera (320 x 240, 16 bit, 0.01 K) over an existing dataset, in two scenarios.

usage (system Python with ROS sourced):
  source /opt/ros/jazzy/setup.bash
  python3 sim/doorsim/thermal_render.py datasets/door_room_v4_2k --workers 4 [--limit-worlds K] [--frames-limit N] [--seed N]

For every world and scenario (room, fire) temperatures are drawn (seeded per world + scenario), saved to
DATASET/thermal/<world>_<scenario>.json, injected into the world file, and every recorded scene is replayed (same clutter, door
positions, camera poses as meta.json). Output: frames/<frame>/thermal_<scenario>.png (uint16, 0.01 K; thermal.from_png16),
thermal/camera.json (intrinsics). Frames that already have their file are skipped, so a run can be resumed. Workers: one Gazebo run
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
    partition, ds_dir, ds, name, index, plan, seed, frames_limit = job
    os.environ["GZ_PARTITION"] = partition  # before gz.transport is imported
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(1))
    from doorsim import gzsim
    from gz.msgs10.image_pb2 import Image as ImageMsg
    gzsim.SENSORS = {**gzsim.SENSORS, "thermal": ("/thermal", ImageMsg)}  # Sim subscribes to every entry in SENSORS
    sdf = effects.world_file(ds, {"world": name})
    attrs = json.loads(sdf.with_suffix(".attributes.json").read_text())
    written, t0 = 0, time.time()
    for scenario in SCENARIOS:
        views = [(p, [(f, c) for f, c in p["views"] if not frame_file(ds_dir, f, scenario).exists()][:frames_limit]) for p in plan]
        views = [(p, v) for p, v in views if v]
        if not views:
            continue
        temps = world_temps(ds_dir, name, index, scenario, list(attrs["objects"]) + ["floor", "back_wall", "room_shell", "rig"], seed)
        with tempfile.TemporaryDirectory(prefix="thermal_") as tmp:
            path = pathlib.Path(tmp) / sdf.name
            path.write_text(thermal.inject(sdf.read_text(), temps["models"]))
            sim = gzsim.Sim(path, name, attrs, ds_dir / "thermal" / f"gz_{name}_{scenario}.log")
            try:
                for p, vs in views:
                    sim.place_clutter(p["clutter"])
                    sim.set_doors(p["doors"])
                    for frame, cam in vs:
                        sim.set_pose("rig", cam["world_position"], cam["world_rpy"])
                        img = np.frombuffer(sim.render(["thermal"])["thermal"].data, np.uint16)
                        m = sim.latest["thermal"]
                        k = img.reshape(m.height, m.width).astype(float) * thermal.RES
                        Image.fromarray(thermal.to_png16(k)).save(frame_file(ds_dir, frame, scenario))
                        written += 1
                    print(f"[{name}/{scenario}] scene {p['scene']}: {written} frames, {time.time() - t0:.0f} s", flush=True)
            finally:
                sim.stop()
    return written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--limit-worlds", type=int)
    ap.add_argument("--frames-limit", type=int, help="at most N new frames per scene and scenario (for quick checks)")
    ap.add_argument("--seed", type=int, default=0)
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
             [p for p in plan if p["world"] == n], a.seed, a.frames_limit) for i, n in enumerate(names)]
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
