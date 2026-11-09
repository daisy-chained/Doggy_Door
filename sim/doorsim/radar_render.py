"""Radar replay: wide depth + label renders (480 x 100, 120 x ~40 deg, radar.inject) for every frame of an existing dataset.
radar.simulate turns them into radar returns later (the render is the expensive part, the radar model can change freely).

usage (system Python with ROS sourced):
  source /opt/ros/jazzy/setup.bash
  python3 sim/doorsim/radar_render.py datasets/door_room_v4_2k --workers 4 [--limit-worlds K] [--frames-limit N]

Every recorded scene is replayed (same clutter, door positions, camera poses as meta.json). Output per frame in
frames/<frame>/: radar_depth.png (uint16 mm, glass.to_png16), radar_seg.png (uint16 labels), radar_depth_noglass.png (depth with
the glass panes moved away; equal to radar_depth.png where the world has no panes), radar_behind.png (what is behind the back
wall / doors / glass: the hidden-wall gpu_lidar "radar_behind" as planar depth on the same pixel grid, radar.behind_depth; 0 =
no return); radar/camera.json (width, height, K, hfov, behind_lidar).
A frame is done when radar_behind.png exists (frames rendered before see-through are re-rendered), so a run can be resumed (retries are up to the caller). Workers: one Gazebo
run each, own GZ_PARTITION (same scheme as thermal_render.py)."""
import argparse, json, math, multiprocessing, os, pathlib, signal, sys, tempfile, time
import numpy as np
from PIL import Image

SIM = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIM))
from doorsim import effects, glass, radar, thermal
from doorsim.generate import start_sim

FILES = ("radar_depth.png", "radar_seg.png", "radar_depth_noglass.png", "radar_behind.png")  # last = the done marker


def frame_file(ds_dir, frame, name=FILES[-1]):
    return ds_dir / "frames" / frame / name


def camera_info():
    w, h = radar.WIDE_SIZE
    return {"width": w, "height": h, "K": radar.camera_K(w, h, radar.WIDE_HFOV), "hfov": radar.WIDE_HFOV,
            "behind_lidar": radar.behind_lidar_geometry(w, h, radar.WIDE_HFOV)}


def make_sim(gzsim):
    """gzsim.Sim that also listens to the wide radar topics (depth, labels, the hidden-wall lidar) (SENSORS is read in Sim.__init__) and can render the wide
    depth without the glass panes (a copy of Sim.depth_without_panes for the "radar_depth" key)."""
    gzsim.SENSORS.setdefault("radar_depth", ("/radar/depth", gzsim.Image))
    gzsim.SENSORS.setdefault("radar_seg", ("/radar/seg/labels_map", gzsim.Image))
    from gz.msgs10.laserscan_pb2 import LaserScan  # lazily: only importable with ROS sourced, like gzsim
    gzsim.SENSORS.setdefault("radar_behind", ("/radar/behind", LaserScan))

    class RadarSim(gzsim.Sim):
        def radar_depth_without_panes(self, panes):
            saved = {n: self.model_pose(n) for n in panes}
            for n in panes:
                if n in self.pane_pub:
                    self.pane_pub[n][0].publish(gzsim.Empty())
            self.step(2)
            for n in panes:
                self.set_pose(n, [0, 0, -50])
            depth = gzsim.image_array(self.render(["radar_depth"])["radar_depth"], "depth")
            for n, (pos, q) in saved.items():
                self.set_pose(n, pos, orientation=q)
            self.step(2)
            for n in panes:
                if n in self.pane_pub:
                    self.pane_pub[n][1].publish(gzsim.Empty())
            self.step(2)
            return depth
    return RadarSim


def render_world(job):
    """Worker: all scenes of one world. Returns the number of frames written."""
    partition, ds_dir, ds, name, plan, frames_limit = job
    os.environ["GZ_PARTITION"] = partition  # before gz.transport is imported
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(1))
    from doorsim import gzsim
    sdf = effects.world_file(ds, {"world": name})
    attrs = json.loads(sdf.with_suffix(".attributes.json").read_text())
    views = [(p, [(f, c) for f, c in p["views"] if not frame_file(ds_dir, f).exists()][:frames_limit]) for p in plan]
    views = [(p, v) for p, v in views if v]
    if not views:
        return 0
    written, t0 = 0, time.time()
    with tempfile.TemporaryDirectory(prefix="radar_") as tmp:
        path = pathlib.Path(tmp) / sdf.name
        path.write_text(radar.inject(sdf.read_text()))
        sim = start_sim(lambda: make_sim(gzsim)(path, name, attrs, ds_dir / "radar" / f"gz_{name}.log"), name)  # retries a slow start (2026-11-07)
        try:
            for p, vs in views:
                sim.place_clutter(p["clutter"])
                sim.set_doors(p["doors"])
                for frame, cam in vs:
                    sim.set_pose("rig", cam["world_position"], cam["world_rpy"])
                    m = sim.render(["radar_depth", "radar_seg", "radar_behind"])
                    depth = gzsim.image_array(m["radar_depth"], "depth")
                    seg = gzsim.image_array(m["radar_seg"], "seg")
                    w_px, h_px = radar.WIDE_SIZE
                    geom = radar.behind_lidar_geometry(w_px, h_px, radar.WIDE_HFOV)
                    behind = radar.behind_depth(np.array(m["radar_behind"].ranges).reshape(geom["v_samples"], geom["h_samples"]),
                                                geom, radar.camera_K(w_px, h_px, radar.WIDE_HFOV), (h_px, w_px))
                    noglass = sim.radar_depth_without_panes(list(sim.panes)) if sim.panes else depth
                    out = ds_dir / "frames" / frame
                    Image.fromarray(glass.to_png16(depth)).save(out / FILES[0])
                    Image.fromarray(seg.astype(np.uint16)).save(out / FILES[1])
                    Image.fromarray(glass.to_png16(noglass)).save(out / FILES[2])
                    Image.fromarray(glass.to_png16(behind)).save(out / FILES[3])  # last: marks the frame as done
                    written += 1
                    print(f"[{name}] frame {frame}: {written} frames, {(time.time() - t0) / written:.2f} s/frame", flush=True)
        finally:
            sim.stop()
    return written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--limit-worlds", type=int)
    ap.add_argument("--frames-limit", type=int, help="at most N new frames per scene (for quick checks)")
    a = ap.parse_args()
    ds_dir = pathlib.Path(a.dataset).resolve()
    ds = json.loads((ds_dir / "dataset.json").read_text())
    metas = {d.name: json.loads((d / "meta.json").read_text()) for d in sorted((ds_dir / "frames").iterdir())}
    plan = thermal.replay_plan(metas)
    names = list(ds["worlds"])[:a.limit_worlds]
    (ds_dir / "radar").mkdir(exist_ok=True)
    (ds_dir / "radar" / "camera.json").write_text(json.dumps(camera_info(), indent=2) + "\n")
    jobs = [(f"doorsim_{os.getpid()}_{i}", ds_dir, ds, n, [p for p in plan if p["world"] == n], a.frames_limit)
            for i, n in enumerate(names)]
    jobs = [j for j in jobs if j[4]]
    t = time.time()
    pool = multiprocessing.get_context("spawn").Pool(max(1, min(a.workers, len(jobs))), maxtasksperchild=1)
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
