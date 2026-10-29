"""Dataset generator: random scenes (door open amounts, lights) x random viewpoints, saved frame by frame.

usage (system Python with ROS sourced, no venv active):
  source /opt/ros/jazzy/setup.bash
  python3 sim/doorsim/generate.py sim/configs/door_room_v1_small.yaml [--out DIR] [--seed N]

Output (one folder per dataset):
  dataset.json                 config, seed, world, git commit, camera intrinsics, attributes, counts
  frames/000000/rgb.png        colour image
  frames/000000/depth.png      sensor-like depth, uint16 mm, 0 = no value (glass handled per glass_depth)
  frames/000000/depth_gt.png   rendered depth (glass solid), same format
  frames/000000/seg.png        object id per pixel (uint8, 0 = background; ids in dataset.json attributes)
  frames/000000/meta.json      frame metadata (sim/doorsim/frames.py) + camera pose, scene, glass depth mode
"""
import argparse, datetime, json, pathlib, subprocess, sys, time
import numpy as np
import yaml
from PIL import Image

SIM = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIM))
from doorsim import frames, glass, gzsim, sampling


def git_state():
    def run(*a):
        return subprocess.run(["git", *a], cwd=SIM, capture_output=True, text=True).stdout.strip()
    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain"))}


def save_frame(d, rgb, depth, depth_gt, seg, meta):
    d.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(d / "rgb.png")
    Image.fromarray(glass.to_png16(depth)).save(d / "depth.png")
    Image.fromarray(glass.to_png16(depth_gt)).save(d / "depth_gt.png")
    Image.fromarray(seg).save(d / "seg.png")
    (d / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--out", help="output folder (default: datasets/<config name> in the repo)")
    ap.add_argument("--seed", type=int, help="overrides the config seed")
    args = ap.parse_args()
    cfg_path = pathlib.Path(args.config)
    cfg = yaml.safe_load(cfg_path.read_text())
    if args.seed is not None:
        cfg["seed"] = args.seed
    world = cfg["world"]
    room = SIM / "door_room"
    attrs = json.loads((room / f"{world}.attributes.json").read_text())
    out = pathlib.Path(args.out) if args.out else SIM.parent / "datasets" / cfg_path.stem
    if (out / "dataset.json").exists():
        sys.exit(f"{out} already has a dataset; pick another --out")
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg["seed"])
    rej, gcfg = cfg["reject"], cfg["glass_depth"]
    glass_ids = {o["id"]: n for n, o in attrs["objects"].items() if o.get("part") == "glass_pane"}
    keys = ["rgb", "depth", "seg", "boxes", "boxes3d"]
    info = {"world": world, "config": cfg, "seed": cfg["seed"], "git": git_state(),
            "created": datetime.datetime.now().isoformat(timespec="seconds"), "attributes": attrs}
    sim = gzsim.Sim(room / f"{world}.sdf", world, attrs, out / "gz.log")
    count, rejected, t_start = 0, 0, time.time()
    try:
        info["camera_intrinsics"] = sim.camera_info()
        for scene in range(cfg["scenes"]):
            targets = sampling.door_targets(rng, attrs, cfg["doors"])
            light = sampling.lights(rng, cfg["lights"])
            for name, l in light.items():
                sim.set_light(name, cfg["lights"][name], l["diffuse"])
            joints = sim.set_doors(targets, max_steps=cfg.get("settle_steps", 6000))
            for v in range(cfg["views_per_scene"]):
                for _ in range(rej["max_tries"]):
                    cam = sampling.view(rng, attrs, cfg["view"])
                    sim.set_pose("rig", cam["world_position"], cam["world_rpy"])
                    m = sim.render(keys)
                    depth_gt = gzsim.image_array(m["depth"], "depth")
                    b2 = frames.boxes_from_msg(m["boxes"], "2d")
                    target_id = attrs["objects"][cam["target_door"]]["id"]
                    too_close = np.nanmin(np.where(np.isfinite(depth_gt), depth_gt, np.nan)) < rej["min_depth"]
                    if too_close or (rej["need_target_box"] and target_id not in {b["label"] for b in b2}):
                        rejected += 1
                        continue
                    break
                else:
                    raise RuntimeError(f"scene {scene} view {v}: no usable viewpoint in {rej['max_tries']} tries")
                seg = gzsim.image_array(m["seg"], "seg")
                in_view = [n for i, n in glass_ids.items() if (seg == i).any()]
                mask = np.isin(seg, [attrs["objects"][n]["id"] for n in in_view])
                mode, behind = None, None
                if in_view:
                    mode = glass.pick_mode(rng, gcfg["mix"])
                    if mode != "holes":
                        behind = sim.depth_without_panes(in_view)
                depth = glass.apply(depth_gt, mask, mode or "holes", rng, behind, gcfg.get("partial"))
                meta = frames.build_frame(attrs, b2, frames.boxes_from_msg(m["boxes3d"], "3d"), joints,
                                          {"world_position": cam["world_position"], "world_rpy": cam["world_rpy"]},
                                          {"scene": scene, "view": v, "target_door": cam["target_door"],
                                           "sim_time_ns": gzsim.ns(m["rgb"].header.stamp), "lights": light,
                                           "glass_depth_mode": mode})
                save_frame(out / "frames" / f"{count:06d}", gzsim.image_array(m["rgb"], "rgb"), depth, depth_gt, seg, meta)
                count += 1
            print(f"scene {scene + 1}/{cfg['scenes']}: {count} frames, {rejected} viewpoints rejected, {time.time() - t_start:.0f} s", flush=True)
    finally:
        sim.stop()
        info.update({"frames": count, "rejected_viewpoints": rejected, "seconds": round(time.time() - t_start)})
        (out / "dataset.json").write_text(json.dumps(info, indent=2) + "\n")
    print(f"done: {count} frames in {out}")


if __name__ == "__main__":
    main()
