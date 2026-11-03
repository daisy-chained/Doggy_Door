"""Dataset generator: random scenes (door open amounts, lights) x random viewpoints, saved frame by frame.

usage (system Python with ROS sourced, no venv active):
  source /opt/ros/jazzy/setup.bash
  python3 sim/doorsim/generate.py sim/configs/door_room_v1_small.yaml [--out DIR] [--seed N]
  --split test [--export-seed N]   render only the scenes the exporter would put in that split
  --light-level L [--headlight H]  every scene at one fixed light level (v3 worlds), headlight 0 unless given

Many worlds (door room v4): a config with `worlds: door_room/v4/worlds.json` (made by sim/door_room/make_v4.py) renders
`scenes_per_world` scenes in each world, one Gazebo run per world (gz_<world>.log), scene numbers global and consecutive per
world. The split comes from the worlds (--split uses it; dataset.json "split_scenes" always has it), `max_worlds` keeps only the
first N worlds. Each frame's meta gets "world" and "side" (camera in front of / behind the wall: the door properties in the meta
are what that side sees). dataset.json adds "worlds" (split, held-out flag, door specs per world) and "world_files".

Output (one folder per dataset):
  dataset.json                 config, seed, world, git commit, camera intrinsics, attributes, counts
  frames/000000/rgb.png        colour image
  frames/000000/depth.png      sensor-like depth, uint16 mm, 0 = no value (glass handled per glass_depth)
  frames/000000/depth_gt.png   rendered depth (glass solid), same format
  frames/000000/seg.png        object id per pixel (uint8, 0 = background; ids in dataset.json attributes)
  frames/000000/meta.json      frame metadata (sim/doorsim/frames.py) + camera pose, scene, glass depth mode;
                               with clutter: visible_fraction per door / handle and the placed clutter items
"""
import argparse, datetime, json, pathlib, subprocess, sys, time, types
import numpy as np
import yaml
from PIL import Image

SIM = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIM))
from doorsim import clutter, doors, export, frames, glass, sampling  # gzsim (needs the gz python module) is imported in main()


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


def scenes_to_render(n_scenes, split=None, export_seed=0, ratios=(0.7, 0.15, 0.15)):
    """Scene numbers to render: all, or only those the exporter puts in `split`."""
    if split is None:
        return list(range(n_scenes))
    return sorted(export.split_scenes(range(n_scenes), ratios, export_seed)[split])


def world_scenes(worlds, scenes_per_world, split=None):
    """[(world, [scene numbers])]: world i gets scenes i * spw .. (i + 1) * spw - 1; only the worlds of `split` if given."""
    plan = [(w, list(range(i * scenes_per_world, (i + 1) * scenes_per_world))) for i, w in enumerate(worlds)]
    return [(w, s) for w, s in plan if split is None or w["split"] == split]


def world_split_scenes(worlds, scenes_per_world):
    """{split: scene numbers}: a world's scenes all go to the world's split (a door never leaks into test)."""
    return {sp: [s for _, scenes in world_scenes(worlds, scenes_per_world, sp) for s in scenes] for sp in export.SPLITS}


def render_scenes(sim, attrs, scenes, run, world=None):
    """Render `scenes` in a running world. run = shared state: cfg, lcfg, out, rng, count, rejected, t_start, scenes_total, scenes_done.
    world = the world dict for many-worlds datasets (frames then get "world" / "side"), None for a single-world config."""
    from doorsim import gzsim
    cfg, lcfg, out = run.cfg, run.lcfg, run.out
    rej, gcfg, ccfg = cfg["reject"], cfg["glass_depth"], cfg.get("clutter")
    glass_ids = {o["id"]: n for n, o in attrs["objects"].items() if o.get("part") == "glass_pane"}
    clutter_ids = {p["name"]: p["id"] for p in clutter.POOL}
    keys = ["rgb", "depth", "seg", "boxes", "boxes3d"]
    keep_out = doors.keep_out(attrs["objects"])
    rng = run.rng
    for scene in scenes:
        if cfg.get("per_scene_rng"):
            rng, light_rng = sampling.scene_rngs(cfg["seed"], scene)
        else:
            light_rng = rng
        targets = sampling.door_targets(rng, attrs, cfg["doors"])
        lit = sampling.lighting(light_rng, lcfg) if lcfg else None
        level = lit["level"] if lit else 1.0
        base = sampling.lights(light_rng, cfg["lights"])  # at level 1; per view the level only scales these
        at = lambda lv: {n: {"diffuse": [c * lv for c in l["diffuse"]]} for n, l in base.items()}
        light = at(level)
        for name, l in light.items():
            sim.set_light(name, cfg["lights"][name], l["diffuse"])
        if lit:
            h = lit["headlight"]
            sim.set_light("headlight", lcfg["headlight"], [h, h, 0.95 * h])
        placed = None
        if ccfg:
            placed = clutter.place(rng, clutter.POOL, ccfg, keep_out)
            sim.place_clutter(placed)
        joints = sim.set_doors(targets, max_steps=cfg.get("settle_steps", 6000))
        view_level = level
        for v in range(cfg["views_per_scene"]):
            if lit:
                lv = sampling.flicker(light_rng, lcfg, level)
                if lv != view_level:  # room lights dip for this view (and come back)
                    view_level = lv
                    for name, l in at(lv).items():
                        sim.set_light(name, cfg["lights"][name], l["diffuse"])
            for _ in range(rej["max_tries"]):
                cam = sampling.view(rng, attrs, cfg["view"])
                if placed and clutter.camera_blocked(cam["world_position"], placed, clutter.POOL, ccfg["camera_margin"]):
                    run.rejected += 1
                    continue
                sim.set_pose("rig", cam["world_position"], cam["world_rpy"])
                m = sim.render(keys)
                depth_gt = gzsim.image_array(m["depth"], "depth")
                b2 = frames.boxes_from_msg(m["boxes"], "2d")
                target_id = attrs["objects"][cam["target_door"]]["id"]
                too_close = np.nanmin(np.where(np.isfinite(depth_gt), depth_gt, np.nan)) < rej["min_depth"]
                if too_close or (rej["need_target_box"] and target_id not in {b["label"] for b in b2}):
                    run.rejected += 1
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
            extra = {"scene": scene, "view": v, "target_door": cam["target_door"],
                     "sim_time_ns": gzsim.ns(m["rgb"].header.stamp), "lights": light, "glass_depth_mode": mode}
            if world:
                extra.update(world=world["name"], side=cam["side"])
            if lit:
                extra["lighting"] = {"mode": lit["mode"], "level": level, "view_level": view_level, "headlight": lit["headlight"]}
            vis = None
            if placed:
                in_scene = {n: p for n, p in placed.items() if p["in_scene"]}
                seg_clear = seg
                if np.isin(seg, [clutter_ids[n] for n in in_scene]).any():
                    seg_clear = sim.seg_without_clutter(placed)
                vis = clutter.visible_fraction(seg, seg_clear, attrs)
                extra["clutter"] = in_scene
            meta = frames.build_frame(attrs, b2, frames.boxes_from_msg(m["boxes3d"], "3d"), joints,
                                      {"world_position": cam["world_position"], "world_rpy": cam["world_rpy"]},
                                      extra)
            if world:
                doors.apply_side(meta["objects"], cam["side"])  # push / pull, hinge side, slide direction as this side sees them
            for n, f in (vis or {}).items():
                meta["objects"][n]["visible_fraction"] = f
            save_frame(out / "frames" / f"{run.count:06d}", gzsim.image_array(m["rgb"], "rgb"), depth, depth_gt, seg, meta)
            run.count += 1
        run.scenes_done += 1
        print(f"scene {scene} ({run.scenes_done}/{run.scenes_total}): {run.count} frames, {run.rejected} viewpoints rejected, {time.time() - run.t_start:.0f} s", flush=True)
    run.rng = rng


def main():
    from doorsim import gzsim
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--out", help="output folder (default: datasets/<config name> in the repo)")
    ap.add_argument("--seed", type=int, help="overrides the config seed")
    ap.add_argument("--split", choices=export.SPLITS, help="render only the scenes of this split (as export.py would split them; many-worlds datasets: by world)")
    ap.add_argument("--export-seed", type=int, default=0, help="seed of that split (export.py --seed)")
    ap.add_argument("--light-level", type=float, help="fixed room light level for every scene (needs a lighting section)")
    ap.add_argument("--headlight", type=float, help="fixed headlight brightness (default 0 with --light-level)")
    args = ap.parse_args()
    cfg_path = pathlib.Path(args.config)
    cfg = yaml.safe_load(cfg_path.read_text())
    if args.seed is not None:
        cfg["seed"] = args.seed
    lcfg = cfg.get("lighting")
    if args.light_level is not None:
        if not lcfg:
            sys.exit("--light-level needs a lighting section in the config")
        lcfg["fixed"] = {"level": args.light_level, "headlight": args.headlight or 0.0}
    elif args.headlight is not None:
        sys.exit("--headlight goes with --light-level")
    world = cfg["world"]
    room = SIM / "door_room"
    out = pathlib.Path(args.out) if args.out else SIM.parent / "datasets" / cfg_path.stem
    if (out / "dataset.json").exists():
        sys.exit(f"{out} already has a dataset; pick another --out")
    out.mkdir(parents=True, exist_ok=True)
    info = {"world": world, "config": cfg, "seed": cfg["seed"], "git": git_state(),
            "created": datetime.datetime.now().isoformat(timespec="seconds")}
    if "worlds" in cfg:  # many worlds: [(world dict, its sdf, its attributes, its scenes)]
        worlds = json.loads((SIM / cfg["worlds"]).read_text())["worlds"][:cfg.get("max_worlds")]
        spw = cfg["scenes_per_world"]
        plan = [(w, SIM / w["sdf"], json.loads((SIM / w["attributes"]).read_text()), scenes) for w, scenes in world_scenes(worlds, spw, args.split)]
        first = json.loads((SIM / worlds[0]["attributes"]).read_text())
        info["attributes"] = {"classes": first["classes"], "objects": first["objects"]}  # the first world's, for readers of single-world datasets
        info["worlds"] = {w["name"]: {"split": w["split"], "heldout": w["heldout"], "doors": w["doors"]} for w, *_ in plan}
        info["world_files"] = {w["name"]: w["sdf"] for w, *_ in plan}
        info["split_scenes"] = world_split_scenes(worlds, spw)
    else:
        attrs = json.loads((room / f"{world}.attributes.json").read_text())
        info["attributes"] = attrs
        plan = [(None, room / f"{world}.sdf", attrs, scenes_to_render(cfg["scenes"], args.split, args.export_seed))]
        if args.split:
            info["split_scenes"] = export.split_scenes(range(cfg["scenes"]), (0.7, 0.15, 0.15), args.export_seed)
    if lcfg:
        fixed = lcfg.get("fixed")
        info["lighting"] = {"mode": "fixed", "level": fixed["level"], "headlight": fixed["headlight"]} if fixed else {"mode": "random"}
    run = types.SimpleNamespace(cfg=cfg, lcfg=lcfg, out=out, rng=np.random.default_rng(cfg["seed"]), count=0, rejected=0, t_start=time.time(),
                                scenes_total=sum(len(s) for *_, s in plan), scenes_done=0)
    try:
        for w, sdf, attrs, scenes in plan:  # one Gazebo run per world, stopped before the next starts
            name = w["name"] if w else world
            sim = gzsim.Sim(sdf, name, attrs, out / (f"gz_{name}.log" if w else "gz.log"))
            try:
                info.setdefault("camera_intrinsics", sim.camera_info())
                render_scenes(sim, attrs, scenes, run, w)
            finally:
                sim.stop()
    finally:
        info.update({"frames": run.count, "rejected_viewpoints": run.rejected, "seconds": round(time.time() - run.t_start)})
        (out / "dataset.json").write_text(json.dumps(info, indent=2) + "\n")
    print(f"done: {run.count} frames in {out}")


if __name__ == "__main__":
    main()
