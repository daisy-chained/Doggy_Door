# Patrols in the dark and in smoke (brief gap after B3; Madelief 2026-11-13: "patrols in dark + smoke" first). Conditions design =
# Madelief 2026-11-07 round 2 (6): one light / smoke level per run, swept over the levels, for the numbers; mixed per room for the
# validation runs. Route (Madelief 2026-11-02): darkness in Gazebo (room lights x level + the robot's own headlight), smoke on the
# images (doorsim/effects.py, the same code as the training sets), camera noise on every frame. Details = Claude decisions
# (research-based, dev log 2026-11-14):
#   - light level = factor on every ceiling light of the building; the sun (fill light through the ceiling, as in the door room)
#     gets the lowest room level, so a dark room is never lit by it. Mixed: each space draws a mode normal / dim / dark (the
#     training mix, door_room_v9_6k.yaml `lighting.modes`) and a level in that mode's range.
#   - headlight = one spot light per camera (front + ring), pointing along the camera, same spot / attenuation as the training rig's
#     headlight (door_room_v9_6k.yaml `lighting.headlight`); 0 = no light elements at all (robot SDF as before).
#   - smoke = effects.sample_smoke with a fixed visibility (layer / patch / seed drawn from the run seed, so all levels of a sweep
#     share the same smoke shape, only V changes). Mixed: each space smoky with the training chance p, V log-uniform in the
#     training range, else clear; beta at a point = 3 / V of the space it is in (0 in a clear space).
#   - every saved frame: clean copies kept (rgb_clean.png, depth_clean.png), rgb.png / depth.png overwritten with smoke + depth
#     effects + camera noise, so the detector / door step see what a real camera would. The radar (Nav2, SLAM) is not touched:
#     real radar sees through smoke.
# No Gazebo / ROS needed: python3 -m pytest sim/tests/test_conditions.py
#
# Module sim/doorsim/conditions.py:
#   LIGHT_MODES = {"normal": {"p": 0.34, "level": [0.6, 1.2]}, "dim": {"p": 0.33, "level": [0.08, 0.4]},
#                  "dark": {"p": 0.33, "level": [0.003, 0.05]}}                      (= door_room_v9_6k.yaml)
#   HEADLIGHT = {"range": 10, "attenuation": [0.2, 0.05, 0.02], "spot": [0.4, 0.8, 1.0]}   (= the training rig)
#   scale_lights(sdf_text, levels, sun_level) -> new SDF text: diffuse rgb of light "ceiling_light_<space>" x levels[space]
#     (levels = float for all spaces, or {space: level}); light "sun" diffuse x sun_level; nothing else changes.
#   sample_rooms(rng, spaces, light, smoke, smoke_cfg) -> {space: {"mode", "light", "visibility_m"}}: light = float (every space
#     that level, mode "fixed") or "mixed"; smoke = None (all clear, visibility_m None), float (every space that V) or "mixed"
#     (per space: smoky with smoke_cfg["p"], V log-uniform smoke_cfg["visibility_m"], else None). Same rng state -> same result.
#   make(bjson, sdf_text, light, smoke, headlight, seed, effects_cfg) -> (world_sdf_text, cond): bjson = the building.json dict.
#     cond = {"light", "smoke", "headlight", "seed", "sun_level",
#             "spaces": {space: {"rect", "mode", "light", "visibility_m"}},
#             "smoke_shape": {"layer", "patch", "patch_scale", "seed"}   (from effects.sample_smoke(rng, ..., visibility=1.0))}
#     JSON-serialisable (inf / nan never appear).
#   smoke_params(cond) -> smoke dict for effects.smoke_render, or None when no space is smoky:
#     {"visibility_m": None, "spaces": [{"rect", "visibility_m"}...], **smoke_shape}
#   camera_pose(x, y, yaw, height) -> {"world_position": [x, y, height], "world_rpy": [0, 0, yaw]}
#   lamps(world_sdf_text, cond, cameras) -> effects lamp list: every light of the world WITH its diffuse (from the SDF, i.e. after
#     scale_lights), plus when cond["headlight"] > 0 one spot per camera pose in `cameras` at the camera, pointing along its yaw,
#     diffuse [h, h, 0.95 h]. Lamps whose diffuse is all 0 are left out.
#   apply(frame_dir, cond, effects_cfg, lamps, camera, K, rng) -> meta["effects"]: reads rgb.png / depth.png (uint16 mm) /
#     meta.json, writes rgb_clean.png + depth_clean.png (byte copies), then smoke (if smoke_params) via effects.smoke_render on the
#     clean depth + add_smoke (x scatter_gain) + effects.smoke_depth, then effects.sensor_noise ALWAYS; overwrites rgb.png, and
#     depth.png only when there is smoke (no smoke: depth.png left as is); meta.json gets "effects" = {"smoke": smoke_params or None, "noise_gain", "light": level of the camera's space
#     (None outside every space), "headlight"}.
# effects.py: smoke_render / transmission honour smoke["spaces"] (per-point base beta = 3 / V of the containing rect, 0 when that
#   space's V is None; points in no rect use smoke["visibility_m"], None = 0); without "spaces" exactly as before.
#   world_lights() also returns "diffuse" (rgb) for every non-ambient light.
# robot.robot_sdf(p, name, odom_truth=False, headlight=0.0): headlight > 0 -> a <light type="spot" name="headlight_<k>"> in every
#   camera link (k = 0 front, 1.. ring) with diffuse h h 0.95h, direction 1 0 0, HEADLIGHT spot / attenuation / range.
# CLI sim/building/conditions.py BUILDING --light L|mixed --smoke V|mixed|none --headlight H --seed S --out DIR
#   [--effects-config sim/configs/effects_train.yaml] -> DIR/world.sdf + DIR/conditions.json (cond + "effects_config" path).
import json, math, pathlib, subprocess, sys, xml.etree.ElementTree as ET
import numpy as np
import pytest
import yaml
from PIL import Image

SIM = pathlib.Path(__file__).resolve().parents[1]
REPO = SIM.parent
sys.path.insert(0, str(SIM))
from doorsim import building, conditions, effects, glass, robot

EFX = yaml.safe_load((SIM / "configs" / "effects_train.yaml").read_text())
BCFG = yaml.safe_load((SIM / "configs" / "buildings_v1.yaml").read_text())


@pytest.fixture(scope="module")
def bld(tmp_path_factory):
    d = tmp_path_factory.mktemp("bld")
    b = building.make_building(BCFG, 5, "train")
    building.write(b, d, "bld_train_005")
    base = d / "bld_train_005"
    return base, json.loads((base.with_suffix(".building.json")).read_text()), base.with_suffix(".sdf").read_text()


def light_diffuse(sdf_text):
    world = ET.fromstring(sdf_text[sdf_text.index("<sdf"):]).find("world")
    return {l.get("name"): [float(v) for v in l.findtext("diffuse").split()[:3]] for l in world.iter("light")}


# --- lights in the world file ---
def test_scale_lights_one_level(bld):
    _, _, sdf = bld
    before = light_diffuse(sdf)
    after = light_diffuse(conditions.scale_lights(sdf, 0.05, 0.05))
    assert before.keys() == after.keys()
    for n, rgb in before.items():
        assert after[n] == pytest.approx([0.05 * c for c in rgb]), n


def test_scale_lights_per_space_and_sun(bld):
    _, bj, sdf = bld
    spaces = ["corridor"] + [r["id"] for r in bj["layout"]["rooms"]]
    levels = {s: 0.1 * (i + 1) for i, s in enumerate(spaces)}
    before, after = light_diffuse(sdf), light_diffuse(conditions.scale_lights(sdf, levels, 0.02))
    for s, lv in levels.items():
        assert after[f"ceiling_light_{s}"] == pytest.approx([lv * c for c in before[f"ceiling_light_{s}"]])
    assert after["sun"] == pytest.approx([0.02 * c for c in before["sun"]])


def test_scale_lights_leaves_everything_else(bld):
    _, _, sdf = bld
    out = conditions.scale_lights(sdf, 0.5, 0.5)
    strip = lambda t: [l for l in t.splitlines() if "<diffuse>" not in l]
    assert strip(out) == strip(sdf)


# --- per-room draws ---
def test_sample_rooms_fixed():
    r = conditions.sample_rooms(np.random.default_rng(0), ["corridor", "r0"], 0.05, 4.0, EFX["smoke"])
    assert r == {s: {"mode": "fixed", "light": 0.05, "visibility_m": 4.0} for s in ("corridor", "r0")}
    r = conditions.sample_rooms(np.random.default_rng(0), ["corridor"], 1.0, None, EFX["smoke"])
    assert r["corridor"]["visibility_m"] is None


def test_sample_rooms_mixed_follows_training_mix():
    spaces = [f"r{i}" for i in range(3000)]
    r = conditions.sample_rooms(np.random.default_rng(1), spaces, "mixed", "mixed", EFX["smoke"])
    modes = [v["mode"] for v in r.values()]
    for m, spec in conditions.LIGHT_MODES.items():
        assert modes.count(m) / len(modes) == pytest.approx(spec["p"], abs=0.04)
        lv = [v["light"] for v in r.values() if v["mode"] == m]
        assert min(lv) >= spec["level"][0] and max(lv) <= spec["level"][1]
    vis = [v["visibility_m"] for v in r.values()]
    smoky = [v for v in vis if v is not None]
    assert len(smoky) / len(vis) == pytest.approx(EFX["smoke"]["p"], abs=0.04)
    lo, hi = EFX["smoke"]["visibility_m"]
    assert min(smoky) >= lo and max(smoky) <= hi


def test_sample_rooms_seeded():
    a = conditions.sample_rooms(np.random.default_rng(7), ["corridor", "r0", "r1"], "mixed", "mixed", EFX["smoke"])
    b = conditions.sample_rooms(np.random.default_rng(7), ["corridor", "r0", "r1"], "mixed", "mixed", EFX["smoke"])
    assert a == b


# --- make ---
def test_make_fixed(bld):
    _, bj, sdf = bld
    world, cond = conditions.make(bj, sdf, 0.05, 4.0, 0.8, 3, EFX)
    json.dumps(cond, allow_nan=False)
    assert cond["light"] == 0.05 and cond["smoke"] == 4.0 and cond["headlight"] == 0.8 and cond["seed"] == 3
    assert set(cond["spaces"]) == {"corridor", *[r["id"] for r in bj["layout"]["rooms"]]}
    for s, v in cond["spaces"].items():
        rect = bj["layout"]["corridor"] if s == "corridor" else next(r["rect"] for r in bj["layout"]["rooms"] if r["id"] == s)
        assert v["rect"] == rect and v["light"] == 0.05 and v["visibility_m"] == 4.0
    assert cond["sun_level"] == pytest.approx(0.05)
    assert light_diffuse(world)["sun"] == pytest.approx([0.05 * c for c in light_diffuse(sdf)["sun"]])
    assert set(cond["smoke_shape"]) == {"layer", "patch", "patch_scale", "seed"}


def test_make_mixed_sun_is_darkest_room(bld):
    _, bj, sdf = bld
    _, cond = conditions.make(bj, sdf, "mixed", "mixed", 0.8, 11, EFX)
    assert cond["sun_level"] == pytest.approx(min(v["light"] for v in cond["spaces"].values()))


def test_make_same_seed_same_smoke_shape_across_levels(bld):
    _, bj, sdf = bld
    a = conditions.make(bj, sdf, 1.0, 8.0, 0.0, 5, EFX)[1]["smoke_shape"]
    b = conditions.make(bj, sdf, 1.0, 2.0, 0.0, 5, EFX)[1]["smoke_shape"]
    assert a == b


def test_smoke_params(bld):
    _, bj, sdf = bld
    assert conditions.smoke_params(conditions.make(bj, sdf, 1.0, None, 0.0, 0, EFX)[1]) is None
    cond = conditions.make(bj, sdf, 1.0, 4.0, 0.0, 0, EFX)[1]
    sp = conditions.smoke_params(cond)
    assert sp["visibility_m"] is None
    assert {(tuple(s["rect"]), s["visibility_m"]) for s in sp["spaces"]} == \
           {(tuple(v["rect"]), 4.0) for v in cond["spaces"].values()}
    for k in ("layer", "patch", "patch_scale", "seed"):
        assert sp[k] == cond["smoke_shape"][k]


# --- smoke per space in effects.py ---
W, H = 64, 48
K = [40.0, 0.0, 32.0, 0.0, 40.0, 24.0, 0.0, 0.0, 1.0]
CAM = {"world_position": [0.0, 0.0, 1.0], "world_rpy": [0.0, 0.0, 0.0]}


def test_spaces_beta_matches_plain_smoke_when_one_space_covers_all():
    depth = np.full((H, W), 3.0)
    plain = {"visibility_m": 4.0, "layer": None, "patch": 0.0, "patch_scale": 1.5, "seed": 1}
    spaced = {**plain, "visibility_m": None, "spaces": [{"rect": [-10, -10, 10, 10], "visibility_m": 4.0}]}
    assert effects.transmission(depth, K, CAM, spaced) == pytest.approx(effects.transmission(depth, K, CAM, plain))


def test_spaces_clear_space_has_no_smoke_and_boundary_cuts_it():
    depth = np.full((H, W), 4.0)  # straight ahead along +x from x 0 to 4
    clear = {"visibility_m": None, "layer": None, "patch": 0.0, "patch_scale": 1.5, "seed": 1,
             "spaces": [{"rect": [-10, -10, 10, 10], "visibility_m": None}]}
    assert effects.transmission(depth, K, CAM, clear) == pytest.approx(np.ones((H, W)))
    half = {**clear, "spaces": [{"rect": [-10, -10, 2, 10], "visibility_m": None}, {"rect": [2, -10, 10, 10], "visibility_m": 2.0}]}
    t = effects.transmission(depth, K, CAM, half, samples=200)
    # centre pixel: 2 m clear then 2 m at beta 1.5
    assert t[H // 2, W // 2] == pytest.approx(math.exp(-1.5 * 2.0), rel=0.03)


def test_world_lights_have_diffuse(bld):
    base, _, _ = bld
    ls = effects.world_lights(base.with_suffix(".sdf"))
    pts = [l for l in ls if l["type"] != "ambient"]
    assert pts and all(len(l["diffuse"]) == 3 for l in pts)


# --- lamps + camera pose ---
def test_camera_pose():
    assert conditions.camera_pose(1.0, 2.0, 0.5, 0.4) == {"world_position": [1.0, 2.0, 0.4], "world_rpy": [0.0, 0.0, 0.5]}


def test_lamps_scaled_room_lights_and_headlights(bld):
    _, bj, sdf = bld
    world, cond = conditions.make(bj, sdf, 0.0, None, 0.8, 0, EFX)
    cams = [conditions.camera_pose(1.0, 2.0, 0.0, 0.4), conditions.camera_pose(1.0, 2.0, math.pi / 2, 0.4)]
    ls = conditions.lamps(world, cond, cams)
    assert len(ls) == 2 and all(l["type"] == "spot" for l in ls)       # room lights at level 0 drop out
    assert ls[0]["diffuse"] == pytest.approx([0.8, 0.8, 0.76])
    assert ls[0]["position"] == pytest.approx([1.0, 2.0, 0.4])
    assert ls[1]["direction"] == pytest.approx([0.0, 1.0, 0.0], abs=1e-9)
    assert ls[0]["spot"] == conditions.HEADLIGHT["spot"]
    world, cond = conditions.make(bj, sdf, 1.0, None, 0.0, 0, EFX)
    ls = conditions.lamps(world, cond, cams)
    assert ls and all(l["name"] != "headlight" and l["type"] != "spot" for l in ls)


# --- apply on a saved frame ---
def write_frame(d, rgb_val=150, depth_m=3.0):
    d.mkdir(parents=True)
    Image.fromarray(np.full((H, W, 3), rgb_val, np.uint8)).save(d / "rgb.png")
    Image.fromarray(glass.to_png16(np.full((H, W), depth_m, np.float32))).save(d / "depth.png")
    (d / "meta.json").write_text(json.dumps({"door": 1}))


def test_apply_keeps_clean_copies_and_adds_noise_only_without_smoke(tmp_path, bld):
    _, bj, sdf = bld
    world, cond = conditions.make(bj, sdf, 1.0, None, 0.0, 0, EFX)
    f = tmp_path / "000_door1_front_far"
    write_frame(f)
    raw_rgb, raw_depth = (f / "rgb.png").read_bytes(), (f / "depth.png").read_bytes()
    cam = conditions.camera_pose(*bj["layout"]["corridor"][:2], 0.0, 0.4)
    eff = conditions.apply(f, cond, EFX, conditions.lamps(world, cond, [cam]), cam, K, np.random.default_rng(0))
    assert (f / "rgb_clean.png").read_bytes() == raw_rgb and (f / "depth_clean.png").read_bytes() == raw_depth
    assert eff["smoke"] is None and eff["noise_gain"] >= 1
    assert (f / "depth.png").read_bytes() == raw_depth                 # no smoke: depth untouched
    assert not np.array_equal(np.asarray(Image.open(f / "rgb.png")), np.asarray(Image.open(f / "rgb_clean.png")))
    assert json.loads((f / "meta.json").read_text())["effects"] == eff


def test_apply_thick_smoke_hides_the_scene(tmp_path, bld):
    _, bj, sdf = bld
    world, cond = conditions.make(bj, sdf, 1.0, 1.0, 0.0, 0, EFX)
    c = bj["layout"]["corridor"]
    cam = conditions.camera_pose((c[0] + c[2]) / 2, (c[1] + c[3]) / 2, 0.0, 0.4)
    f = tmp_path / "001_door1_front_far"
    write_frame(f, rgb_val=230, depth_m=6.0)
    eff = conditions.apply(f, cond, EFX, conditions.lamps(world, cond, [cam]), cam, K, np.random.default_rng(0))
    assert eff["smoke"] is not None and eff["light"] == 1.0 and eff["headlight"] == 0.0
    clean = np.asarray(Image.open(f / "rgb_clean.png"), np.float32)
    smoky = np.asarray(Image.open(f / "rgb.png"), np.float32)
    assert abs(smoky.mean() - clean.mean()) > 30   # the bright wall is gone behind smoke
    d = np.asarray(Image.open(f / "depth.png"))
    assert (d != np.asarray(Image.open(f / "depth_clean.png"))).mean() > 0.5        # depth mostly lost / shortened


def test_apply_light_is_the_camera_space(tmp_path, bld):
    _, bj, sdf = bld
    world, cond = conditions.make(bj, sdf, "mixed", None, 0.8, 4, EFX)
    r = bj["layout"]["rooms"][0]
    cam = conditions.camera_pose((r["rect"][0] + r["rect"][2]) / 2, (r["rect"][1] + r["rect"][3]) / 2, 0.0, 0.4)
    f = tmp_path / "002"
    write_frame(f)
    eff = conditions.apply(f, cond, EFX, conditions.lamps(world, cond, [cam]), cam, K, np.random.default_rng(0))
    assert eff["light"] == cond["spaces"][r["id"]]["light"] and eff["headlight"] == 0.8
    far = conditions.camera_pose(-100.0, -100.0, 0.0, 0.4)
    f2 = tmp_path / "003"
    write_frame(f2)
    assert conditions.apply(f2, cond, EFX, [], far, K, np.random.default_rng(0))["light"] is None


# --- robot headlights ---
def test_robot_without_headlight_has_no_lights():
    p = robot.load_platform(SIM / "configs" / "robots" / "wheeled_ring.yaml")
    assert "<light" not in robot.robot_sdf(p, "robot")
    assert robot.robot_sdf(p, "robot") == robot.robot_sdf(p, "robot", headlight=0.0)


def test_robot_headlight_per_camera():
    p = robot.load_platform(SIM / "configs" / "robots" / "wheeled_ring.yaml")
    m = ET.fromstring(robot.robot_sdf(p, "robot", headlight=0.8)).find("model")
    names = {"camera_link": 0, **{f"camera_link_{k}": k for k in range(1, p["camera_ring"]["count"])}}
    for ln, k in names.items():
        l = m.find(f"link[@name='{ln}']/light[@name='headlight_{k}']")
        assert l is not None and l.get("type") == "spot", ln
        assert [float(v) for v in l.findtext("diffuse").split()[:3]] == pytest.approx([0.8, 0.8, 0.76])
        assert [float(v) for v in l.findtext("direction").split()] == pytest.approx([1, 0, 0])
        assert float(l.findtext("spot/outer_angle")) == pytest.approx(conditions.HEADLIGHT["spot"][1])
        assert float(l.findtext("attenuation/range")) == pytest.approx(conditions.HEADLIGHT["range"])


# --- CLI ---
def test_cli_writes_world_and_conditions(tmp_path, bld):
    base, _, _ = bld
    out = tmp_path / "run"
    subprocess.run([sys.executable, str(SIM / "building" / "conditions.py"), str(base), "--light", "mixed", "--smoke", "4",
                    "--headlight", "0.8", "--seed", "2", "--out", str(out)], check=True, cwd=REPO)
    cond = json.loads((out / "conditions.json").read_text())
    assert cond["light"] == "mixed" and cond["smoke"] == 4.0 and cond["headlight"] == 0.8 and cond["seed"] == 2
    assert pathlib.Path(cond["effects_config"]).name == "effects_train.yaml"
    ET.fromstring((out / "world.sdf").read_text().split("?>", 1)[-1])
    subprocess.run([sys.executable, str(SIM / "building" / "conditions.py"), str(base), "--light", "1", "--smoke", "none",
                    "--headlight", "0", "--seed", "0", "--out", str(tmp_path / "clear")], check=True, cwd=REPO)
    assert json.loads((tmp_path / "clear" / "conditions.json").read_text())["smoke"] is None
