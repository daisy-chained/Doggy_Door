# Balanced door types (door room v6, Madelief 2026-11-08: "an equal amount of doors of all types"): door types dealt in
# turn per split (worlds config `balance`), each door closed in half its scenes (doors.balance_states) and the views going
# round the doors and sides (view.balance). No Gazebo needed: python3 -m pytest sim/tests/test_balance.py
import collections, json, pathlib, sys
import numpy as np
import yaml

SIM = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIM))
from doorsim import doors, sampling

V5 = yaml.safe_load((SIM / "configs" / "worlds_v5.yaml").read_text())
V6 = yaml.safe_load((SIM / "configs" / "worlds_v6.yaml").read_text())
GEN6 = yaml.safe_load((SIM / "configs" / "door_room_v6_6k.yaml").read_text())


def cell(spec):
    return (spec["glass"], spec["door_type"])


# --- door specs ---
def test_sample_door_force_sets_the_type_and_keeps_the_rest_random():
    rng = np.random.default_rng(0)
    for force in ({"glass": True, "door_type": "hinged"}, {"glass": False, "door_type": "sliding"},
                  {"glass": True, "door_type": "hinged", "handle_type": "knob"}):
        specs = [doors.sample_door(rng, V6["door"], "a", force=force) for _ in range(50)]
        for s in specs:
            assert all(s[k] == v for k, v in force.items())
            if s["door_type"] == "sliding":
                assert s["handle_type"] == "flush_pull" and "slide_y" in s and "hinge_y" not in s
            else:
                assert s["handle_type"] in ("lever", "knob") and "hinge_y" in s and "slide_y" not in s
        assert len({s["width"] for s in specs}) > 40  # sizes still random
        if "handle_type" not in force and force["door_type"] == "hinged":
            assert {s["handle_type"] for s in specs} == {"lever", "knob"}


def test_sample_door_without_force_unchanged():
    a = [doors.sample_door(np.random.default_rng(3), V5["door"], "b") for _ in range(5)]
    b = [doors.sample_door(np.random.default_rng(3), V5["door"], "b", force=None) for _ in range(5)]
    assert a == b


def test_v5_worlds_unchanged():
    # configs without `balance` still give the same worlds as the committed v5 world set
    saved = json.loads((SIM / "door_room" / "v5" / "worlds.json").read_text())["worlds"]
    made = doors.make_worlds(V5)
    assert [w["doors"] for w in made] == [w["doors"] for w in saved]


# --- balanced worlds ---
def test_balanced_worlds_deal_door_types_per_split():
    ws = doors.make_worlds(V6)
    assert len(ws) == V6["count"] + V6["heldout_count"]
    assert doors.make_worlds(V6) == ws  # fixed seed
    cells = V6["balance"]["cells"]
    total = sum(c["w"] for c in cells)
    for sp in ("train", "val", "test"):
        normal = [d for w in ws if w["split"] == sp and not w["heldout"] for d in w["doors"].values()]
        got = collections.Counter(cell(d) for d in normal)
        assert set(got) == {(c["glass"], c["door_type"]) for c in cells}
        for c in cells:
            want = len(normal) * c["w"] / total
            assert abs(got[(c["glass"], c["door_type"])] - want) <= c["w"]  # dealt in turn: off by at most one cycle's share


def test_balanced_normal_worlds_have_no_heldout_doors():
    for w in doors.make_worlds(V6):
        if not w["heldout"]:
            assert not any(doors.is_heldout(d, V6["heldout"]) for d in w["doors"].values())
            assert not any(d["glass"] and d["handle_type"] == "knob" for d in w["doors"].values())


def test_balanced_heldout_worlds_take_heldout_cells_in_turn():
    held = [w for w in doors.make_worlds(V6) if w["heldout"]]
    assert len(held) == V6["heldout_count"] and all(w["split"] == "test" for w in held)
    hcells = V6["balance"]["heldout_cells"]
    got = []
    for w in held:
        h = [d for s, d in w["doors"].items() if w["doors_heldout"][s]]
        assert len(h) == 1  # one held-out door, the other is a normal one
        got.append(h[0])
    for i, d in enumerate(got):
        assert all(d[k] == v for k, v in hcells[i % len(hcells)].items())


# --- balanced states ---
def attrs_of(world):
    return doors.attributes(world, "t")


def test_balance_states_each_door_closed_in_half_its_scenes():
    w = doors.make_worlds(V6)[0]
    a = attrs_of(w)
    cfg = GEN6["doors"]
    assert cfg["balance_states"]
    closed = collections.Counter()
    for scene in range(6):
        t = sampling.door_targets(np.random.default_rng(scene), a, cfg, scene=scene)
        assert len(t) == 2
        shut = [n for n, p in t.items() if p == 0.0]
        assert len(shut) == 1  # one closed, one open in every scene
        closed.update(shut)
        for n, p in t.items():
            if p != 0.0:
                lo, hi = a["objects"][n]["joint"]["range"]
                end = lo if abs(lo) > abs(hi) else hi
                assert cfg["min_open_fraction"] * abs(end) - 1e-9 <= abs(p) <= abs(end) + 1e-9 and p * end > 0
    assert set(closed.values()) == {3}


def test_door_targets_without_balance_unchanged():
    w = doors.make_worlds(V5)[0]
    a = attrs_of(w)
    cfg = {"p_closed": 0.3, "min_open_fraction": 0.1}
    assert sampling.door_targets(np.random.default_rng(1), a, cfg) == sampling.door_targets(np.random.default_rng(1), a, cfg, scene=7)


# --- balanced views ---
def test_balanced_views_go_round_doors_and_sides():
    a = attrs_of(doors.make_worlds(V6)[0])
    cfg = GEN6["view"]
    assert cfg["balance"]
    rng = np.random.default_rng(0)
    seen = collections.Counter()
    for i in range(20):
        v = sampling.view(rng, a, cfg, index=i)
        assert v["side"] == doors.side_of(v["world_position"])
        seen[(v["target_door"], v["side"])] += 1
        again = sampling.view(np.random.default_rng(99), a, cfg, index=i)  # a retried pose keeps the door and side
        assert (again["target_door"], again["side"]) == (v["target_door"], v["side"])
    assert len(seen) == 4 and set(seen.values()) == {5}


def test_view_without_balance_unchanged():
    a = attrs_of(doors.make_worlds(V5)[0])
    cfg = dict(GEN6["view"], balance=False)
    assert sampling.view(np.random.default_rng(2), a, cfg) == sampling.view(np.random.default_rng(2), a, cfg, index=5)


# --- the whole plan: every train cell gets the same number of target views ---
def test_v6_plan_gives_equal_views_per_type_and_state():
    ws = doors.make_worlds(V6)
    spw, nv = GEN6["scenes_per_world"], GEN6["views_per_scene"]
    counts = collections.Counter()
    for i, w in enumerate(ws):
        if w["split"] != "train":
            continue
        a = attrs_of(w)
        for k in range(spw):
            scene = i * spw + k
            t = sampling.door_targets(np.random.default_rng(scene), a, GEN6["doors"], scene=scene)
            for v in range(nv):
                cam = sampling.view(np.random.default_rng([scene, v]), a, GEN6["view"], index=v)
                spec = w["doors"][cam["target_door"].split("_")[1]]
                op = doors.seen_from(spec, cam["side"])["opening"]
                counts[(spec["glass"], op, "closed" if t[cam["target_door"]] == 0.0 else "open")] += 1
    assert len(counts) == 10  # glass x push / pull / slide x open / closed, minus glass sliding (held out)
    assert max(counts.values()) / min(counts.values()) <= 1.15
    assert min(counts.values()) >= 300  # vs 36 closed glass doors in the v5 train split


def test_generator_passes_scene_and_view_index():
    import inspect
    from doorsim import generate
    src = inspect.getsource(generate.render_scenes)
    assert "door_targets(rng, attrs, cfg[\"doors\"], scene=scene)" in src
    assert "sampling.view(rng, attrs, cfg[\"view\"], index=v)" in src
