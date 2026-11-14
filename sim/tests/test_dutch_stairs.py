# Dutch staircase world for the Quad-SDK Spot test. python3 -m pytest sim/tests/test_dutch_stairs.py
import pathlib, sys, xml.etree.ElementTree as ET
SIM = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIM))
from quad import dutch_stairs as ds


def test_steps_climb_evenly_to_the_landing():
    b = {n: (x, y, z, sx, sy, sz) for n, x, y, z, sx, sy, sz in ds.boxes(steps=13, rise=0.2, going=0.22)}
    tops = [b[f"step_{i}"][2] + b[f"step_{i}"][5] / 2 for i in range(1, 13)] + [b["landing"][2] + b["landing"][5] / 2]
    assert [round(t, 6) for t in tops] == [round(0.2 * k, 6) for k in range(1, 14)]
    fronts = [b[f"step_{i}"][0] - 0.11 for i in range(1, 13)]
    assert [round(f, 6) for f in fronts] == [round(1.5 + 0.22 * k, 6) for k in range(12)]
    assert b["step_1"][4] == 0.9 and b["step_1"][2] - b["step_1"][5] / 2 == 0   # solid down to the floor


def test_walls_leave_the_clear_width():
    b = {n: (x, y, z, sx, sy, sz) for n, x, y, z, sx, sy, sz in ds.boxes(width=0.9)}
    inner = b["wall_l"][1] - b["wall_l"][4] / 2
    assert abs(inner - 0.45) < 1e-9 and abs(b["wall_r"][1] + b["wall_r"][4] / 2 + 0.45) < 1e-9
    assert "wall_l" not in {n for n, *_ in ds.boxes(walls=False)}


def test_world_parses_with_physics_plugins():
    w = ET.fromstring(ds.world(rise=0.15)).find("world")
    assert {p.get("name") for p in w.iter("plugin")} >= {"gz::sim::systems::Physics", "gz::sim::systems::Contact"}
    assert len(w.find("model[@name='stairs']").findall("link")) == 12 + 1 + 2
