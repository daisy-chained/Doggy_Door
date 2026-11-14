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


def test_ply_matches_the_boxes():
    txt = ds.ply(steps=13, rise=0.2, landing=2.0)
    head, body = txt.split("end_header\n")
    nv = int(head.split("element vertex ")[1].split()[0]); nf = int(head.split("element face ")[1].split()[0])
    nb = len(ds.boxes(steps=13, rise=0.2, landing=2.0))
    assert (nv, nf) == (4 + 8 * nb, 2 + 12 * nb)
    rows = body.strip().split("\n")
    zs = [float(r.split()[2]) for r in rows[:nv]]
    assert max(zs) == 3.8 and min(zs) == 0.0          # walls 1.2 m above the 2.6 m landing
    assert all(r.startswith("3 ") for r in rows[nv:]) and len(rows) == nv + nf


def test_install_writes_world_and_ply(tmp_path):
    import subprocess
    share = tmp_path / "quad_sim_scripts" / "share" / "quad_sim_scripts"; (share / "worlds").mkdir(parents=True)
    subprocess.run([sys.executable, str(SIM / "quad" / "dutch_stairs.py"), "--rise", "0.15", "--name", "ds15",
                    "--install", str(tmp_path)], check=True, capture_output=True)
    assert (share / "worlds" / "ds15.sdf").exists() and (share / "models" / "ds15" / "meshes" / "ds15.ply").exists()
