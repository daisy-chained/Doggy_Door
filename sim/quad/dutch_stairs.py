"""Straight Dutch staircase world for the Quad-SDK Spot test (Gazebo Harmonic, same physics block as Quad-SDK's step_20cm.sdf).

    python3 sim/quad/dutch_stairs.py --rise 0.20 --out runs/quad/stairs_20.sdf
    python3 sim/quad/dutch_stairs.py --rise 0.20 --landing 2.0 --name dutch_stairs_20 --install ~/git/quad_ws/install
    ros2 launch quad_utils quad_gazebo.py world:=dutch_stairs_20.sdf

Quad-SDK's planners need the terrain as a mesh too: its mesh_to_grid_map node loads
<quad_sim_scripts share>/models/<world>/meshes/<world>.ply. --install writes the world + that PLY into the colcon install space
(not into their source tree), so Quad-SDK's own code stays untouched.

Typical Dutch house stair (Bouwbesluit-era homes): 13 steps, rise ~19-21 cm, going ~22 cm, ~0.9 m between the walls, walls on
both sides. The robot starts at the origin facing +x; the first riser is at x = run_up. Each step is a solid box down to the
floor (no open risers), with a flat landing at the top. --rise lets us make an easier ladder (10 / 15 / 18 / 20 cm)."""
import argparse, pathlib

PHYSICS = """    <physics type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1</real_time_factor>
      <real_time_update_rate>1000</real_time_update_rate>
      <ode><solver><iters>300</iters></solver></ode>
    </physics>
    <scene><shadows>false</shadows></scene>
    <gravity>0 0 -9.81</gravity>
""" + "".join(f'    <plugin filename="libgz-sim8-{f}-system" name="gz::sim::systems::{n}"/>\n' for f, n in (
    ("physics", "Physics"), ("scene-broadcaster", "SceneBroadcaster"), ("sensors", "Sensors"),
    ("user-commands", "UserCommands"), ("contact", "Contact"), ("imu", "Imu"), ("forcetorque", "ForceTorque")))


def _box(name, x, y, z, sx, sy, sz, rgb):
    g = f"<box><size>{sx:.4f} {sy:.4f} {sz:.4f}</size></box>"
    return (f'      <link name="{name}"><pose>{x:.4f} {y:.4f} {z:.4f} 0 0 0</pose>'
            f'<collision name="c"><geometry>{g}</geometry><surface><friction><ode><mu>0.9</mu><mu2>0.9</mu2></ode></friction></surface></collision>'
            f'<visual name="v"><geometry>{g}</geometry><material><diffuse>{rgb} 1</diffuse><ambient>{rgb} 1</ambient></material></visual></link>\n')


def boxes(steps=13, rise=0.20, going=0.22, width=0.90, run_up=1.5, landing=1.2, walls=True, wall_t=0.1):
    """[(name, x, y, z, sx, sy, sz)] in world coords: steps (solid to the floor), top landing, side walls."""
    out = []
    for i in range(steps - 1):                      # the last rise goes onto the landing
        h = rise * (i + 1)
        out.append((f"step_{i + 1}", run_up + going * (i + 0.5), 0.0, h / 2, going, width, h))
    top = rise * steps
    x0 = run_up + going * (steps - 1)
    out.append(("landing", x0 + landing / 2, 0.0, top / 2, landing, width, top))
    if walls:
        L, H = x0 + landing, top + 1.2
        for s in (1, -1):
            out.append((f"wall_{'l' if s > 0 else 'r'}", L / 2, s * (width + wall_t) / 2, H / 2, L, wall_t, H))
    return out


def world(**kw):
    links = "".join(_box(n, x, y, z, sx, sy, sz, "0.75 0.65 0.5" if n.startswith(("step", "landing")) else "0.9 0.9 0.88")
                    for n, x, y, z, sx, sy, sz in boxes(**kw))
    return f"""<?xml version="1.0"?>
<sdf version="1.6">
  <world name="default">
{PHYSICS}    <light name="sun" type="directional"><cast_shadows>true</cast_shadows><pose>0 0 10 0 0 0</pose>
      <diffuse>1 1 1 1</diffuse><specular>0.1 0.1 0.1 1</specular><direction>-1 0 -1</direction></light>
    <model name="ground"><static>true</static><link name="l">
      <collision name="c"><geometry><plane><normal>0 0 1</normal><size>40 40</size></plane></geometry></collision>
      <visual name="v"><geometry><plane><normal>0 0 1</normal><size>40 40</size></plane></geometry></visual></link></model>
    <model name="stairs"><static>true</static>
{links}    </model>
  </world>
</sdf>
"""


def ply(ground=(-3.0, 9.0, -3.0, 3.0), **kw):
    """ASCII PLY triangle mesh of the stairs + a ground rectangle, for Quad-SDK's terrain map (2.5D height layer)."""
    V, F = [], []
    x0, x1, y0, y1 = ground
    V += [(x0, y0, 0), (x1, y0, 0), (x1, y1, 0), (x0, y1, 0)]; F += [(0, 1, 2), (0, 2, 3)]
    for _, x, y, z, sx, sy, sz in boxes(**kw):
        b = len(V)
        V += [(x + dx * sx / 2, y + dy * sy / 2, z + dz * sz / 2) for dz in (-1, 1) for dy in (-1, 1) for dx in (-1, 1)]
        for q in ((0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)):
            F += [(b + q[0], b + q[1], b + q[2]), (b + q[0], b + q[2], b + q[3])]
    head = (f"ply\nformat ascii 1.0\nelement vertex {len(V)}\nproperty float x\nproperty float y\nproperty float z\n"
            f"element face {len(F)}\nproperty list uchar int vertex_indices\nend_header\n")
    return head + "".join(f"{a:.4f} {b:.4f} {c:.4f}\n" for a, b, c in V) + "".join(f"3 {a} {b} {c}\n" for a, b, c in F)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=13)
    ap.add_argument("--rise", type=float, default=0.20)
    ap.add_argument("--going", type=float, default=0.22)
    ap.add_argument("--width", type=float, default=0.90)
    ap.add_argument("--landing", type=float, default=1.2)
    ap.add_argument("--no-walls", action="store_true")
    ap.add_argument("--out", default="")
    ap.add_argument("--name", default="", help="world name for --install")
    ap.add_argument("--install", default="", help="colcon install dir of the Quad-SDK workspace")
    a = ap.parse_args()
    kw = dict(steps=a.steps, rise=a.rise, going=a.going, width=a.width, landing=a.landing, walls=not a.no_walls)
    if a.out:
        p = pathlib.Path(a.out); p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(world(**kw)); print(p)
    if a.install:
        share = pathlib.Path(a.install).expanduser() / "quad_sim_scripts" / "share" / "quad_sim_scripts"
        if not share.is_dir():
            raise SystemExit(f"no quad_sim_scripts share dir in {a.install} (build Quad-SDK first)")
        m = share / "models" / a.name / "meshes"; m.mkdir(parents=True, exist_ok=True)
        (share / "worlds" / f"{a.name}.sdf").write_text(world(**kw))
        (m / f"{a.name}.ply").write_text(ply(**kw))
        print(share / "worlds" / f"{a.name}.sdf")


if __name__ == "__main__":
    main()
