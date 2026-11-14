"""Straight Dutch staircase world for the Quad-SDK Spot test (Gazebo Harmonic, same physics block as Quad-SDK's step_20cm.sdf).

    python3 sim/quad/dutch_stairs.py --rise 0.20 --out runs/quad/stairs_20.sdf
    ros2 launch quad_utils quad_gazebo.py world:=$PWD/runs/quad/stairs_20.sdf     # an absolute path works as `world`

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=13)
    ap.add_argument("--rise", type=float, default=0.20)
    ap.add_argument("--going", type=float, default=0.22)
    ap.add_argument("--width", type=float, default=0.90)
    ap.add_argument("--no-walls", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    p = pathlib.Path(a.out); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(world(steps=a.steps, rise=a.rise, going=a.going, width=a.width, walls=not a.no_walls))
    print(p)


if __name__ == "__main__":
    main()
