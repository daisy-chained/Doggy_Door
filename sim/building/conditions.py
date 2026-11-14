"""Patrol conditions for one building (dark + smoke runs, Madelief 2026-11-13): the building world with scaled lights + conditions.json.

    python3 sim/building/conditions.py sim/building/out/bld_train_005 --light 0.05 --smoke 4 --headlight 0.8 --seed 0 --out runs/cond/x

BUILDING = path without extension (reads BUILDING.building.json + BUILDING.sdf). --light L|mixed (factor on the room lights),
--smoke V|mixed|none (visibility in m), --headlight H (robot spot light, 0 = off), --seed S. Writes OUT/world.sdf and
OUT/conditions.json (the cond of doorsim.conditions.make + "effects_config"). door_tour.py --conditions OUT/conditions.json
puts smoke + camera noise on every saved frame; robot.launch.py headlight:=H adds the robot's headlights."""
import argparse, json, pathlib, sys
import yaml

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from doorsim import conditions


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("building")
    ap.add_argument("--light", default="1.0", help="factor on every room light, or mixed")
    ap.add_argument("--smoke", default="none", help="visibility in m, mixed or none")
    ap.add_argument("--headlight", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--effects-config", default=str(HERE.parent / "configs" / "effects_train.yaml"))
    a = ap.parse_args()
    base = pathlib.Path(a.building)
    bj = json.loads(base.with_suffix(".building.json").read_text())
    sdf = base.with_suffix(".sdf").read_text()
    cfg_path = pathlib.Path(a.effects_config).resolve()
    light = a.light if a.light == "mixed" else float(a.light)
    smoke = None if a.smoke == "none" else a.smoke if a.smoke == "mixed" else float(a.smoke)
    world, cond = conditions.make(bj, sdf, light, smoke, a.headlight, a.seed, yaml.safe_load(cfg_path.read_text()))
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "world.sdf").write_text(world)
    (out / "conditions.json").write_text(json.dumps({**cond, "effects_config": str(cfg_path)}, indent=1, allow_nan=False) + "\n")
    lv = [v["light"] for v in cond["spaces"].values()]
    vis = [v["visibility_m"] for v in cond["spaces"].values() if v["visibility_m"] is not None]
    print(f"{base.name}: light {a.light} (rooms {min(lv):.3g}..{max(lv):.3g}, sun {cond['sun_level']:.3g}), smoke {a.smoke} "
          f"({len(vis)}/{len(lv)} spaces smoky), headlight {a.headlight:g}, seed {a.seed} -> {out}")


if __name__ == "__main__":
    main()
