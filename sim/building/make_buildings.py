"""Builds the buildings of a config (robot run phase 2e): per building a world file, a building.json (layout, doors with label ids,
approach poses) and a Nav2 map (pgm + yaml) in the output directory. The numbers come from the config (sim/configs/buildings_v1.yaml),
the geometry from sim/doorsim/building.py. Run: python3 sim/building/make_buildings.py sim/configs/buildings_v1.yaml [--out DIR]
Building i uses seed config.seed + i and is named bld_<split>_<seed:03d>."""
import argparse, pathlib, sys
import yaml

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from doorsim import building


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("config")
    ap.add_argument("--out", default=str(HERE / "out"))
    args = ap.parse_args()
    cfg = yaml.safe_load(pathlib.Path(args.config).read_text())
    split = cfg.get("split", "train")
    for seed in range(cfg["seed"], cfg["seed"] + cfg["count"]):
        name = f"bld_{split}_{seed:03d}"
        b = building.make_building(cfg, seed, split)
        building.write(b, args.out, name)
        print(f"{name}: {len(b['layout']['rooms'])} rooms, {len(b['doors'])} doors, {len(b['poses'])} poses")
    print(f"{cfg['count']} buildings in {args.out}")


if __name__ == "__main__":
    main()
