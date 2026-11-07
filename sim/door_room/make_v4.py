"""Builds the door room v4 / v5 worlds (door variety): one world file + attributes file per world in door_room/<out>/ (config key out, default v4), and worlds.json
(the list the generator reads: split, held-out flag and door specs per world). The doors come from sim/doorsim/doors.py,
the numbers from the worlds config. Run: python3 sim/door_room/make_v4.py sim/configs/worlds_v4.yaml"""
import json, pathlib, sys
import yaml

ROOM = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOM.parent))
from doorsim import doors

cfg = yaml.safe_load(pathlib.Path(sys.argv[1]).read_text())
ver = cfg.get("out", "v4")  # v5 = door stop + hinge knuckles (configs/worlds_v5.yaml)
out = ROOM / ver
out.mkdir(exist_ok=True)
index = []
for w in doors.make_worlds(cfg):
    n = w["name"]
    (out / f"{n}.sdf").write_text(doors.world_sdf(w, n))
    (out / f"{n}.attributes.json").write_text(json.dumps(doors.attributes(w, n), indent=2) + "\n")
    index.append({"name": n, "split": w["split"], "heldout": w["heldout"], "sdf": f"door_room/{ver}/{n}.sdf",
                  "attributes": f"door_room/{ver}/{n}.attributes.json", "doors": w["doors"], "doors_heldout": w["doors_heldout"]})
(out / "worlds.json").write_text(json.dumps({"seed": cfg["seed"], "config": cfg, "worlds": index}, indent=2) + "\n")
print(f"{len(index)} worlds in {out}")
