"""Builds door_room_v2 from door_room_v1: same world + the clutter pool (sim/doorsim/clutter.py), and the
attributes file with one class-null entry per pool item. Run: python3 sim/door_room/make_v2.py"""
import json, pathlib, sys

ROOM = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOM.parent))
from doorsim import clutter

sdf = (ROOM / "door_room_v1.sdf").read_text().replace('<world name="door_room_v1">', '<world name="door_room_v2">')
sdf = sdf.replace("  </world>", "\n    <!-- clutter pool (sim/doorsim/clutter.py): visual-only static models, parked under the floor, moved per scene -->\n"
                  + clutter.world_sdf(clutter.POOL) + "  </world>")
(ROOM / "door_room_v2.sdf").write_text(sdf)

attrs = json.loads((ROOM / "door_room_v1.attributes.json").read_text())
attrs["_note"] = (attrs["_note"].replace("door_room_v1", "door_room_v2")
                  + " Clutter pool (ids 100-254, class null, no box of their own): role 'clutter' = shapes and furniture, "
                  "'lookalike' = cabinets, drawers, picture frames etc. that look like doors or handles. They sit under the floor "
                  "until a scene moves them in (sim/doorsim/clutter.py); the generator reports what hides each door / handle as visible_fraction.")
for p in clutter.POOL:
    attrs["objects"][p["name"]] = {"id": p["id"], "class": None, "role": p["role"], "kind": p["kind"], "place": p["place"], "size": p["size"]}
(ROOM / "door_room_v2.attributes.json").write_text(json.dumps(attrs, indent=2) + "\n")
