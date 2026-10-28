# Write the metadata for one frame: attributes.json plus the door angles at that moment.
# usage: meta.py out.json door_a=<rad> door_b=<rad>
import json, sys, pathlib
attrs = json.loads((pathlib.Path(__file__).parent / "attributes.json").read_text())
frame = {"classes": attrs["classes"], "objects": attrs["objects"]}
for arg in sys.argv[2:]:
    name, angle = arg.split("=")
    frame["objects"][name]["angle_rad"] = float(angle)
    frame["objects"][name]["state"] = "closed" if float(angle) == 0 else "open"
pathlib.Path(sys.argv[1]).write_text(json.dumps(frame, indent=2) + "\n")
print(f"saved {sys.argv[1]}")
