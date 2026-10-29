# Write the metadata for one frame: attributes.json, the door angles, and each object's 2D box.
# usage: meta.py out.json boxes.txt door_a=<rad> door_b=<rad>
#   boxes.txt = raw output of `gz topic -e -t /boxes -n 1`; the box label is the object id.
import json, re, sys, pathlib
attrs = json.loads((pathlib.Path(__file__).parent / "attributes.json").read_text())
frame = {"classes": attrs["classes"], "objects": attrs["objects"]}
by_id = {o["id"]: o for o in frame["objects"].values()}
for o in by_id.values():
    o["box"] = None  # stays None when the object is not in view
num = r"(?:x|y):\s*(\d+)"
for block in pathlib.Path(sys.argv[2]).read_text().split("annotated_box")[1:]:
    label = int(re.search(r"label:\s*(\d+)", block).group(1))
    x0, y0, x1, y1 = (int(v) for v in re.findall(num, block)[:4])
    if label not in by_id:
        sys.exit(f"box label {label} is not an id in attributes.json")
    by_id[label]["box"] = {"x_min": x0, "y_min": y0, "x_max": x1, "y_max": y1}
for arg in sys.argv[3:]:
    name, angle = arg.split("=")
    frame["objects"][name]["angle_rad"] = float(angle)
    frame["objects"][name]["state"] = "closed" if float(angle) == 0 else "open"
pathlib.Path(sys.argv[1]).write_text(json.dumps(frame, indent=2) + "\n")
seen = sum(o["box"] is not None for o in by_id.values())
print(f"saved {sys.argv[1]} ({seen}/{len(by_id)} objects with a box)")
