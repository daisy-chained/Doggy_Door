"""Robot body contacts (bumper sensor on base_link): classify the other side, merge touches into episodes, count them.
Stdlib only. Scoped collision names look like "<world>::<model>::<link>::<collision>"."""
import re

_DOOR = re.compile(r"door_(\d+)_")


def kind(scoped_name):
    """{"kind": door | wall | clutter | other, "door": id or None, "model": model name} of one scoped collision name."""
    parts = scoped_name.split("::")
    model = parts[1] if len(parts) >= 4 else parts[0]
    m = _DOOR.match(model + "_")
    if m and model.startswith("door_"):
        return {"kind": "door", "door": int(m.group(1)), "model": model}
    k = "wall" if model == "walls" else "clutter" if model.startswith("clutter_") else "other"
    return {"kind": k, "door": None, "model": model}


def other_side(c1, c2, robot="robot"):
    """The collision name that is not the robot's; None when both or neither are."""
    def mine(n):
        return f"::{robot}::" in n or n.startswith(f"{robot}::")
    a, b = mine(c1), mine(c2)
    return c2 if a and not b else c1 if b and not a else None


def episodes(events, gap=0.5):
    """events = [(t, scoped name of the other collision)] in time order -> one episode per model per run of touches (break <= gap)."""
    out, open_ = [], {}
    for t, name in events:
        k = kind(name)
        e = open_.get(k["model"])
        if e is not None and t - e["t1"] <= gap:
            e["t1"], e["n"] = t, e["n"] + 1
        else:
            e = {"t0": t, "t1": t, "model": k["model"], "kind": k["kind"], "door": k["door"], "n": 1}
            open_[k["model"]] = e
            out.append(e)
    return sorted(out, key=lambda e: e["t0"])


def summary(eps, door_state):
    """Counts of episodes; door_state(door_id, t0) -> "closed" | "open" splits the door touches."""
    s = {"episodes": len(eps), "closed_door": 0, "open_door": 0, "wall": 0, "clutter": 0, "other": 0}
    for e in eps:
        if e["kind"] == "door":
            s["closed_door" if door_state(e["door"], e["t0"]) == "closed" else "open_door"] += 1
        else:
            s[e["kind"]] += 1
    return s
