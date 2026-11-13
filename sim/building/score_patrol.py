"""Score a run-3 patrol tour (no markers; Madelief 2026-11-11: found = right position AND a right side/type report).

    python3 sim/building/score_patrol.py runs/tours/<run> [...]

Per real door of the building: found = some candidate matched it (patrol.match, within 0.5 m) and its door step got a right report
(any try); first try = the first report was right. Per door side the same. Candidates that match no real door = false doors
(the robot ran a door step at a wall / opening that isn't a door). Doors the robot never put a candidate on = missed."""
import re, json, pathlib, sys

REPO = pathlib.Path(__file__).resolve().parents[2]


def score(run):
    t = json.loads((run / "tour.json").read_text())
    pl = t.get("patrol")
    if pl is None:
        return None
    split, bid = re.match(r"bld_([a-z]+)_(\d{3})", run.name).groups()   # bld_train_007_..., bld_test_100_...
    b = json.loads((REPO / f"sim/building/out/bld_{split}_{bid}.building.json").read_text())
    steps = {v.get("cand"): v.get("door_step") or {} for v in t["visits"] if v.get("cand") is not None}
    sides = {}                                                  # (door, side) -> {"first": bool, "right": bool}
    false = 0
    for k, c in enumerate(pl["candidates"]):
        if not c.get("match"):
            false += c.get("status") == "false door"
            continue
        st = steps.get(k, {})
        key = tuple(c["match"])
        cur = sides.setdefault(key, {"first": False, "right": False, "cands": 0})
        cur["cands"] += 1
        cur["first"] |= bool(st.get("first_right"))
        cur["right"] |= bool(st.get("right"))
    ids = [d["id"] for d in b["doors"]]
    found = [i for i in ids if any(s["right"] for (d, _), s in sides.items() if d == i)]
    first = [i for i in ids if any(s["first"] for (d, _), s in sides.items() if d == i)]
    seen = [i for i in ids if any(d == i for d, _ in sides)]
    return {"run": run.name, "doors": len(ids), "found": len(found), "found_first": len(first), "located_only": len(set(seen) - set(found)),
            "missed": sorted(set(ids) - set(seen)), "false_doors": false, "candidates": len(pl["candidates"]),
            "viewpoints": len(pl["viewpoints"]), "sides_right": sum(s["right"] for s in sides.values()), "sides_located": len(sides)}


def main():
    for run in map(pathlib.Path, sys.argv[1:]):
        r = score(run)
        if r is None:
            print(f"{run.name}: not a patrol run"); continue
        print(f"{r['run']}: found {r['found']}/{r['doors']} doors ({r['found_first']} on the first try), "
              f"{r['located_only']} located but no right report, missed {r['missed']}; "
              f"{r['sides_right']}/{r['sides_located']} located sides right; {r['false_doors']} false doors "
              f"({r['candidates']} candidates, {r['viewpoints']} viewpoints)")
        print(json.dumps(r))


if __name__ == "__main__":
    main()
