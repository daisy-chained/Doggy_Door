"""Score a tour run with hinge looks (option A, 2026-11-10) against the sim's true boxes.

    python3 sim/building/score_hinge_look.py runs/tours/<run>

Truth per door side = a true hinge box (label 10 * door + 5..7) of that door in any frame of the visit. "first" = the far stop
(one look, as before), "any" = any look (stops + extra looks). Only looks the robot reached count; a door side whose far stop
was not reached is left out (it never saw the door). Also: how often the handle gave a guess, and the guessed edge vs the
true hinge boxes (right image half / left image half at the far stop)."""
import json, pathlib, sys

SIM = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIM))
from doorsim import hinge_look  # noqa: E402

run = pathlib.Path(sys.argv[1])
tour = json.loads((run / "tour.json").read_text())
visits, guessed = [], 0
for v in tour["visits"]:
    hl = v.get("hinge_look")
    if not hl:
        continue
    hl = {**hl, "looks": [l for l in hl["looks"] if l["result"] == "SUCCEEDED"]}
    if not hl["looks"] or hl["looks"][0]["kind"] != "stop" or hl["looks"][0]["dist"] != v["stops"][0]["dist"]:
        continue
    truth = False
    for l in hl["looks"]:
        if l["frame"]:
            meta = json.loads((run / "stops" / l["frame"] / "meta.json").read_text())
            truth |= any(b["door"] == v["door"] and b["label"] % 10 in (5, 6, 7) for b in meta["boxes"])
    seen = [l["seen"] for l in hl["looks"]]
    visits.append({"truth": truth, "first": seen[0] if seen else False, "looks": seen})
    guessed += hl["guess"] is not None
    extra = [l for l in hl["looks"] if l["kind"] == "look"]
    print(f"door {v['door']:>2} {v['side']:<5} truth {'Y' if truth else '-'} guess {hl['guess'] or '—':<5} "
          f"looks {''.join('Y' if s else '.' for s in seen)}  extra reached {sum(l['result'] == 'SUCCEEDED' for l in extra)}/{len(extra)}")
s = hinge_look.score_visits(visits)
print(f"\ndoor sides {len(visits)}, handle guess on {guessed}")
print(f"hinges found: first look {s['found_first']:.2f} -> all looks {s['found_any']:.2f}  (of {s['n_pos']} with visible hinges)")
print(f"false yes:    first look {s['false_yes_first']:.2f} -> all looks {s['false_yes_any']:.2f}  (of {s['n_neg']} without)")
