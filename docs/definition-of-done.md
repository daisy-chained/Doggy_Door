# Definition of Done + "Good Enough"

**Project:** Doggy Door: Perception for Robot-Door Interaction
**Date:** 2026-11-08 (draft, Claude; numbers are research-based proposals, Madelief can overrule any of them)
**Goes with:** `docs/plan-of-approach.md` (goal, sub-questions, phases) and `docs/document-requirements.md` §4 (which deliverables must exist).

`document-requirements.md` §4 says *which deliverables* must exist. This file adds the other half: **how good the system has to be** before a phase counts as done, so the project can't drift forever (the scope-creep risk in the plan of approach).

---

## 1. The target: an average Dutch house

The project goal (plan of approach) is a perception system that finds doors and handles, tells their properties and gives an honest confidence. My personal goal on top of that: **a robot that drives through an average Dutch house on its own**. So the test world is modelled on one.

What "average Dutch house" means here:

| Thing | Value used | Source / why |
|---|---|---|
| Type | Terraced house (*rijtjeshuis*), corridor + rooms | The Netherlands has more people in terraced houses than anywhere in Europe ([ABN AMRO](https://www.abnamro.nl/nl/zakelijk/insights/sectoren-en-trends/bouw/rijtjeshuis-als-oorzaak-van-het-woningtekort.html)) |
| Size | ~120 m² over 2–3 floors | CBS average dwelling ~120 m² ([CBS 82550NED](https://www.cbs.nl/nl-nl/cijfers/detail/82550NED)) |
| Rooms on one floor | Hall, toilet, living room, kitchen (ground floor); 3 bedrooms + bathroom (upstairs) | Typical terraced-house layout |
| Inner doors | ~83 × 201.5 cm leaf, free passage ≥ 85 cm (new builds) | Bouwbesluit / Bbl: free passage 0.85 × 2.3 m for dwellings ([omgevingsweb](https://omgevingsweb.nl/wetgeving/bouwbesluit-2012/hoofdstuk-4-technische-bouwvoorschriften-uit-het-oogpunt-van-bruikbaarheid/afdeling-4-4-bereikbaarheid-en-toegankelijkheid-nieuwbouw/artikel-4-23-vrije-doorgang-verkeersroute/)); older houses narrower (~78–80 cm) |
| Door sills | ≤ 2 cm | Bbl max for the main entrance ([IPLO](https://iplo.nl/thema/toegankelijkheid-personen/toegankelijkheid-bouwwerken/)); older inner sills can be higher |
| Door mix | Mostly hinged with lever handles, often with glass panels; living-room glass doors; sometimes sliding (*kamer en suite*) | Matches our v6 door types |
| Stairs | Steep, narrow | **A wheeled base can't climb them → see open choice A** |

The robot used in the tests: our own simple base with the sensor rig (Key Decision 2026-11-07), assumed ≤ 50 cm wide. In an 85 cm doorway that leaves ~17 cm each side, which sets the 3D accuracy target below.

---

## 2. Good-enough numbers per phase

Scored in the sim on the **test split** (and the held-out door combos), on **detector boxes** unless noted. "Hard conditions" = the light and smoke level sets (5 % light with headlight, smoke visibility 4 m).

### 2a. Door detection + 3D position

| Metric | Good enough | Now (2026-11-08) | Why this number |
|---|---|---|---|
| Door AP (clear) | ≥ 0.85 | ~0.90 mAP overall | Sim data is clean; real-world detectors sit far lower (DoorDetect ~0.45), so the sim bar must be high to leave room for the sim-to-real drop |
| Door recall in hard conditions | ≥ 0.80 | measured per level, mixed | A missed door = a room the robot never explores |
| Door centre 3D error, closed doors ≤ 3 m | ≤ 10 cm median | ~11–14 cm | 17 cm margin each side of the robot in an 85 cm doorway |
| Door centre 3D error at the 1 m stop | ≤ 5 cm median | not measured | Needed to line up and pass through |
| Windows | detected **or** scoped out with a reason | not built | **Open choice B** |

### 2b. Handles

| Metric | Good enough | Now | Why |
|---|---|---|---|
| Handle found at the 1 m stop | ≥ 0.95 | ~0.9 recall up to 4 m | At the stop the robot is close; a miss there blocks opening |
| Handle type (lever / knob / flush) on found handles | ≥ 0.95 | 0.97 | Type picks the opening motion |

### 2c. Door properties

| Metric | Good enough | Now | Why |
|---|---|---|---|
| **Closed doors called closed** (safety) | ≥ 0.95 | glass 0.89, solid 0.93 | Madelief's call (2026-11-08): calling a closed door open is the worse mistake |
| Open / closed overall | ≥ 0.85 | 0.86 | Open doors called closed only cost a retry |
| Hinged vs sliding | ≥ 0.95 | ~high, check on v6 | Different opening motion |
| Push / pull: first guess | ≥ 0.85 | 0.79 | |
| Push / pull: right within 2 guesses | ≥ 0.95 | 0.90–0.93 | The opening failsafe tries the next guess |
| Hinge side | ≥ 0.90 | 0.93 | |
| Glass vs solid | ≥ 0.90 | check on v6 | Glass changes how depth sensors see the door |
| Held-out combos (unseen) | within 10 points of seen ones | ~0.26 (old), recheck | Shows it learnt properties, not specific doors |

### 2d. Confidence

| Metric | Good enough | Now | Why |
|---|---|---|---|
| D-ECE per model, clear | ≤ 0.05 | ~0.01 (box-aware) | Common "well calibrated" range |
| D-ECE on every hard level | ≤ 0.10 | single models ok; fused in thick smoke 0.106 | Confidence must stay honest when it can't see well (plan of approach SQ5) |
| Confidence on property models | calibrated the same way | not done | The plan asks it for all three models |
| "Sure" means sure | when confidence ≥ 0.9, right ≥ 0.9 of the time | not measured | So act / look again / ask a human can be decided on it |

### 2e. Integration: the Dutch-house run

| Metric | Good enough | Why |
|---|---|---|
| Test houses | ≥ 10 generated Dutch-house layouts (fixed seeds), each under clear, dark and smoke | Enough to see a pattern, still runs overnight |
| Doors found per house | ≥ 0.95 | |
| Collisions with closed doors | **0** | Hard rule: safety |
| Reachable rooms visited (doors open or opened) | 100 % | The personal goal |
| Each closed door reported with position, properties + ranked guesses | 100 % | What a door-opening step would need |
| Runs fast enough | ≥ 5 Hz on at least one hardware target, minimum hardware named | Plan: a range of hardware |

### 2f. Reporting

Done when `document-requirements.md` §4 is ticked off: report, repo, video, this DoD, dev log summary, evaluation.

---

## 3. The project is done when

1. Every 2a–2e row above meets its number, **or** the report explains why it doesn't and what it would take (a clearly explained miss is a result too).
2. The video shows the robot going through at least one Dutch-house layout in clear and in hard conditions, with the detections on screen.
3. All six deliverables exist (`document-requirements.md` §4).
4. Stop rule against scope creep: new side experiments only start once the phase they belong to meets its numbers, or as a backlog item.

---

## 4. Open choices for Madelief

- **A. Stairs.** A wheeled base can't use Dutch stairs. Options: (1) one floor only, the ground floor = "the house"; (2) each floor tested as its own map (robot placed on it); (3) a stair-capable robot (out of scope for now). Claude's pick: (2).
- **B. Windows.** In the brief and SQ2 but never built. Options: (1) add a window class to the detector; (2) scope out with a reason in the report (the robot drives through doors, not windows). Claude's pick: (1) if it fits before 2e, else (2).
- **C. Opening doors.** The plan stops at perception; "drive through the whole house" needs closed doors opened. Options: (1) doors open in the house test, closed doors only checked for "reported + not hit"; (2) a scripted "door opens" once the robot reports it correctly (fake actuator); (3) real opening (out of scope). Claude's pick: (2).
- **D. Any number above** you want stricter or looser.
