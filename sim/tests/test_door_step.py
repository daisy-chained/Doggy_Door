# Unit tests for the reactive door step (DoD item C; Madelief 2026-11-11 ~10:30-10:40): at a door side the robot looks, decides
# itself where to look next until it is sure (2b), reports the door; a right report opens the door, a wrong one -> a NEW look first,
# then a new guess (at most N tries). Correct = hinge side right on hinged doors (1a), "sliding" on sliding doors; the fuller
# report (type, hinge side, push / pull) is stored for later (b). Technical rules = Claude decisions (research-based), dev log
# 2026-11-11 reactive door step. No Gazebo needed: python3 -m pytest sim/tests/test_door_step.py
#
# Module sim/doorsim/door_step.py (stdlib only). Sides are IMAGE sides "left" | "right" as in hinge_look.
#   A look = {"edge": "left" | "right" | None (None = a tour stop, not aimed at an edge), "angle": int | None, "seen": set of edges
#   where a hinge passed the checks in that look}. ANSWERS = (("hinged", "left"), ("hinged", "right"), ("sliding", None)).
#
#   door_step.answer(looks, handle_guess, p_slide=None, other_side_hinged=False, wrong=()) -> report
#     report = {"type": "hinged" | "sliding", "hinge_side": "left" | "right" | None, "push_pull": "pull" | "push" | None, "sure": bool}.
#     wrong = answers (type, side) already reported wrong for this door side; they are never given again.
#     1. A hinge seen at an edge e (any look) and ("hinged", e) not wrong -> hinged, side e, "pull" (hinges show on the pull side),
#        sure. Seen at both edges -> the edge seen in more looks; a tie -> handle_guess, else "left".
#     2. Else hinged is believed when other_side_hinged, or a hinge was seen at all (even at an edge now ruled out), or
#        p_slide < 0.5, or p_slide is None (hinged doors are the more common type). Believed and a hinged answer left -> hinged,
#        side = handle_guess if ("hinged", handle_guess) is not wrong, else the other side not wrong ("left" first when
#        handle_guess is None); push_pull = "push" when other_side_hinged and no hinge seen here, else None; sure False.
#     3. Else ("sliding", None) if not wrong (push_pull None, sure False); else the hinged answer that is left (as in 2).
#     Every answer wrong -> None.
#
#   empty_doorway=False (answer and visit keyword): no door box spanned the doorway at any stop, so the classifier got no crop (an
#     open sliding door: its panel is behind the wall). Madelief 2026-11-11 ~12:45 "sliding first": in step 2, when no hinge was seen
#     at all and not other_side_hinged, empty_doorway -> ("sliding", None) first if not wrong (sure False), before p_slide / the
#     hinged default. Today's runs: no crop on 3 sliding sides + 2 false doors, never on the 42 hinged sides.
#   door_step.next_look(looks, handle_guess, free_fn, n_max) -> (edge, angle) | None
#     extra looks = looks with angle not None. None when len(extra looks) >= n_max. g = handle_guess or "left", o = the other side;
#     candidates in order (g, 20), (g, 40), (o, 20), (o, 40), (g, 0), (o, 0) (0 = head-on at that edge); skip those already in
#     looks (same edge and angle) and those with free_fn(edge, angle) False; first one left, else None.
#
#   door_step.visit(stop_looks, look_fn, check_fn, handle_guess, free_fn, n_max=3, p_slide=None, other_side_hinged=False,
#                   empty_doorway=False) -> log
#     stop_looks = the looks from the tour stops (angle None). look_fn(edge, angle) -> set of edges seen, or None when the robot
#     could not get there (that look still counts as taken, with seen = set()). check_fn(report) -> bool (the sim's truth).
#     n_max = extra looks before the first report; every wrong report allows one more (look cap = n_max + wrong reports so far).
#     Loop: report = answer(...). While not report["sure"] and next_look(..., cap) gives a look: take it, re-answer.
#     Then check_fn(report): right -> done. Wrong -> add (type, side) to wrong; a new look is REQUIRED before the next guess
#     (next_look None -> stop); take it, re-answer (again looking more while not sure and looks are left), check again.
#     At most len(ANSWERS) reports (answer() never repeats a wrong one; None -> stop).
#     log = {"reports": [report, ...] (each with "right": bool), "looks": [all looks incl. stops], "first_right": bool,
#            "right": bool (any report right), "n_reports": int}.
#
#   door_step.truth(door_type, edge_pts, hinge_xy) -> ("sliding", None) | ("hinged", side): side = the key of edge_pts
#     ({"left": (x, y), "right": (x, y)}, image sides) nearest to hinge_xy.
#   door_step.is_right(report, truth) -> bool: same type, and for hinged the same hinge_side (push_pull / sure don't count: 1a).
import pathlib, sys
import pytest

SIM = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIM))
from doorsim import door_step as ds


def L(edge=None, angle=None, seen=()):
    return {"edge": edge, "angle": angle, "seen": set(seen)}


ALL_FREE = lambda e, a: True


# --- answer ---
def test_hinge_seen_is_sure():
    r = ds.answer([L(seen={"right"})], handle_guess="left")
    assert r == {"type": "hinged", "hinge_side": "right", "push_pull": "pull", "sure": True}


def test_hinge_seen_both_edges_majority_then_guess():
    looks = [L(seen={"left", "right"}), L("left", 20, {"left"})]
    assert ds.answer(looks, "right")["hinge_side"] == "left"
    assert ds.answer([L(seen={"left", "right"})], "right")["hinge_side"] == "right"
    assert ds.answer([L(seen={"left", "right"})], None)["hinge_side"] == "left"


def test_seen_edge_ruled_out_falls_back_to_other_hinged_side():
    r = ds.answer([L(seen={"left"})], "left", wrong=[("hinged", "left")])
    assert r["type"] == "hinged" and r["hinge_side"] == "right" and not r["sure"]


def test_no_hinge_handle_guess_hinged_by_default():
    r = ds.answer([L()], "right")
    assert r == {"type": "hinged", "hinge_side": "right", "push_pull": None, "sure": False}


def test_no_hinge_no_handle_left_first():
    assert ds.answer([L()], None)["hinge_side"] == "left"


def test_p_slide_decides_without_hinges():
    assert ds.answer([L()], "left", p_slide=0.8) == {"type": "sliding", "hinge_side": None, "push_pull": None, "sure": False}
    assert ds.answer([L()], "left", p_slide=0.2)["type"] == "hinged"


def test_other_side_hinged_means_push_here():
    r = ds.answer([L()], "left", p_slide=0.9, other_side_hinged=True)
    assert r == {"type": "hinged", "hinge_side": "left", "push_pull": "push", "sure": False}


def test_hinge_seen_overrules_p_slide():
    assert ds.answer([L(seen={"left"})], None, p_slide=0.99)["type"] == "hinged"


def test_wrong_answers_never_repeated():
    r = ds.answer([L()], "left", p_slide=0.9, wrong=[("sliding", None)])
    assert r["type"] == "hinged" and r["hinge_side"] == "left"
    r = ds.answer([L()], "left", wrong=[("hinged", "left"), ("hinged", "right")])
    assert r["type"] == "sliding"
    assert ds.answer([L()], "left", wrong=[a for a in ds.ANSWERS]) is None


# --- next_look ---
def test_next_look_order_guess_edge_first():
    assert ds.next_look([L()], "right", ALL_FREE, 3) == ("right", 20)
    assert ds.next_look([L(), L("right", 20)], "right", ALL_FREE, 3) == ("right", 40)
    assert ds.next_look([L(), L("right", 20), L("right", 40)], "right", ALL_FREE, 4) == ("left", 20)


def test_next_look_no_guess_starts_left():
    assert ds.next_look([L()], None, ALL_FREE, 3) == ("left", 20)


def test_next_look_skips_blocked_and_respects_n_max():
    assert ds.next_look([L()], "left", lambda e, a: (e, a) != ("left", 20), 3) == ("left", 40)
    assert ds.next_look([L(), L("left", 20), L("left", 40)], "left", ALL_FREE, 2) is None
    done = [L()] + [L(e, a) for e in ("left", "right") for a in (20, 40)]
    assert ds.next_look(done, "left", ALL_FREE, 9) == ("left", 0)
    assert ds.next_look([L()], "left", lambda e, a: False, 3) is None


# --- visit ---
def test_visit_sure_at_stop_no_extra_looks():
    calls = []
    log = ds.visit([L(seen={"left"})], lambda e, a: calls.append((e, a)) or set(), lambda r: True, "left", ALL_FREE)
    assert calls == [] and log["first_right"] and log["right"] and log["n_reports"] == 1
    assert log["reports"][0]["right"] is True and log["reports"][0]["sure"]


def test_visit_looks_until_sure():
    seen_at = {("left", 20): set(), ("left", 40): {"left"}}
    log = ds.visit([L()], lambda e, a: seen_at.get((e, a), set()), lambda r: r["hinge_side"] == "left", "left", ALL_FREE)
    assert [(l["edge"], l["angle"]) for l in log["looks"]] == [(None, None), ("left", 20), ("left", 40)]
    assert log["reports"][0]["sure"] and log["first_right"]


def test_visit_wrong_then_new_look_then_right():
    # nothing seen anywhere, handle says left, truth = right: 3 looks -> unsure "left" (wrong) -> 1 more look -> "right" (right)
    log = ds.visit([L()], lambda e, a: set(), lambda r: r["hinge_side"] == "right", "left", ALL_FREE, n_max=3)
    assert log["n_reports"] == 2 and not log["first_right"] and log["right"]
    assert [(l["edge"], l["angle"]) for l in log["looks"]] == [(None, None), ("left", 20), ("left", 40), ("right", 20), ("right", 40)]
    assert [r["right"] for r in log["reports"]] == [False, True]


def test_visit_stops_when_no_new_look_possible():
    # wrong report, but no free spot for a new look -> no re-guess
    log = ds.visit([L(seen={"left"})], lambda e, a: set(), lambda r: r["hinge_side"] == "right", "left", lambda e, a: False)
    assert log["n_reports"] == 1 and not log["right"]


def test_visit_new_look_required_before_reguess():
    order = []
    def look(e, a):
        order.append(("look", e, a)); return set()
    def check(r):
        order.append(("check", r["hinge_side"])); return r["hinge_side"] == "right"
    ds.visit([L(seen={"left"})], look, check, "left", ALL_FREE, n_max=3)
    # sure "left" at the stop -> checked wrong -> a look -> then the next check
    assert order[0] == ("check", "left") and order[1][0] == "look" and order[-1] == ("check", "right")


def test_visit_unreachable_look_counts_as_taken():
    log = ds.visit([L()], lambda e, a: None, lambda r: True, "left", ALL_FREE, n_max=2)
    extra = [l for l in log["looks"] if l["angle"] is not None]
    assert len(extra) == 2 and all(l["seen"] == set() for l in extra)


def test_visit_at_most_n_max_reports():
    log = ds.visit([L()], lambda e, a: set(), lambda r: False, None, ALL_FREE, n_max=1, p_slide=0.1)
    assert log["n_reports"] == 3  # only 3 answers exist, each wrong one buys one more look
    assert not log["right"]


def test_truth_and_is_right():
    pts = {"left": (0.0, 1.0), "right": (0.0, -1.0)}
    assert ds.truth("hinged", pts, (0.05, 0.9)) == ("hinged", "left")
    assert ds.truth("hinged", pts, (0.0, -0.95)) == ("hinged", "right")
    assert ds.truth("sliding", pts, None) == ("sliding", None)
    rep = {"type": "hinged", "hinge_side": "left", "push_pull": "push", "sure": False}
    assert ds.is_right(rep, ("hinged", "left")) and not ds.is_right(rep, ("hinged", "right")) and not ds.is_right(rep, ("sliding", None))
    assert ds.is_right({"type": "sliding", "hinge_side": None, "push_pull": None, "sure": False}, ("sliding", None))


# --- empty doorway -> sliding first ---
def test_empty_doorway_says_sliding_first():
    assert ds.answer([L()], "left", empty_doorway=True) == {"type": "sliding", "hinge_side": None, "push_pull": None, "sure": False}
    assert ds.answer([L()], None, p_slide=0.1, empty_doorway=True)["type"] == "sliding"


def test_empty_doorway_loses_to_hinges_and_other_side():
    assert ds.answer([L(seen={"right"})], None, empty_doorway=True)["type"] == "hinged"
    assert ds.answer([L()], "left", other_side_hinged=True, empty_doorway=True)["type"] == "hinged"
    assert ds.answer([L()], "left", wrong=[("sliding", None)], empty_doorway=True)["hinge_side"] == "left"


def test_visit_empty_doorway_right_first():
    log = ds.visit([L()], lambda e, a: set(), lambda r: r["type"] == "sliding", None, ALL_FREE, empty_doorway=True)
    assert log["first_right"] and log["n_reports"] == 1


# --- confidence option C (Madelief 2026-11-13, "A+b+c"): an unfamiliar-looking door's slide guess is not trusted ---
# door_step.p_slide_trusted(frames) -> {"p": mean | None, "used": int, "unfamiliar": int}: frames = [{"p": P(slide) | None,
#   "unfamiliar": bool}] from the property model per stop frame. Mean over frames with a p that are NOT unfamiliar; "p" None when
#   none is left, so the door step falls back to its looks and trying the door (rule 2 above: p_slide None -> hinged first).
#   "unfamiliar" = number of frames with a p that were flagged (frames without a p are not counted anywhere).
def test_p_slide_trusted_drops_unfamiliar_frames():
    fr = [{"p": 0.9, "unfamiliar": True}, {"p": 0.2, "unfamiliar": False}, {"p": 0.4, "unfamiliar": False}, {"p": None, "unfamiliar": True}]
    r = ds.p_slide_trusted(fr)
    assert r["p"] == pytest.approx(0.3) and r["used"] == 2 and r["unfamiliar"] == 1


def test_p_slide_trusted_all_unfamiliar_or_empty():
    assert ds.p_slide_trusted([{"p": 0.95, "unfamiliar": True}]) == {"p": None, "used": 0, "unfamiliar": 1}
    assert ds.p_slide_trusted([]) == {"p": None, "used": 0, "unfamiliar": 0}
    assert ds.p_slide_trusted([{"p": 0.7, "unfamiliar": False}])["p"] == pytest.approx(0.7)
