"""Reactive door step (DoD item C, Madelief 2026-11-11: answers 1a + store b, 2b, wrong -> new look then guess again): look at a
door side, pick the next look yourself until sure, report; a wrong report needs a new look before the next guess. Technical
rules = Claude decisions (dev log 2026-11-11 reactive door step). Sides are image sides "left" | "right"; pure, stdlib only."""

ANSWERS = (("hinged", "left"), ("hinged", "right"), ("sliding", None))


def _hinged(side, pull_push):
    return {"type": "hinged", "hinge_side": side, "push_pull": pull_push, "sure": False}


def answer(looks, handle_guess, p_slide=None, other_side_hinged=False, wrong=(), empty_doorway=False):
    """Report {"type", "hinge_side", "push_pull", "sure"} from the looks, never an answer in wrong; None when all are wrong.
    empty_doorway: no door box spanned the doorway at any stop (open sliding door) -> sliding first (Madelief 2026-11-11)."""
    count = {e: sum(e in l["seen"] for l in looks) for e in ("left", "right")}
    left = [s for s in ("left", "right") if ("hinged", s) not in wrong]
    cand = [e for e in left if count[e]]
    if cand:
        top = max(count[e] for e in cand)
        best = [e for e in cand if count[e] == top]
        side = best[0] if len(best) == 1 else (handle_guess if handle_guess in best else "left")
        return {"type": "hinged", "hinge_side": side, "push_pull": "pull", "sure": True}
    any_seen = any(count.values())
    pick = lambda: _hinged(handle_guess if handle_guess in left else left[0],
                           "push" if other_side_hinged and not any_seen else None)
    if empty_doorway and not any_seen and not other_side_hinged and ("sliding", None) not in wrong:
        return {"type": "sliding", "hinge_side": None, "push_pull": None, "sure": False}
    if left and (other_side_hinged or any_seen or p_slide is None or p_slide < 0.5):
        return pick()
    if ("sliding", None) not in wrong:
        return {"type": "sliding", "hinge_side": None, "push_pull": None, "sure": False}
    return pick() if left else None


def next_look(looks, handle_guess, free_fn, n_max):
    """Next extra look (edge, angle): guess edge 20, 40, other edge 20, 40, head-on; skips taken / blocked; None at n_max or none left."""
    if sum(l["angle"] is not None for l in looks) >= n_max:
        return None
    g = handle_guess or "left"
    o = "right" if g == "left" else "left"
    taken = {(l["edge"], l["angle"]) for l in looks}
    for c in ((g, 20), (g, 40), (o, 20), (o, 40), (g, 0), (o, 0)):
        if c not in taken and free_fn(*c):
            return c
    return None


def visit(stop_looks, look_fn, check_fn, handle_guess, free_fn, n_max=3, p_slide=None, other_side_hinged=False, empty_doorway=False):
    """Run one door side: look until sure, report, check; wrong -> new look, guess again. Returns {"reports", "looks", "first_right", "right", "n_reports"}."""
    looks, wrong, reports = list(stop_looks), [], []

    def look():
        c = next_look(looks, handle_guess, free_fn, n_max + len(wrong))
        if c is None:
            return False
        looks.append({"edge": c[0], "angle": c[1], "seen": look_fn(*c) or set()})
        return True

    def ask():
        return answer(looks, handle_guess, p_slide, other_side_hinged, wrong, empty_doorway)

    while len(reports) < len(ANSWERS):
        if wrong and not look():
            break
        rep = ask()
        while rep is not None and not rep["sure"] and look():
            rep = ask()
        if rep is None:
            break
        rep = dict(rep, right=bool(check_fn(rep)))
        reports.append(rep)
        if rep["right"]:
            break
        wrong.append((rep["type"], rep["hinge_side"]))
    return {"reports": reports, "looks": looks, "first_right": bool(reports) and reports[0]["right"],
            "right": any(r["right"] for r in reports), "n_reports": len(reports)}


def truth(door_type, edge_pts, hinge_xy):
    """("sliding", None) or ("hinged", side): side = the edge_pts key nearest to hinge_xy."""
    if door_type == "sliding":
        return ("sliding", None)
    import math
    side = min(edge_pts, key=lambda k: math.hypot(edge_pts[k][0] - hinge_xy[0], edge_pts[k][1] - hinge_xy[1]))
    return ("hinged", side)


def is_right(report, truth):
    """Same type and, for hinged, same hinge side."""
    if report["type"] != truth[0]:
        return False
    return truth[0] == "sliding" or report["hinge_side"] == truth[1]


def p_slide_trusted(frames):
    """frames = [{"p": P(slide) | None, "unfamiliar": bool}] -> {"p": mean over frames with a p that are not unfamiliar | None, "used", "unfamiliar"}.
    p None when nothing is left: the door step then falls back to its looks and trying the door (Madelief 2026-11-13, option C)."""
    withp = [f for f in frames if f.get("p") is not None]
    keep = [f["p"] for f in withp if not f.get("unfamiliar")]
    return {"p": sum(keep) / len(keep) if keep else None, "used": len(keep), "unfamiliar": len(withp) - len(keep)}
