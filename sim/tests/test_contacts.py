# Collisions (DoD: "collisions with closed doors = 0", hard safety rule; Madelief 2026-11-13 order: dark + smoke patrols -> missing
# DoD numbers). Measured with a Gazebo contact sensor on the robot body (Claude decision 2026-11-14, research-based: the sim's own
# contact physics, not a geometric re-check; gz-sim Contact system + <sensor type="contact">). The wheels / casters are not watched
# (they touch the floor all the time); the body box starts 0.06 m above the floor, so it only touches walls, doors and clutter.
# No Gazebo needed: python3 -m pytest sim/tests/test_contacts.py
#
# robot.robot_sdf(p, name, ...): base_link gets <sensor name="bumper" type="contact"> with <contact><collision>collision</collision>
#   <topic>/<name>/contacts</topic></contact>, update_rate 20, always_on; the model gets the plugin gz-sim-contact-system
#   (gz::sim::systems::Contact). Always on (it only measures).
# Module sim/doorsim/contacts.py (stdlib only):
#   kind(scoped_name) -> {"kind": "door" | "wall" | "clutter" | "other", "door": int | None, "model": str}: the other side of a
#     contact, from its scoped collision name "<world>::<model>::<link>::<collision>" (model = 2nd part when there are >= 3 parts,
#     else the 1st). Models: door_<id>_panel / door_<id>_panel_b / door_<id>_frame / door_<id>_handle_* / door_<id>_hinge_* ->
#     door <id> ("door"); "walls" -> "wall"; clutter_<n> -> "clutter"; anything else "other".
#   episodes(events, gap=0.5) -> [{"t0", "t1", "model", "kind", "door", "n"}]: events = [(t seconds, scoped name of the OTHER
#     collision)] in time order; one episode per model per run of touches with no break longer than `gap` s (n = events in it).
#     Episodes come out sorted by t0.
#   other_side(c1, c2, robot="robot") -> the name of the two that is not the robot's (scoped "...::robot::..." or starting
#     "robot::"); None when both or neither are the robot's.
#   summary(eps, door_state) -> {"episodes", "closed_door", "open_door", "wall", "clutter", "other"}: counts of episodes;
#     door_state(door_id, t) -> "closed" | "open" decides closed_door vs open_door.
import pathlib, sys, xml.etree.ElementTree as ET
import pytest

SIM = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIM))
from doorsim import contacts, robot

PLAT = robot.load_platform(SIM / "configs" / "robots" / "wheeled_ring.yaml")


def test_robot_has_bumper_on_the_body():
    m = ET.fromstring(robot.robot_sdf(PLAT, "robot")).find("model")
    s = m.find("link[@name='base_link']/sensor[@name='bumper']")
    assert s is not None and s.get("type") == "contact"
    assert s.findtext("contact/collision") == "collision"
    assert s.findtext("contact/topic") == "/robot/contacts"
    assert float(s.findtext("update_rate")) == 20
    assert any(pl.get("filename") == "gz-sim-contact-system" and pl.get("name") == "gz::sim::systems::Contact" for pl in m.iter("plugin"))
    assert m.find("link[@name='base_link']/collision[@name='collision']") is not None


def test_bumper_topic_follows_the_name():
    m = ET.fromstring(robot.robot_sdf(PLAT, "bot2")).find("model")
    assert m.find("link[@name='base_link']/sensor[@name='bumper']").findtext("contact/topic") == "/bot2/contacts"


@pytest.mark.parametrize("name, want", [
    ("bld_train_005::door_3_panel::panel::c_leaf", ("door", 3, "door_3_panel")),
    ("bld::door_12_panel_b::panel::c", ("door", 12, "door_12_panel_b")),
    ("bld::door_7_frame::link::trim", ("door", 7, "door_7_frame")),
    ("bld::door_7_handle_front::link::c", ("door", 7, "door_7_handle_front")),
    ("bld::door_2_hinge_1::link::c", ("door", 2, "door_2_hinge_1")),
    ("bld::walls::link::w12", ("wall", None, "walls")),
    ("bld::clutter_4::link::c0", ("clutter", None, "clutter_4")),
    ("bld::floor::link::floor", ("other", None, "floor")),
    ("walls::link::w1", ("wall", None, "walls")),
])
def test_kind(name, want):
    k = contacts.kind(name)
    assert (k["kind"], k["door"], k["model"]) == want


def test_other_side():
    r = "bld::robot::base_link::collision"
    assert contacts.other_side(r, "bld::walls::link::w") == "bld::walls::link::w"
    assert contacts.other_side("bld::walls::link::w", r) == "bld::walls::link::w"
    assert contacts.other_side("robot::base_link::collision", "bld::walls::link::w") == "bld::walls::link::w"
    assert contacts.other_side(r, r) is None
    assert contacts.other_side("bld::walls::link::w", "bld::floor::link::f") is None


def test_episodes_merge_and_split():
    ev = [(1.0, "b::walls::l::w"), (1.1, "b::walls::l::w"), (1.3, "b::walls::l::w2"),   # same model -> one
          (1.2, "b::door_3_panel::panel::c"),
          (3.0, "b::walls::l::w"),                                                     # gap > 0.5 -> new episode
          (3.2, "b::door_3_panel::panel::c")]
    ev.sort()
    eps = contacts.episodes(ev, gap=0.5)
    got = [(e["model"], e["t0"], e["t1"], e["n"], e["kind"], e["door"]) for e in eps]
    assert got == [("walls", 1.0, 1.3, 3, "wall", None), ("door_3_panel", 1.2, 1.2, 1, "door", 3),
                   ("walls", 3.0, 3.0, 1, "wall", None), ("door_3_panel", 3.2, 3.2, 1, "door", 3)]
    assert contacts.episodes([]) == []


def test_summary_splits_closed_and_open_doors():
    eps = contacts.episodes([(1.0, "b::door_3_panel::p::c"), (5.0, "b::door_3_panel::p::c"), (6.0, "b::door_4_frame::l::c"),
                             (7.0, "b::walls::l::w"), (8.0, "b::clutter_1::l::c"), (9.0, "b::floor::l::f")])
    state = lambda d, t: "closed" if (d == 3 and t < 4) else "open"
    assert contacts.summary(eps, state) == {"episodes": 6, "closed_door": 1, "open_door": 2, "wall": 1, "clutter": 1, "other": 1}
