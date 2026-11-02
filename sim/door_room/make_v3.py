"""Builds door_room_v3 from door_room_v2: the room closed (ceiling + walls, no sky), no ambient light, a spot light
`headlight` on the camera rig (starts off), the sun without shadows (the ceiling would block it),
and a top track + hangers for sliding door B (without them the panel looks shifted off its frame).
The attributes file is v2's. Run: python3 sim/door_room/make_v3.py"""
import json, pathlib, re

ROOM = pathlib.Path(__file__).resolve().parent
WALL, CEILING, METAL = "0.85 0.83 0.78 1", "0.9 0.9 0.88 1", "0.15 0.15 0.15 1"

# (name, colour, x0, x1, y0, y1, z0, z1): room x -2..6.5, y -3.1..3.1, ceiling top 2.8
BOXES = [("ceiling", CEILING, -2.0, 6.5, -3.1, 3.1, 2.7, 2.8),
         ("wall_left", WALL, -2.0, 6.5, 3.0, 3.1, 0.0, 2.7),
         ("wall_right", WALL, -2.0, 6.5, -3.1, -3.0, 0.0, 2.7),
         ("wall_behind_camera", WALL, -2.1, -2.0, -3.1, 3.1, 0.0, 2.8),
         ("wall_far", WALL, 6.5, 6.6, -3.1, 3.1, 0.0, 2.8),
         ("above_back_wall", WALL, 2.95, 3.05, -3.0, 3.0, 2.6, 2.7),
         # door B's top track, on the wall face over the opening and the slide path (panel: y 0.51..1.49, slides +0.9)
         ("slide_rail", METAL, 2.90, 2.94, 0.45, 2.45, 2.15, 2.21)]


def shell():
    out = ['    <!-- closed room: ceiling, side walls, end walls and the gap above the back wall (background, no label) -->',
           '    <model name="room_shell">', '      <static>true</static>', '      <link name="link">']
    for name, colour, x0, x1, y0, y1, z0, z1 in BOXES:
        pose, size = [(a + b) / 2 for a, b in ((x0, x1), (y0, y1), (z0, z1))], [x1 - x0, y1 - y0, z1 - z0]
        geom = f'<geometry><box><size>{" ".join(f"{v:g}" for v in size)}</size></box></geometry>'
        out.append(f'        <visual name="{name}"><pose>{" ".join(f"{v:g}" for v in pose)} 0 0 0</pose>{geom}'
                   f'<material><ambient>{colour}</ambient><diffuse>{colour}</diffuse></material></visual>')
        out.append(f'        <collision name="{name}"><pose>{" ".join(f"{v:g}" for v in pose)} 0 0 0</pose>{geom}</collision>')
    return "\n".join(out + ["      </link>", "    </model>"]) + "\n"


# two hangers on door B's panel (panel link frame: top edge at z 1.06) reaching up into the track; part of the door
HANGERS = "".join(f'<visual name="hanger_{i}"><pose>0 {y} 1.095 0 0 0</pose><geometry><box><size>0.02 0.06 0.07</size></box></geometry>'
                  f'<material><ambient>{METAL}</ambient><diffuse>{METAL}</diffuse></material></visual>\n' for i, y in enumerate((-0.35, 0.35)))

HEADLIGHT = """        <light type="spot" name="headlight">
          <pose>0 0 0.05 0 0 0</pose>
          <cast_shadows>true</cast_shadows>
          <diffuse>0 0 0 1</diffuse>
          <specular>0.1 0.1 0.1 1</specular>
          <attenuation><range>10</range><constant>0.2</constant><linear>0.05</linear><quadratic>0.02</quadratic></attenuation>
          <direction>1 0 0</direction>
          <spot><inner_angle>0.4</inner_angle><outer_angle>0.8</outer_angle><falloff>1</falloff></spot>
        </light>
"""

sdf = (ROOM / "door_room_v2.sdf").read_text().replace('<world name="door_room_v2">', '<world name="door_room_v3">')
sdf = re.sub(r"\s*<scene>.*?</scene>", "", sdf, flags=re.S)
sdf = sdf.replace('<world name="door_room_v3">\n', '<world name="door_room_v3">\n    <scene><ambient>0 0 0 1</ambient><background>0 0 0 1</background></scene>\n', 1)
head, sun = sdf.split('<light type="directional" name="sun">', 1)
sdf = head + '<light type="directional" name="sun">' + sun.replace("<cast_shadows>true</cast_shadows>", "<cast_shadows>false</cast_shadows>", 1)
rig = sdf.index('<model name="rig">')
sdf = sdf[:rig] + sdf[rig:].replace("      <link name=\"link\">\n", "      <link name=\"link\">\n" + HEADLIGHT, 1)
door_b = sdf.index('<model name="door_b">')
panel_end = sdf.index("</link>", sdf.index('<link name="panel">', door_b))
sdf = sdf[:panel_end] + HANGERS + "      " + sdf[panel_end:]
sdf = sdf.replace("  </world>", "\n" + shell() + "  </world>")
(ROOM / "door_room_v3.sdf").write_text(sdf)

attrs = json.loads((ROOM / "door_room_v2.attributes.json").read_text())
attrs["_note"] = (attrs["_note"].replace("door_room_v2", "door_room_v3")
                  + " door_room_v3 = v2 in a closed room (room_shell: ceiling + walls, no id / label = background) with a spot light "
                  "'headlight' on the camera rig (off until the generator turns it on) and no ambient light.")
(ROOM / "door_room_v3.attributes.json").write_text(json.dumps(attrs, indent=2) + "\n")
