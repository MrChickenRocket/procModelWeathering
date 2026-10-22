"""Generate a test pistol following CONVENTIONS.md.

blender -b --factory-startup -P tools/make_test_pistol.py -- <out.blend> [preview_dir]
"""
import os
import sys

import bpy
from mathutils import Euler, Matrix

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from meshgen import R, boolean, box, build_scene, cyl, render_previews  # noqa: E402

GRIP_XF = Matrix.Translation((0, -18, -4)) @ Euler((R(-18), 0, 0)).to_matrix().to_4x4()

MATERIALS = {
    # name: (base color, metallic, roughness)
    "MAT_cerakote_steel": ((0.20, 0.22, 0.18, 1), 0.0, 0.55),
    "MAT_polymer": ((0.05, 0.05, 0.05, 1), 0.0, 0.65),
    "MAT_steel_bare": ((0.55, 0.55, 0.57, 1), 1.0, 0.35),
    "MAT_steel_blued": ((0.09, 0.10, 0.13, 1), 1.0, 0.30),
}


def build_slide():
    s = box("slide", (25.5, 186, 28), (0, 48, 14))
    boolean(s, [
        box("rear_sight", (20, 6, 5), (0, -38, 30.5)),
        box("front_sight", (3.5, 5, 5), (0, 132, 30.5)),
    ], 'UNION')
    cut = [
        box("rear_notch", (3.5, 10, 3), (0, -38, 33)),
        box("cavity", (14, 148, 16), (0, 52, 14)),
        box("channel", (17, 200, 20), (0, 48, 0)),
        cyl("bore", 6.0, 30, (0, 141, 15)),
        box("port", (14, 34, 14), (7, 35, 26)),
        box("nose", (40, 6, 6), (0, 141, 28), rot=(45, 0, 0)),
    ]
    for sx in (-1, 1):
        cut.append(box("chamfer", (4, 200, 4), (sx * 12.75, 48, 28), rot=(0, 45, 0)))
        for i in range(8):
            cut.append(box("serr_r", (2.4, 1.2, 22), (sx * 12.75, -40 + i * 2.5, 13), rot=(12, 0, 0)))
        for i in range(5):
            cut.append(box("serr_f", (2.4, 1.2, 22), (sx * 12.75, 100 + i * 2.5, 13), rot=(12, 0, 0)))
    boolean(s, cut, 'DIFFERENCE')
    return s


def build_frame():
    f = box("frame", (23, 173, 16), (0, 41.5, -8))
    grip = box("grip", (30, 44, 116), (0, 0, -50), parent=GRIP_XF)
    boolean(grip, [box("grip_clip", (100, 200, 50), (0, 0, 23))], 'DIFFERENCE')
    boolean(f, [
        box("rail", (16, 165, 8), (0, 42, 3)),
        grip,
        box("beavertail", (22, 16, 8), (0, -50, -8)),
        box("guard", (10, 46, 32), (0, 27, -26)),
    ], 'UNION')
    cut = [
        box("magwell", (19, 33, 90), (0, 0, -65), parent=GRIP_XF),
        box("guard_hole", (20, 40, 21), (0, 26, -27.5)),
    ]
    # Trigger is unioned after the cuts; guard_hole would otherwise remove it.
    for i in range(3):
        cut.append(cyl("finger", 6, 40, (0, 26, -30 - i * 20), rot=(0, 90, 0), parent=GRIP_XF))
        cut.append(box("rail_slot", (30, 3, 3), (0, 90 + i * 10, -16)))
    for sx in (-1, 1):
        cut.append(box("rail_groove", (3, 50, 3), (sx * 11.5, 104, -10)))
    boolean(f, cut, 'DIFFERENCE')
    boolean(f, [box("trigger", (6, 5, 21), (0, 20, -26.5), rot=(15, 0, 0))], 'UNION')
    return f


def build_barrel():
    b = cyl("barrel", 5.8, 93, (0, 93.5, 15))
    boolean(b, [box("hood", (12.5, 30, 12.5), (0, 32, 15))], 'UNION')
    boolean(b, [cyl("bore", 4.5, 40, (0, 140, 15))], 'DIFFERENCE')
    return b


def build_magazine():
    m = box("magazine", (17, 31, 96), (0, 0, -62), parent=GRIP_XF)
    boolean(m, [box("baseplate", (22, 36, 6), (0, 0, -111), parent=GRIP_XF)], 'UNION')
    return m


WEAR_VOLUMES = [
    # name, center (mm), half extents (mm), parent frame, weight
    ("WEAR_grip", (0, 0, -55), (17, 26, 58), GRIP_XF, 1.0),
    ("WEAR_mag_base", (0, 0, -111), (13, 20, 6), GRIP_XF, 0.6),
    ("WEAR_serrations_rear", (0, -31, 14), (15, 12, 12), None, 0.8),
    ("WEAR_serrations_front", (0, 106, 14), (15, 8, 12), None, 0.4),
    ("WEAR_holster", (0, 60, 5), (16, 85, 26), None, 0.3),
]


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    out_blend = os.path.abspath(argv[0] if argv else "assets/test/pistol.blend")
    preview_dir = os.path.abspath(argv[1]) if len(argv) > 1 else None

    cols = build_scene("pistol", MATERIALS, [
        ("slide", build_slide, "MAT_cerakote_steel", None),
        ("frame", build_frame, "MAT_polymer", None),
        ("barrel", build_barrel, "MAT_steel_bare", None),
        ("magazine", build_magazine, "MAT_steel_blued", None),
    ], WEAR_VOLUMES)

    os.makedirs(os.path.dirname(out_blend), exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=out_blend)
    print(f"[pistol] saved {out_blend}")

    if preview_dir:
        render_previews(preview_dir, cols[1], cols[0], cols[2], (0, 0.045, -0.05), 0.32, {
            "right": (0.5, 0.045, -0.05), "left_rear_3q": (-0.35, -0.3, 0.15), "right_front_3q": (0.3, 0.35, 0.12)})


main()
