"""Generate an AR-15 pattern test carbine following CONVENTIONS.md.

blender -b --factory-startup -P tools/make_test_rifle.py -- <out.blend> [preview_dir]

Bore axis runs along +Y at z = 50 mm; the upper/lower parting line is z = 25 mm.
"""
import math
import os
import sys

import bpy
from mathutils import Matrix

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from meshgen import (R, arc, boolean, box, build_scene, cyl, prism, render_previews,  # noqa: E402
                     rot_x, rot_y, stadium, text)

BORE_Z = 50
GRIP_XF = Matrix.Translation((0, -6, 0)) @ rot_x(-25)

# Magazine curve: front/rear edges are arcs about (MAG_CY, MAG_CZ), straight above MAG_CZ.
MAG_Y0, MAG_Y1 = 81, 143
MAG_CZ = -40
MAG_R_REAR = 420
MAG_CY = MAG_Y0 + MAG_R_REAR
MAG_R_FRONT = MAG_CY - MAG_Y1
MAG_SWEEP = 15

# Text, left side: reads with the muzzle on the viewer's left. Columns map text X, Y, Z (depth).
LEFT = Matrix(((0, 0, -1, 0), (-1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 0, 1)))
RIGHT = Matrix(((0, 0, 1, 0), (1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 0, 1)))

MATERIALS = {
    "MAT_anodized_alu": ((0.03, 0.03, 0.032, 1), 1.0, 0.45),
    "MAT_anodized_alu_grey": ((0.10, 0.10, 0.095, 1), 1.0, 0.5),
    "MAT_steel_parkerized": ((0.04, 0.042, 0.038, 1), 0.0, 0.75),
    "MAT_polymer": ((0.05, 0.05, 0.05, 1), 0.0, 0.65),
}


def rail(name, y0, y1, z_base):
    """Picatinny rail with cross slots on a 10 mm pitch."""
    top = BORE_Z + 23.7
    prof = [(-8, z_base), (8, z_base), (8, top - 6.2), (10.6, top - 3.6), (10.6, top - 2.6), (8, top),
            (-8, top), (-10.6, top - 2.6), (-10.6, top - 3.6), (-8, top - 6.2)]
    r = prism(name, prof, y0, y1, axis='Y')
    first = math.ceil((y0 + 4) / 10) * 10
    slots = [box("rail_slot", (30, 5.23, 6), (0, y, top)) for y in range(first, int(y1) - 3, 10)]
    return r, slots


def build_upper():
    u = box("upper", (24, 172, 27), (0, 64, 38.5))
    r, slots = rail("upper_rail", -22, 150, 58)
    boolean(u, [
        cyl("upper_tube", 13.5, 172, (0, 64, BORE_Z)),
        r,
        cyl("fa_housing", 7.5, 36, (11.5, 0, 56), rot=(90, 0, 4)),
        cyl("fa_button", 6, 7, (12.2, -21, 56), rot=(90, 0, 4)),
        prism("deflector", [(11, 26), (14, 26), (18.5, 33), (18.5, 41), (11, 41)], 44, 61, axis='Z'),
    ], 'UNION')
    cuts = slots + [
        box("ch_channel", (13, 20, 6), (0, -13, 69)),
        box("ejection_port", (14, 54, 17), (13.5, 69, 46.5)),
        box("rear_relief", (30, 6, 10), (0, -22, 30), rot=(-35, 0, 0)),
    ]
    cuts += [box("fa_groove", (1.2, 3, 14), (12.2, -24.4, 56), rot=(0, a, 0)) for a in (0, 60, 120)]
    boolean(u, cuts, 'DIFFERENCE')
    boolean(u, [cyl("cover_hinge", 1.6, 62, (12.3, 69, 37.6), segments=16)], 'UNION')
    return u


def upper_details():
    return [text("t_upper", "PROC ARMS  CAL .223 / 5.56", 2.6, 0.2, Matrix.Translation((0, 64, 30)) @ LEFT
                 @ Matrix.Translation((0, 0, 12)))]


def build_lower():
    body = prism("lower", [(-40, 25), (150, 25), (150, 10), (74, 10), (74, 6), (10, 6), (4, 0), (-24, 0),
                           (-40, 12)], -12, 12)
    boolean(body, [
        prism("magwell", [(73, 16), (150, 16), (147.5, -42), (74, -42), (73, -36)], -14.5, 14.5),
        prism("flare", [(74.5, -35), (149.1, -35), (148.1, -42), (74.5, -42)], -15.5, 15.5),
        cyl("tower", 16.5, 18, (0, -31, BORE_Z)),
        box("tower_web", (24, 18, 26), (0, -31, 37)),
        prism("guard", [(75, 8), (75, -20), (70, -24), (18, -24), (8, -10), (6, 0), (6, 8)], -6, 6),
        cyl("pivot_pin", 3.5, 26, (0, 143, 19), rot=(0, 90, 0), segments=24),
        cyl("takedown_pin", 3.5, 26, (0, -32, 19), rot=(0, 90, 0), segments=24),
        cyl("selector", 6.5, 1.6, (-12.6, 10, 16), rot=(0, 90, 0), segments=24),
        box("selector_lever", (1.6, 16, 3.2), (-13.4, 17, 16)),
        box("bolt_catch", (2.4, 11, 7), (-13.2, 76, 20.5)),
        box("bolt_catch_paddle", (3.2, 6, 4), (-14.3, 75, 18), rot=(25, 0, 0)),
        cyl("mag_release", 4.5, 3.4, (13.4, 68, 12), rot=(0, 90, 0), segments=24),
        box("mag_fence", (3, 12, 3), (13.5, 68, 5.5)),
    ], 'UNION')
    boolean(body, [
        box("magwell_in", (22.6, 64, 90), (0, 112, -5)),
        prism("guard_hole", [(68, 8), (68, -16), (64, -18), (20, -18), (14, -8), (13, 8)], -8, 8),
        box("tower_bore", (20, 20, 20), (0, -31, BORE_Z), rot=(0, 45, 0)),
    ], 'DIFFERENCE')
    boolean(body, [prism("trigger", [(40, 9), (44, -2), (44, -10), (40, -15), (37.5, -14), (40, -8), (39, 0),
                                     (36, 9)], -2.8, 2.8)], 'UNION')
    return body


def lower_details():
    return [
        text("t_brand", "PROC-15", 6, 0.25, Matrix.Translation((-14.5, 112, 0)) @ LEFT),
        text("t_cal", "CAL MULTI", 2.4, 0.2, Matrix.Translation((-14.5, 112, -8)) @ LEFT),
        text("t_serial", "SER  PW000417", 2.4, 0.2, Matrix.Translation((-14.5, 112, -14)) @ LEFT),
        text("t_safe", "SAFE", 2.2, 0.2, Matrix.Translation((-12, 0, 21.5)) @ LEFT),
        text("t_fire", "FIRE", 2.2, 0.2, Matrix.Translation((-12, 20, 21.5)) @ LEFT),
    ]


def build_handguard():
    apothem, inner = 19, 16.5
    oct_r = 1 / math.cos(R(22.5))
    h = cyl("handguard", apothem * oct_r, 230, (0, 265, BORE_Z), rot=(90, 22.5, 0), segments=8)
    r, slots = rail("hg_rail", 150, 380, BORE_Z + 16)
    boolean(h, [r], 'UNION')
    cuts = slots + [cyl("hg_bore", inner * oct_r, 240, (0, 265, BORE_Z), rot=(90, 22.5, 0), segments=8)]
    for ang, ys in ((90, (195, 245, 295, 345)), (-90, (195, 245, 295, 345)), (180, (195, 245, 295, 345)),
                    (135, (220, 270, 320)), (-135, (220, 270, 320))):
        for y in ys:
            cuts.append(prism("mlok", stadium(32, 7), apothem - 4, apothem + 2, axis='Z',
                              parent=Matrix.Translation((0, y, BORE_Z)) @ rot_y(ang)))
    for ang in (45, -45):
        cuts.append(box("vent", (4, 8, 8), (0, 375, apothem), parent=Matrix.Translation((0, 0, BORE_Z)) @ rot_y(ang)))
    boolean(h, cuts, 'DIFFERENCE')
    return h


def handguard_details():
    return [text("t_hg", "PROC ARMS", 4, 0.2, Matrix.Translation((-19, 270, BORE_Z)) @ LEFT)]


def build_barrel():
    b = cyl("barrel", 8.3, 268, (0, 284, BORE_Z))
    boolean(b, [
        cyl("shoulder", 10, 4, (0, 418, BORE_Z)),
        cyl("muzzle", 11, 46, (0, 443, BORE_Z)),
        cyl("muzzle_taper", 11, 4, (0, 468, BORE_Z), rot=(-90, 0, 0), radius2=9.5),
    ], 'UNION')
    cuts = [
        cyl("bore", 2.9, 340, (0, 330, BORE_Z), segments=24),
        cyl("cage", 8, 46, (0, 448, BORE_Z)),
    ]
    for a in (-90, -45, 0, 45, 90):
        cuts.append(box("cage_slot", (3.2, 22, 10), (0, 446, 8), parent=Matrix.Translation((0, 0, BORE_Z)) @ rot_y(a)))
    boolean(b, cuts, 'DIFFERENCE')
    return b


def build_charging_handle():
    c = prism("ch", [(-21, -31), (21, -31), (21, -27), (7, -23), (6, -4), (-6, -4), (-7, -23), (-21, -27)],
              66.5, 71.5, axis='Z')
    boolean(c, [prism("latch", [(-22, -32), (-12, -32), (-12, -25), (-20, -26)], 66.5, 73, axis='Z')], 'UNION')
    boolean(c, [box("ch_groove", (1.2, 5, 2), (sx * (14 + i * 2.2), -29, 71.5))
                for sx in (-1, 1) for i in range(3)], 'DIFFERENCE')
    return c


def build_buffer_tube():
    t = cyl("buffer_tube", 14.8, 192, (0, -136, BORE_Z))
    boolean(t, [
        cyl("castle_nut", 17.5, 10, (0, -45, BORE_Z)),
        box("key", (7, 165, 4), (0, -142, BORE_Z - 15.5)),
    ], 'UNION')
    cuts = [box("nut_notch", (4, 4, 40), (0, -48.5, 0), parent=Matrix.Translation((0, 0, BORE_Z)) @ rot_y(a))
            for a in range(0, 180, 45)]
    cuts += [cyl("pos_hole", 1.6, 6, (0, y, BORE_Z - 17), rot=(0, 0, 0), segments=16) for y in range(-215, -135, 13)]
    boolean(t, cuts, 'DIFFERENCE')
    return t


def build_stock():
    s = cyl("stock_housing", 19, 113, (0, -241.5, BORE_Z))
    boolean(s, [
        prism("stock_web", [(-190, 50), (-292, 50), (-292, -30), (-190, 30)], -10, 10),
        box("butt", (42, 14, 106), (0, -299, 19)),
        box("lever", (8, 15, 9), (0, -197, 28)),
    ], 'UNION')
    cuts = [
        prism("window", [(-212, 29), (-280, 29), (-280, -14)], -12, 12),
        cyl("stock_bore", 15.2, 120, (0, -240, BORE_Z)),
        box("sling_slot", (30, 4, 12), (0, -286, 2)),
    ]
    for sx in (-1, 1):
        cuts.append(box("butt_chamfer", (8, 8, 120), (sx * 21, -306, 19), rot=(0, 0, 45)))
    cuts += [box("pad_groove", (50, 1.6, 1.6), (0, -306, z)) for z in range(-28, 70, 5)]
    boolean(s, cuts, 'DIFFERENCE')
    return s


def build_grip():
    g = prism("grip", [(14, 15), (14, -30), (18, -38), (14, -46), (14, -88), (9, -96), (-12, -96), (-17, -89),
                       (-18, -10), (-24, 4), (-24, 15)], -13, 13, parent=GRIP_XF)
    boolean(g, [box("grip_clip", (100, 200, 50), (0, 0, 25))], 'DIFFERENCE')
    boolean(g, [box("grip_pocket", (18, 18, 40), (0, -2, -96), parent=GRIP_XF)], 'DIFFERENCE')
    return g


def grip_details():
    """Diamond checkering on both side panels: V grooves on two diagonals, clipped to the panel."""
    out = []
    for sx in (-1, 1):
        grooves = []
        for ang in (40, -40):
            for i in range(-14, 15):
                m = GRIP_XF @ Matrix.Translation((sx * 13, -1, -55)) @ rot_x(ang) \
                    @ Matrix.Translation((0, 0, i * 2.2)) @ rot_y(45)
                grooves.append(box("groove", (0.7, 120, 0.7), (0, 0, 0), parent=m))
        panel = box("panel", (3, 20, 64), (sx * 13, -1, -55), parent=GRIP_XF)
        first, rest = grooves[0], grooves[1:]
        boolean(first, rest, 'UNION')
        boolean(panel, [first], 'INTERSECT')
        out.append(panel)
    return out


def mag_sector(r0, r1, a0, a1, n=24):
    return arc(MAG_CY, MAG_CZ, r1, 180 + a0, 180 + a1, n) + arc(MAG_CY, MAG_CZ, r0, 180 + a1, 180 + a0, n)


def build_magazine():
    front = arc(MAG_CY, MAG_CZ, MAG_R_FRONT, 180, 180 + MAG_SWEEP, 24)
    rear = arc(MAG_CY, MAG_CZ, MAG_R_REAR, 180 + MAG_SWEEP, 180, 24)
    m = prism("magazine", [(MAG_Y1, 25)] + front + rear + [(MAG_Y0, 25)], -11, 11)
    plate_t = 6 / MAG_R_REAR * 180 / math.pi
    boolean(m, [prism("baseplate", mag_sector(MAG_R_FRONT - 2, MAG_R_REAR + 2, MAG_SWEEP - 0.3,
                                                 MAG_SWEEP + plate_t), -12.5, 12.5)], 'UNION')
    cuts = []
    for sx in (-1, 1):
        for r in (MAG_R_REAR - 15, MAG_R_FRONT + 15):
            cuts.append(prism("rib", mag_sector(r - 1.6, r + 1.6, 1.0, MAG_SWEEP - 1.5),
                              sx * 10.2, sx * 12))
        cuts.append(box("rib_top", (1.8, 31, 3.2), (sx * 11, (MAG_Y0 + MAG_Y1) / 2 - 1, -46)))
    boolean(m, cuts, 'DIFFERENCE')
    return m


WEAR_VOLUMES = [
    # name, center (mm), half extents (mm), parent frame, weight
    ("WEAR_grip", (0, -3, -50), (17, 24, 52), GRIP_XF, 1.0),
    ("WEAR_handguard", (0, 300, BORE_Z), (25, 70, 25), None, 0.9),
    ("WEAR_charging_handle", (0, -28, 70), (24, 7, 6), None, 0.9),
    ("WEAR_controls", (0, 40, 14), (18, 42, 14), None, 0.5),
    ("WEAR_magwell", (0, 112, -38), (18, 42, 6), None, 0.6),
    ("WEAR_mag_body", (0, 118, -85), (15, 40, 40), None, 0.4),
    ("WEAR_mag_base", (0, 0, 0), (16, 36, 12), Matrix.Translation((0, 125.3, -140.7)) @ rot_x(MAG_SWEEP), 0.8),
    ("WEAR_stock_butt", (0, -300, 19), (24, 10, 56), None, 0.7),
    ("WEAR_cheek", (0, -245, 65), (22, 55, 10), None, 0.5),
    ("WEAR_muzzle", (0, 450, BORE_Z), (15, 25, 15), None, 0.4),
    ("WEAR_carry", (0, 90, 30), (28, 320, 50), None, 0.2),
]


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    out_blend = os.path.abspath(argv[0] if argv else "assets/test/rifle.blend")
    preview_dir = os.path.abspath(argv[1]) if len(argv) > 1 else None

    cols = build_scene("rifle", MATERIALS, [
        ("upper", build_upper, "MAT_anodized_alu", upper_details),
        ("lower", build_lower, "MAT_anodized_alu", lower_details),
        ("handguard", build_handguard, "MAT_anodized_alu", handguard_details),
        ("charging_handle", build_charging_handle, "MAT_anodized_alu", None),
        ("buffer_tube", build_buffer_tube, "MAT_anodized_alu", None),
        ("barrel", build_barrel, "MAT_steel_parkerized", None),
        ("stock", build_stock, "MAT_polymer", None),
        ("grip", build_grip, "MAT_polymer", grip_details),
        ("magazine", build_magazine, "MAT_anodized_alu_grey", None),
    ], WEAR_VOLUMES)

    os.makedirs(os.path.dirname(out_blend), exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=out_blend)
    print(f"[rifle] saved {out_blend}")

    if preview_dir:
        render_previews(preview_dir, cols[1], cols[0], cols[2], (0, 0.085, -0.02), 0.85, {
            "right": (1.0, 0.085, -0.02), "left": (-1.0, 0.085, -0.02),
            "left_rear_3q": (-0.6, -0.5, 0.35), "right_front_3q": (0.6, 0.7, 0.3)})


main()
