"""Pipeline .blend for a square wear tile on a chamfered box face (ArcRoyale MeshKit buildBox, planar UV per face).

blender -b -P tools/make_tile_box.py -- <out.blend> <stack> [--size_studs 1.2 --chamfer 0.08 --m_per_stud 0.050872]

The front (+Z) face maps to [c, 1 - c] of the tile exactly as MeshKit's faceUV; the chamfer strips and corner caps
around it fill the border ring out to the tile edge. Every other face sits off the image (UV 5, 5) so the bake ignores
it, but it stays in the scene for occlusion and edge curvature.
"""
import argparse
import sys

import bpy
import numpy as np

argv = sys.argv[sys.argv.index("--") + 1:]
ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("stack")
ap.add_argument("--size_studs", type=float, default=1.2)
ap.add_argument("--chamfer", type=float, default=0.08)
ap.add_argument("--m_per_stud", type=float, default=0.050872)
args = ap.parse_args(argv)

L = args.size_studs * args.m_per_stud
c = args.chamfer
h = 0.5
corners = [np.array((sx, sy, sz)) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
AX = np.eye(3)


def pos(s, axis):
    k = np.full(3, h - c)
    k[axis] = h
    return s * k


verts, vkey = [], {}


def V(s, axis):
    key = (*s, axis)
    if key not in vkey:
        vkey[key] = len(verts)
        verts.append(pos(s, axis))
    return vkey[key]


faces, fnorm = [], []
for axis in range(3):
    for sign in (-1, 1):
        u, v = (axis + 1) % 3, (axis + 2) % 3
        quad = sorted([s for s in corners if s[axis] == sign], key=lambda s: np.arctan2(s[v], s[u]))
        f = [V(s, axis) for s in quad]
        faces.append(f if sign > 0 else f[::-1])
for e in range(3):
    a1, a2 = (e + 1) % 3, (e + 2) % 3
    for s1 in (-1, 1):
        for s2 in (-1, 1):
            def corner(se):
                t = np.zeros(3)
                t[e], t[a1], t[a2] = se, s1, s2
                return t
            cA, cB = corner(-1), corner(1)
            f = [V(cA, a1), V(cB, a1), V(cB, a2), V(cA, a2)]
            out = AX[a1] * s1 + AX[a2] * s2
            n = np.cross(verts[f[1]] - verts[f[0]], verts[f[2]] - verts[f[0]])
            faces.append(f if n @ out > 0 else f[::-1])
for s in corners:
    f = [V(s, 0), V(s, 1), V(s, 2)]
    n = np.cross(verts[f[1]] - verts[f[0]], verts[f[2]] - verts[f[0]])
    faces.append(f if n @ s > 0 else f[::-1])
axis_of = {i: k[3] for k, i in vkey.items()}
sgn_of = {i: np.array(k[:3]) for k, i in vkey.items()}
P = np.array(verts)


def uv_of(vi, face):
    """Tile UV of a front-region corner, None off the front region."""
    p, s, a = P[vi], sgn_of[vi], axis_of[vi]
    on_front = all(sgn_of[w][2] > 0 for w in face)
    if not on_front:
        return None
    uv = np.array([p[0] + 0.5, p[1] + 0.5])
    if a != 2:
        # Side-face end of a chamfer strip or cap: pushed out to the tile edge on that side.
        uv[a] = 1.0 if s[a] > 0 else 0.0
    return uv


bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene["ao_distance"] = 0.01
me = bpy.data.meshes.new("tv_a1_low")
me.from_pydata((P * L).tolist(), [], faces)
me.update()
uvl = me.uv_layers.new(name="UVMap")
for poly in me.polygons:
    fv = list(poly.vertices)
    for li, vi in zip(poly.loop_indices, fv):
        uv = uv_of(vi, fv)
        uvl.data[li].uv = tuple(uv) if uv is not None else (5.0, 5.0)
me.polygons.foreach_set("use_smooth", [True] * len(me.polygons))
me.normals_split_custom_set([tuple(AX[axis_of[me.loops[i].vertex_index]] * sgn_of[me.loops[i].vertex_index][axis_of[me.loops[i].vertex_index]])
                             for i in range(len(me.loops))])
me.materials.append(bpy.data.materials.new(args.stack))
low_col, high_col, wear_col = (bpy.data.collections.new(n) for n in ("low", "high", "wear"))
for col in (low_col, high_col, wear_col):
    scene.collection.children.link(col)
low = bpy.data.objects.new("tv_a1_low", me)
low["self_high"] = True
low_col.objects.link(low)
high = low.copy()
high.data = me.copy()
high.name = high.data.name = "tv_a1_high"
high_col.objects.link(high)
bpy.ops.wm.save_as_mainfile(filepath=args.out)
print(f"[tile] {len(P)} verts, {len(faces)} faces, box {L * 1000:.0f} mm, chamfer {c} -> {args.out}")
