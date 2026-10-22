"""Write a weathered mech back out in the MogCompany layout, with the atlas UVs.

blender -b <mech.blend> -P tools/export_mog.py -- <out_dir>

Out:
  <out_dir>/<name>.obj       the source OBJs (bones and piston stages) with only vt/f rewritten (same v/vn order, studs,
                             bone space); UVs are 0..1 in the OBJ's own atlas
  <out_dir>/atlases.json     {"count": N, "objs": {"<name>.obj": k}}, k = 1..N
  <out_dir>/mog_a<k>.mtl     every slot points at atlas k's scarred maps
  <out_dir>/<mech>.fbx       the source FBX rig, UVs transferred onto it, one material per atlas
The maps themselves (textures_<tier>/a<k>/) are copied by tools/run_mog.sh.
"""
import json
import os
import sys

import bpy
import numpy as np
from mathutils.kdtree import KDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mogio  # noqa: E402

out_dir = os.path.abspath(sys.argv[sys.argv.index("--") + 1])
with open(os.path.splitext(bpy.data.filepath)[0] + ".mog.json") as f:
    mog = json.load(f)
mech, src, n_atlas = mog["mech"], mog["src"], mog["atlases"]
os.makedirs(out_dir, exist_ok=True)


def mesh_arrays(me):
    uv = np.empty(len(me.loops) * 2)
    me.uv_layers.active.data.foreach_get("uv", uv)
    co = np.empty(len(me.vertices) * 3)
    me.vertices.foreach_get("co", co)
    tri_v = np.empty(len(me.polygons) * 3, int)
    me.polygons.foreach_get("vertices", tri_v)
    return uv.reshape(-1, 3, 2), co.reshape(-1, 3), tri_v.reshape(-1, 3)


parts = {f"{mech}_a{k}": mesh_arrays(bpy.data.objects[f"{mech}_a{k}_low"].data) for k in range(1, n_atlas + 1)}

rig, _ = mogio.load_rig(src)
atlas_json = {"count": n_atlas, "objs": {}}
for bone, path, _ in mogio.mesh_objs(src, rig):
    part, a, b = mog["face_ranges"][bone]
    k = mog["atlas_of"][bone]
    uv = parts[part][0]
    obj = mogio.parse_obj(path)
    assert len(obj["faces"]) == b - a, f"{bone}: source OBJ changed since import ({len(obj['faces'])} vs {b - a} faces)"
    lines = [f"# {bone}.obj: Roblox studs, bone space; UVs: {mech} atlas {k} of {n_atlas}", f"mtllib mog_a{k}.mtl"]
    lines += [f"v {x:.5f} {y:.5f} {z:.5f}" for x, y, z in obj["v"]]
    lines += [f"vt {u:.6f} {v:.6f}" for u, v in uv[a:b].reshape(-1, 2)]
    lines += [f"vn {x:.4f} {y:.4f} {z:.4f}" for x, y, z in obj["vn"]]
    slot = None
    for i, (s, vi, _, ni) in enumerate(obj["faces"]):
        if s != slot:
            lines.append(f"usemtl {s}")
            slot = s
        lines.append("f " + " ".join(f"{vi[c] + 1}/{3 * i + c + 1}/{ni[c] + 1}" for c in range(3)))
    with open(os.path.join(out_dir, f"{bone}.obj"), "w") as f:
        f.write("\n".join(lines) + "\n")
    atlas_json["objs"][f"{bone}.obj"] = k

with open(os.path.join(out_dir, "atlases.json"), "w") as f:
    json.dump(atlas_json, f, indent=1)
for k in range(1, n_atlas + 1):
    t = f"textures_scarred/a{k}"
    with open(os.path.join(out_dir, f"mog_a{k}.mtl"), "w") as f:
        for s in rig["slots"]:
            f.write(f"newmtl {s}\nKd 1 1 1\nmap_Kd {t}/color.png\nmap_Pm {t}/metalness.png\n"
                    f"map_Pr {t}/roughness.png\nnorm {t}/normal.png\n\n")
print(f"[export] {len(atlas_json['objs'])} OBJs in {n_atlas} atlases")

if not os.path.exists(os.path.join(src, f"{mech}.fbx")):
    sys.exit(0)  # piece exports (tools/arcpieces.py) have no rig FBX

# One triangle soup over all atlases for the FBX transfer; atlas_t says which atlas each triangle belongs to.
uv = np.concatenate([p[0] for p in parts.values()])
offs = np.cumsum([0] + [len(p[1]) for p in parts.values()])
co = np.concatenate([p[1] for p in parts.values()])
tri_v = np.concatenate([p[2] + o for p, o in zip(parts.values(), offs)])
atlas_t = np.concatenate([np.full(len(p[2]), k) for k, p in enumerate(parts.values())])

# FBX: keep the source rig and skinning; copy each corner's UV from the atlas mesh by matching position and
# face normal. The source FBX polygons are quads/ngons, the atlas mesh is the OBJs' triangulation.
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.fbx(filepath=os.path.join(src, f"{mech}.fbx"))
fbx_ob = next(o for o in bpy.context.scene.objects if o.type == 'MESH')
fm = fbx_ob.data
mw = fbx_ob.matrix_world
fco = np.array([mw @ v.co for v in fm.vertices])


def normalize(p, ref):
    lo, hi = ref.min(0), ref.max(0)
    return (p - (lo + hi) / 2) / (hi - lo).max()


fco_n = normalize(fco, fco)
co_n = normalize(co, co)
tri_c = co_n[tri_v].mean(1)
tri_n = np.cross(co_n[tri_v[:, 1]] - co_n[tri_v[:, 0]], co_n[tri_v[:, 2]] - co_n[tri_v[:, 0]])
tri_n /= np.maximum(np.linalg.norm(tri_n, axis=1, keepdims=True), 1e-12)
corners_of = [[] for _ in range(len(co))]
for t, vs in enumerate(tri_v):
    for k, v in enumerate(vs):
        corners_of[v].append((t, k))
kd = KDTree(len(co_n))
for i, c in enumerate(co_n):
    kd.insert(c, i)
kd.balance()

uv_layer = fm.uv_layers.active or fm.uv_layers.new(name="UVMap")
worst, misses = 0.0, 0
poly_atlas = np.zeros(len(fm.polygons), int)
for poly in fm.polygons:
    pts = fco_n[list(poly.vertices)]
    n = np.array(mw.to_3x3() @ poly.normal)
    n /= max(np.linalg.norm(n), 1e-12)
    pc = pts.mean(0)
    for li, p in zip(poly.loop_indices, pts):
        best, best_d = None, np.inf
        near = kd.find_range(p, 1e-4)
        # Non-planar quads: their normal is off every triangle's, so fall back to centroid distance alone.
        min_dot = 0.98 if any(tri_n[t] @ n >= 0.98 for _, v, _ in near for t, _ in corners_of[v]) else -1.0
        for _, v, dv in near:
            for t, k in corners_of[v]:
                if tri_n[t] @ n < min_dot:
                    continue
                # Coplanar triangles at a shared vertex can sit on different islands; the one inside this
                # polygon is closest to its centroid.
                d = np.linalg.norm(tri_c[t] - pc)
                if d < best_d:
                    best, best_d, best_dv, best_t = uv[t, k], d, dv, t
        if best is None:
            misses += 1
            continue
        worst = max(worst, best_dv)
        uv_layer.data[li].uv = best
        poly_atlas[poly.index] = atlas_t[best_t]
print(f"[export] fbx uv transfer: worst vertex match {worst:.1e} (normalized), {misses} of {len(fm.loops)} corners unmatched")


def atlas_material(k):
    mat = bpy.data.materials.new(f"{mech}_a{k}")
    nt = mat.node_tree
    bsdf = nt.nodes["Principled BSDF"]

    def tex(name, data):
        n = nt.nodes.new("ShaderNodeTexImage")
        n.image = bpy.data.images.load(os.path.join(out_dir, "textures_scarred", f"a{k}", f"{name}.png"),
                                       check_existing=True)
        n.image.colorspace_settings.is_data = data
        return n

    nt.links.new(tex("color", False).outputs["Color"], bsdf.inputs["Base Color"])
    nt.links.new(tex("metalness", True).outputs["Color"], bsdf.inputs["Metallic"])
    nt.links.new(tex("roughness", True).outputs["Color"], bsdf.inputs["Roughness"])
    nmap = nt.nodes.new("ShaderNodeNormalMap")
    nt.links.new(tex("normal", True).outputs["Color"], nmap.inputs["Color"])
    nt.links.new(nmap.outputs["Normal"], bsdf.inputs["Normal"])
    return mat


fm.materials.clear()
for k in range(1, n_atlas + 1):
    fm.materials.append(atlas_material(k))
# materials.clear() leaves material_index pointing past the new slots; set every polygon explicitly.
fm.polygons.foreach_set("material_index", poly_atlas.tolist())
bpy.ops.object.select_all(action='SELECT')
bpy.ops.export_scene.fbx(filepath=os.path.join(out_dir, f"{mech}.fbx"), use_selection=True, add_leaf_bones=False,
                         bake_anim=False, axis_forward="Z", axis_up="Y", mesh_smooth_type="FACE", apply_unit_scale=True,
                         path_mode='RELATIVE')
print(f"[export] {os.path.join(out_dir, mech + '.fbx')}")
