"""Build a pipeline .blend from a MogCompany mech export: bone meshes joined into one part per texture atlas.

blender -b -P tools/import_mog.py -- <export_dir> <config.json> <out.blend>

Parts are <mech>_a<k>_low/_high with one material slot per sim stack (the bake writes a matid map from them).
config "atlases" lists the bones of each atlas; piston stages follow their A bone unless listed by OBJ name (e.g.
"kneeL_1"). Without it there is one atlas.
config "density" weights texel density per connected piece by its in-game height (see piece_weights).
Face order in a part is the rig's mesh order then each OBJ's face order; export_mog.py relies on that.
"""
import json
import os
import sys

import bpy
import numpy as np
from mathutils import Matrix

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mogio  # noqa: E402

TEX_SIZE = 4096
UV_PAD_PX = 8
# Above the 45 degree bevel strips, so they join their faces; tiny strip islands waste most of the atlas in padding.
UV_ANGLE = np.radians(50)


def components(faces):
    parent = {}

    def find(a):
        while parent.setdefault(a, a) != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for f in faces:
        for b in f[1:]:
            parent[find(f[0])] = find(b)
    return [find(f[0]) for f in faces]


def tint_faces(V, F, stacks, rule, face_ranges):
    """Reassign the slot's connected pieces to rule['stack']: pieces whose centroid is within within_studs of a "near"
    point (Roblox world studs, mirror_x adds -x copies), else every piece of tube radius <= max_radius_studs, else
    rule['fraction'] of them, else all. "bones" limits the rule to those bones' meshes. pick "thinnest" takes thin
    tubes first; "symmetric" takes random left/right mirror pairs (rule seed). Returns (picked, total)."""
    sel = [i for i, s in enumerate(stacks) if s == rule["slot_stack"]]
    if "bones" in rule:
        ok = np.zeros(len(stacks), bool)
        for b in rule["bones"]:
            a, e = face_ranges[b]
            ok[a:e] = True
        sel = [i for i in sel if ok[i]]
    comp = components([F[i] for i in sel])
    pieces = {}
    for i, c in zip(sel, comp):
        pieces.setdefault(c, []).append(i)
    radius = {}
    for c, fs in pieces.items():
        tri = V[np.array([F[i] for i in fs])]
        area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1).sum()
        pts = tri.reshape(-1, 3)
        length = np.linalg.norm(pts.max(0) - pts.min(0))
        radius[c] = area / max(2 * np.pi * length, 1e-9)
    if rule.get("pick", "thinnest") == "symmetric":
        keys = list(pieces)
        cen = np.array([V[np.array([F[i] for i in pieces[c]])].reshape(-1, 3).mean(0) for c in keys])
        dm = np.linalg.norm(cen[:, None] - (cen * np.array([-1.0, 1.0, 1.0]))[None], axis=2)
        j = dm.argmin(0)
        d = dm[j, np.arange(len(keys))]
        tol = 0.02 * np.linalg.norm(V.max(0) - V.min(0))
        group = {k: min(k, keys[jj]) if dd < tol else k for k, dd, jj in zip(keys, d, j)}
        heads = sorted(set(group.values()))
        np.random.default_rng(rule.get("seed", 1)).shuffle(heads)
        rank = {h: r for r, h in enumerate(heads)}
        order = sorted(pieces, key=lambda c: rank[group[c]])
    else:
        order = sorted(pieces, key=radius.get)
    if "near" in rule:
        pts = np.array(rule["near"], np.float64)
        if rule.get("mirror_x"):
            pts = np.concatenate([pts, pts * [-1, 1, 1]])
        pts = pts @ mogio.RBX_TO_BLENDER.T * rule["m_per_stud"]
        tol = rule.get("within_studs", 0.1) * rule["m_per_stud"]
        cen = {c: V[np.array([F[i] for i in fs])].reshape(-1, 3).mean(0) for c, fs in pieces.items()}
        pick = [c for c in order if np.linalg.norm(pts - cen[c], axis=1).min() <= tol]
    elif "max_radius_studs" in rule:
        pick = [c for c in order if radius[c] <= rule["max_radius_studs"] * rule["m_per_stud"]]
    elif "fraction" in rule:
        pick = order[:round(len(order) * rule["fraction"])]
    else:
        pick = order
    for c in pick:
        for i in pieces[c]:
            stacks[i] = rule["stack"]
    return len(pick), len(order)


def atlas_groups(cfg, rig, face_ranges):
    groups = cfg.get("atlases") or [[b["name"] for b in rig["bones"]]]
    of = {b: k for k, bones in enumerate(groups, 1) for b in bones}
    for pst in rig.get("pistons", []):
        for st in pst.get("stages", []):
            of.setdefault(os.path.splitext(st["obj"])[0], of[pst["a"]["bone"]])
    missing = [n for n in face_ranges if n not in of]
    assert not missing, f"meshes in no atlas: {missing}"
    return {n: of[n] for n in face_ranges}


def piece_weights(V, F, s, dcfg):
    """Texel density multiplier per face, from its connected piece's centre height in game (build y * install
    scale): 1 up to full_below studs, easing to floor at floor_at and above. Pieces centred in a rider box (build
    studs [[x0, y0, z0], [x1, y1, z1]]) keep 1: players stand on them."""
    w = np.ones(len(F))
    if not dcfg:
        return w
    comp = np.array(components(F.tolist()))
    _, piece = np.unique(comp, return_inverse=True)
    cen = np.zeros((piece.max() + 1, 3))
    np.add.at(cen, piece, V[F].mean(1))
    cen /= np.bincount(piece)[:, None]
    build = cen @ mogio.RBX_TO_BLENDER / s
    t = np.clip((build[:, 1] * dcfg.get("install_scale", 1.0) - dcfg["full_below"])
                / (dcfg["floor_at"] - dcfg["full_below"]), 0, 1)
    pw = 1 - (1 - dcfg["floor"]) * t * t * (3 - 2 * t)
    for lo, hi in dcfg.get("rider_boxes", []):
        pw[np.all((build >= lo) & (build <= hi), axis=1)] = 1.0
    print(f"[mog] density weights: {len(pw)} pieces, {np.mean(pw < 0.99):.0%} below full, min {pw.min():.2f}")
    return pw[piece]


def uv_unwrap(low, w):
    vl = bpy.context.view_layer
    vl.objects.active = low
    low.select_set(True)
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.uv.smart_project(angle_limit=UV_ANGLE, island_margin=0.0, scale_to_bounds=False)
    bpy.ops.uv.select_all(action='SELECT')
    bpy.ops.uv.average_islands_scale()
    bpy.ops.object.mode_set(mode='OBJECT')
    if (w < 0.999).any():
        me = low.data
        uv = np.empty(len(me.loops) * 2)
        me.uv_layers.active.data.foreach_get("uv", uv)
        uv = uv.reshape(-1, 3, 2)
        # Scaling each face about its own centroid would tear islands; one shared origin keeps them whole, and the
        # pack below moves every island anyway.
        uv *= w[:, None, None]
        me.uv_layers.active.data.foreach_set("uv", uv.ravel())
    _pack(low, scale=True)
    low.select_set(False)


def _pack(ob, scale):
    vl = bpy.context.view_layer
    vl.objects.active = ob
    ob.select_set(True)
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.uv.select_all(action='SELECT')
    bpy.ops.uv.pack_islands(scale=scale, rotate=True, margin_method='FRACTION', margin=UV_PAD_PX / TEX_SIZE,
                            shape_method='CONCAVE')
    bpy.ops.object.mode_set(mode='OBJECT')
    ob.select_set(False)


def _density(me, w):
    """Full-weight texels per meter: sqrt(UV area * TEX_SIZE^2 / weighted world area)."""
    uv = np.empty(len(me.loops) * 2)
    me.uv_layers.active.data.foreach_get("uv", uv)
    uv = uv.reshape(-1, 3, 2)
    a_uv = 0.5 * np.abs(np.cross(uv[:, 1] - uv[:, 0], uv[:, 2] - uv[:, 0])).sum()
    a_w = (np.array([p.area for p in me.polygons]) * w ** 2).sum()
    return np.sqrt(a_uv / a_w) * TEX_SIZE


def equalize_density(lows, weights):
    """Scale every atlas down to the sparsest one's full-weight texel density and repack, so all atlases (and mirror
    pairs split across them) match. Returns that density in texels per meter."""
    d = [_density(o.data, w) for o, w in zip(lows, weights)]
    target = min(d)
    for o, dk in zip(lows, d):
        print(f"[mog] {o.name[:-4]}: {dk:.0f} texels/m packed, {100 * (target / dk) ** 2:.0f}% of that area kept")
        if dk <= target * 1.001:
            continue
        me = o.data
        uv = np.empty(len(me.loops) * 2)
        me.uv_layers.active.data.foreach_get("uv", uv)
        me.uv_layers.active.data.foreach_set("uv", uv * (target / dk))
        _pack(o, scale=False)
    return target


def main():
    src, cfg_path, out = sys.argv[sys.argv.index("--") + 1:][:3]
    with open(cfg_path) as f:
        cfg = json.load(f)
    s = cfg["m_per_stud"]
    rig, bones = mogio.load_rig(src)
    mech = rig["mech"]

    V, F, loop_n, stacks, bone_of_face, face_ranges, bone_local = [], [], [], [], [], {}, {}
    for bi, (name, path, frame) in enumerate(mogio.mesh_objs(src, rig)):
        obj = mogio.parse_obj(path)
        base = sum(len(v) for v in V)
        V.append(mogio.to_world(obj["v"], frame, s))
        nw = mogio.to_world(obj["vn"], frame, s, normals=True)
        start = len(F)
        for slot, vi, _, ni in obj["faces"]:
            F.append(tuple(base + i for i in vi))
            loop_n.extend(nw[list(ni)])
            stacks.append(cfg["slots"][slot])
            bone_of_face.append(bi)
        face_ranges[name] = [start, len(F)]
        bone_local[name] = obj["v"]
    V = np.concatenate(V)
    for rule in cfg.get("tint", []):
        rule = dict(rule, slot_stack=cfg["slots"][rule["slot"]], m_per_stud=s)
        k, n = tint_faces(V, F, stacks, rule, face_ranges)
        print(f"[mog] tint: {k} of {n} {rule['slot']} pieces -> {rule['stack']}")
    F = np.array(F)
    loop_n = np.array(loop_n).reshape(-1, 3, 3)
    bone_of_face = np.array(bone_of_face)
    weight = piece_weights(V, F, s, cfg.get("density"))
    atlas_of = atlas_groups(cfg, rig, face_ranges)
    n_atlas = max(atlas_of.values())

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    for k, v in cfg.get("bake", {}).items():
        scene[k] = v
    low_col, high_col, wear_col = (bpy.data.collections.new(n) for n in ("low", "high", "wear"))
    for col in (low_col, high_col, wear_col):
        scene.collection.children.link(col)

    # Every part carries every stack's slot, so each bakes a matid map and runs the multi-material sim path.
    names = sorted(set(stacks))
    mats = [bpy.data.materials.new(n) for n in names]
    ranges, lows, part_w = {}, [], []
    for k in range(1, n_atlas + 1):
        objs = [n for n in face_ranges if atlas_of[n] == k]
        sel = np.concatenate([np.arange(*face_ranges[n]) for n in objs])
        part = f"{mech}_a{k}"
        start = 0
        for n in objs:
            c = face_ranges[n][1] - face_ranges[n][0]
            ranges[n] = [part, start, start + c]
            start += c
        used, Fk = np.unique(F[sel], return_inverse=True)
        me = bpy.data.meshes.new(f"{part}_low")
        me.from_pydata(V[used].tolist(), [], Fk.reshape(-1, 3).tolist())
        for m in mats:
            me.materials.append(m)
        me.polygons.foreach_set("material_index", [names.index(stacks[i]) for i in sel])
        me.polygons.foreach_set("use_smooth", [True] * len(sel))
        me.attributes.new("bone_id", 'INT', 'FACE').data.foreach_set("value", bone_of_face[sel])
        me.update()
        me.normals_split_custom_set([tuple(n) for n in loop_n[sel].reshape(-1, 3)])
        low = bpy.data.objects.new(f"{part}_low", me)
        low_col.objects.link(low)
        low["self_high"] = True
        uv_unwrap(low, weight[sel])
        lows.append(low)
        part_w.append(weight[sel])
    density = equalize_density(lows, part_w)
    for low in lows:
        high = low.copy()
        high.data = low.data.copy()
        high.name = high.data.name = low.name[:-4] + "_high"
        high_col.objects.link(high)

    for wi, w in enumerate(cfg.get("wear", [])):
        if "box" in w:
            # Build-space studs: [[x0, y0, z0], [x1, y1, z1]].
            R, p = np.eye(3), np.zeros(3)
            lo, hi = np.array(w["box"], np.float64)
        else:
            R, p = bones[w["bone"]]
            lv = bone_local[w["bone"]]
            lo, hi = lv.min(0), lv.max(0)
        local = np.eye(4)
        local[:3, :3] = np.diag(np.maximum((hi - lo) / 2 * 1.1, 0.02))
        local[:3, 3] = (hi + lo) / 2
        bone_m = np.eye(4)
        bone_m[:3, :3], bone_m[:3, 3] = R, p
        to_b = np.eye(4)
        to_b[:3, :3] = mogio.RBX_TO_BLENDER * s
        e = bpy.data.objects.new(f"WEAR_{w.get('bone', f'box{wi}')}", None)
        e.empty_display_type = 'CUBE'
        e.empty_display_size = 1.0
        e.matrix_world = Matrix((to_b @ bone_m @ local).tolist())
        e["weight"] = w["weight"]
        wear_col.objects.link(e)

    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(os.path.splitext(out)[0] + ".mog.json", "w") as f:
        json.dump(dict(mech=mech, src=os.path.abspath(src), m_per_stud=s, atlases=n_atlas, face_ranges=ranges,
                       atlas_of=atlas_of, stacks=names, config=cfg), f, indent=1)
    bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(out))
    print(f"[mog] {mech}: {len(F)} tris, {len(ranges)} meshes in {n_atlas} atlases, "
          f"{density * s:.0f} texels/stud at full weight, stacks {names} -> {out}")


main()
