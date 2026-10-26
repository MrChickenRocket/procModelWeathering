"""Bake sim input maps from <part>_high onto <part>_low for every part in a pipeline .blend.

blender -b <in.blend> -P tools/bake.py -- <out_dir> [--size 4096] [--parts slide,frame] [--maps normal_ts_round]

--maps bakes only the listed maps and merges them into each part's existing maps.npz.

Writes per part:
  <out_dir>/<part>/maps.npz      float16 arrays, shape (size, size[, C]), row 0 = v 0 (bottom of UV space)
  <out_dir>/<part>/meta.json
  <out_dir>/<part>/preview/*.png 1024 previews for eyeballing
"""
import argparse
import json
import os
import sys
import time

import bpy
import numpy as np

CAGE_EXTRUSION = 0.0005
MAX_RAY_DISTANCE = 0.001
DATA_MARGIN_PX = 4
AO_SAMPLES = 64
PREVIEW_SIZE = 1024

# name: (channels, shader builder key, builder args, render samples, description)
MAPS = {
    "mask": (1, "const", {}, 1, "1 where the texel is covered by a UV island"),
    "position": (3, "position", {}, 1, "object-space position, meters"),
    "normal_obj": (3, "normal", {}, 1, "object-space shading normal of the high-poly, -1..1"),
    "normal_ts": (3, None, {}, 1, "tangent-space normal, OpenGL (+Y), -1..1"),
    "normal_ts_round": (3, "bevel", dict(radius=0.0025), 32,
                        "low-poly's own normal with its edges rounded to ~2.5 mm (Bevel node), tangent space, -1..1; "
                        "no high-poly detail, so the sim layers normal_ts on top"),
    "ao": (1, "ao", dict(distance=0.02, inside=False, only_local=True), AO_SAMPLES,
           "ambient occlusion from the part alone, 1 = open"),
    "ao_assembled": (1, "ao", dict(distance=0.02, inside=False, only_local=False), AO_SAMPLES,
                     "ambient occlusion with all other parts present, 1 = open"),
    "convex_small": (1, "ao_inv", dict(distance=0.00075, inside=True, only_local=True), AO_SAMPLES,
                     "edge convexity within ~0.75 mm, 1 = sharp convex edge"),
    "convex_large": (1, "ao_inv", dict(distance=0.003, inside=True, only_local=True), AO_SAMPLES,
                     "edge convexity within ~3 mm"),
    "concave_small": (1, "ao_inv", dict(distance=0.00075, inside=False, only_local=True), AO_SAMPLES,
                      "cavity within ~0.75 mm, 1 = deep crease"),
    "concave_large": (1, "ao_inv", dict(distance=0.003, inside=False, only_local=True), AO_SAMPLES,
                      "cavity within ~3 mm"),
    "concave_assembled": (1, "ao_inv", dict(distance=0.003, inside=False, only_local=False), AO_SAMPLES,
                          "cavity within ~3 mm with all other parts present; minus concave_large, it is the contact "
                          "line where another piece sits on or over this one"),
    "thickness": (1, "ao", dict(distance=0.01, inside=True, only_local=True), AO_SAMPLES,
                  "material thickness up to 10 mm, 1 = thick"),
    "matid": (1, "matid", {}, 1, "index into meta materials as (i + 0.5) / MATID_SCALE; multi-material parts only"),
}
MATID_SCALE = 32


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("out_dir")
    p.add_argument("--size", type=int, default=4096)
    p.add_argument("--parts", default="")
    p.add_argument("--maps", default="")
    p.add_argument("--volumes-only", action="store_true")
    return p.parse_args(argv)


def export_wear_volumes(path):
    col = bpy.data.collections.get("wear")
    vols = [{"name": o.name,
             "weight": float(o.get("weight", 1.0)),
             "world_to_local": [list(r) for r in o.matrix_world.inverted()]}
            for o in (col.objects if col else []) if o.name.startswith("WEAR_")]
    with open(path, "w") as f:
        json.dump(vols, f, indent=2)
    print(f"[bake] {len(vols)} wear volumes -> {path}")


def setup_cycles():
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.use_denoising = False
    prefs = bpy.context.preferences.addons["cycles"].preferences
    for backend in ("OPTIX", "CUDA", "HIP", "METAL", "ONEAPI"):
        try:
            prefs.compute_device_type = backend
        except TypeError:
            continue
        prefs.get_devices()
        gpus = [d for d in prefs.devices if d.type == backend]
        if gpus:
            for d in prefs.devices:
                d.use = d.type == backend
            scene.cycles.device = 'GPU'
            print(f"[bake] device: {backend} {[d.name for d in gpus]}")
            return
    scene.cycles.device = 'CPU'
    print("[bake] device: CPU")


def emit_material(kind, args):
    mat = bpy.data.materials.new(f"_bake_{kind}")
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    emit = nt.nodes.new("ShaderNodeEmission")
    nt.links.new(emit.outputs["Emission"], out.inputs["Surface"])
    if kind == "const":
        emit.inputs["Color"].default_value = (1, 1, 1, 1)
    elif kind == "position":
        tc = nt.nodes.new("ShaderNodeTexCoord")
        nt.links.new(tc.outputs["Object"], emit.inputs["Color"])
    elif kind == "normal":
        geo = nt.nodes.new("ShaderNodeNewGeometry")
        nt.links.new(geo.outputs["Normal"], emit.inputs["Color"])
    elif kind in ("ao", "ao_inv"):
        ao = nt.nodes.new("ShaderNodeAmbientOcclusion")
        ao.samples = AO_SAMPLES
        ao.inside = args["inside"]
        ao.only_local = args["only_local"]
        ao.inputs["Distance"].default_value = args["distance"]
        src = ao.outputs["AO"]
        if kind == "ao_inv":
            inv = nt.nodes.new("ShaderNodeMath")
            inv.operation = 'SUBTRACT'
            inv.inputs[0].default_value = 1.0
            nt.links.new(src, inv.inputs[1])
            src = inv.outputs["Value"]
        nt.links.new(src, emit.inputs["Color"])
    return mat


def matid_material(i):
    mat = bpy.data.materials.get(f"_bake_matid_{i}") or emit_material("const", {})
    mat.name = f"_bake_matid_{i}"
    v = (i + 0.5) / MATID_SCALE
    mat.node_tree.nodes["Emission"].inputs["Color"].default_value = (v, v, v, 1)
    return mat


def target_material(image):
    mat = bpy.data.materials.new("_bake_target")
    nt = mat.node_tree
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.image = image
    nt.nodes.active = tex
    return mat


def set_all_slots(ob, *mats):
    """One material fills every slot; several are assigned slot by slot. materials.clear() resets every
    face's material_index to 0, which silently breaks multi-material parts."""
    for i in range(len(ob.data.materials)):
        ob.data.materials[i] = mats[i] if len(mats) > 1 else mats[0]


def low_materials(low):
    """The low's own slots are swapped for the bake target while baking; this recovers their names."""
    return [bpy.data.materials[n] for n in low["_bake_materials"]]


def set_ray_visible(ob, visible):
    for attr in ("visible_camera", "visible_diffuse", "visible_glossy",
                 "visible_transmission", "visible_volume_scatter", "visible_shadow"):
        setattr(ob, attr, visible)


def bake_map(name, low, high, image, highs):
    channels, kind, args, samples, _ = MAPS[name]
    if kind == "matid":
        args = dict(materials=[m.name for m in low_materials(low)])
    scene = bpy.context.scene
    scene.cycles.samples = samples
    margin = 0 if name == "mask" else DATA_MARGIN_PX

    if kind == "bevel":
        return bake_bevel(low, image, args, channels)
    if kind is None and low.get("self_high"):
        # The high is a copy of the low, so the tangent normal is flat by definition; a ray bake would only
        # pick up neighbouring pieces that sit within the cage distance.
        return np.broadcast_to(np.array([0, 0, 1], np.float32), (image.size[1], image.size[0], 3)).copy()
    if name in ("ao", "ao_assembled") and "ao_distance" in scene:
        args = dict(args, distance=scene["ao_distance"])

    saved = {h: list(h.data.materials) for h in highs}
    if kind == "matid":
        for h in high:
            for i, m in enumerate(saved[h]):
                h.data.materials[i] = matid_material(args["materials"].index(m.name))
    elif kind is not None:
        mat = emit_material(kind, args)
        for h in highs:
            set_all_slots(h, mat)

    bpy.ops.object.select_all(action='DESELECT')
    for h in high:
        h.select_set(True)
    low.select_set(True)
    bpy.context.view_layer.objects.active = low

    bake = scene.render.bake
    bake.use_selected_to_active = True
    bake.cage_extrusion = CAGE_EXTRUSION
    bake.max_ray_distance = MAX_RAY_DISTANCE
    bake.margin = margin
    bake.use_clear = True
    bake.normal_space = 'TANGENT'
    bake.normal_r, bake.normal_g, bake.normal_b = 'POS_X', 'POS_Y', 'POS_Z'

    t = time.time()
    is_normal = kind is None
    bpy.ops.object.bake(type='NORMAL' if is_normal else 'EMIT')
    print(f"[bake] {low.name} {name}: {time.time() - t:.1f}s")

    for h, mats in saved.items():
        set_all_slots(h, *mats)

    size = image.size[0]
    px = np.empty(size * size * 4, np.float32)
    image.pixels.foreach_get(px)
    px = px.reshape(size, size, 4)[..., :channels]
    if is_normal:
        px = px * 2.0 - 1.0
    return px[..., 0] if channels == 1 else px


def bake_bevel(low, image, args, channels):
    """Bake the low-poly onto itself with Bevel-node normals, so only the low's big edges get rounded."""
    mat = bpy.data.materials.new("_bake_bevel")
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfDiffuse")
    bevel = nt.nodes.new("ShaderNodeBevel")
    bevel.samples = 16
    bevel.inputs["Radius"].default_value = args["radius"]
    # Normal bakes read the BSDF's normal input, so the Bevel node's rounded normal is what gets baked.
    nt.links.new(bevel.outputs["Normal"], bsdf.inputs["Normal"])
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.image = image
    nt.nodes.active = tex
    saved = list(low.data.materials)
    set_all_slots(low, mat)
    # The low is ray-invisible for the other bakes; the Bevel node needs to hit it.
    set_ray_visible(low, True)

    bpy.ops.object.select_all(action='DESELECT')
    low.select_set(True)
    bpy.context.view_layer.objects.active = low
    bake = bpy.context.scene.render.bake
    bake.use_selected_to_active = False
    bake.margin = DATA_MARGIN_PX
    bake.use_clear = True
    bake.normal_space = 'TANGENT'
    bake.normal_r, bake.normal_g, bake.normal_b = 'POS_X', 'POS_Y', 'POS_Z'
    t = time.time()
    bpy.ops.object.bake(type='NORMAL')
    print(f"[bake] {low.name} bevel normal: {time.time() - t:.1f}s")

    set_ray_visible(low, False)
    set_all_slots(low, *saved)
    size = image.size[0]
    px = np.empty(size * size * 4, np.float32)
    image.pixels.foreach_get(px)
    return px.reshape(size, size, 4)[..., :channels] * 2.0 - 1.0


def preview(arr, mask, path):
    a = arr.astype(np.float32)
    if a.ndim == 2:
        a = a[..., None]
    covered = mask > 0.5
    if a.shape[-1] == 3:
        lo = a[covered].min(0) if covered.any() else 0
        hi = a[covered].max(0) if covered.any() else 1
        a = (a - lo) / np.maximum(hi - lo, 1e-9)
    a = np.where(covered[..., None], np.clip(a, 0, 1), 0)
    step = max(1, a.shape[0] // PREVIEW_SIZE)
    a = a[::step, ::step]
    if a.shape[-1] == 1:
        a = np.repeat(a, 3, -1)
    h, w = a.shape[:2]
    img = bpy.data.images.new("_preview", w, h, alpha=False)
    rgba = np.concatenate([a, np.ones((h, w, 1), np.float32)], -1)
    img.pixels.foreach_set(rgba.ravel())
    img.filepath_raw = path
    img.file_format = 'PNG'
    img.save()
    bpy.data.images.remove(img)


def main():
    args = parse_args()
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    export_wear_volumes(os.path.join(out_dir, "wear_volumes.json"))
    if args.volumes_only:
        return
    setup_cycles()

    lows = {o.name[:-4]: o for o in bpy.data.collections["low"].objects if o.name.endswith("_low")}
    highs = {}
    for o in bpy.data.collections["high"].objects:
        highs.setdefault(o.get("part", o.name[:-5]), []).append(o)
    wanted = [p for p in args.parts.split(",") if p] or sorted(lows)

    # Low meshes overlap the highs; they must not occlude AO rays.
    for low in lows.values():
        set_ray_visible(low, False)

    for part in wanted:
        low, high = lows[part], highs[part]
        image = bpy.data.images.new(f"_bake_{part}", args.size, args.size, alpha=True, float_buffer=True)
        image.colorspace_settings.is_data = True
        low_mats = list(low.data.materials)
        low["_bake_materials"] = [m.name for m in low_mats]
        set_all_slots(low, target_material(image))

        t0 = time.time()
        part_dir = os.path.join(out_dir, part)
        names = [n for n in args.maps.split(",") if n] or [n for n in MAPS if n != "matid" or len(low_mats) > 1]
        maps = {}
        if args.maps:
            with np.load(os.path.join(part_dir, "maps.npz")) as old:
                maps = {k: old[k].astype(np.float32) for k in old.files}
        maps.update({name: bake_map(name, low, high, image, [h for hs in highs.values() for h in hs]) for name in names})

        set_all_slots(low, *low_mats)
        bpy.data.images.remove(image)

        os.makedirs(os.path.join(part_dir, "preview"), exist_ok=True)
        np.savez_compressed(os.path.join(part_dir, "maps.npz"),
                            **{k: v.astype(np.float16) for k, v in maps.items()})
        for name in names:
            arr = maps[name]
            preview(arr, maps["mask"], os.path.join(part_dir, "preview", f"{name}.png"))

        mask = maps["mask"] > 0.5
        meta = {
            "part": part,
            "material": low_mats[0].name if low_mats else None,
            "materials": [m.name for m in low_mats],
            "size": args.size,
            "row0": "v=0 (bottom of UV space)",
            "coverage": float(mask.mean()),
            "area_m2": sum(p.area for p in low.data.polygons),
            "maps": {k: {"channels": v[0], "description": v[4]} for k, v in MAPS.items() if k in maps},
            "position_bounds_m": [maps["position"][mask].min(0).tolist(), maps["position"][mask].max(0).tolist()],
            "normal_ts_flipped": float((maps["normal_ts"][mask][:, 2] < 0).mean()),
        }
        with open(os.path.join(part_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        print(f"[bake] {part} done in {time.time() - t0:.1f}s, coverage {meta['coverage']:.1%}")
        if meta["normal_ts_flipped"] > 0.001:
            # Usually internal faces (touching but unmerged booleans) or a cage that misses the high-poly.
            print(f"[bake] WARNING {part}: {meta['normal_ts_flipped']:.2%} of texels have flipped tangent normals")


main()
