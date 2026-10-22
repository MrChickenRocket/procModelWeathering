"""Render textured previews of a mech (all its atlas parts): side, front, 3/4, head and legs close-ups.

blender -b <mech.blend> -P tools/render_mog.py -- <tex_dir> <out_dir> [--size 1600] [--samples 128]
"""
import argparse
import json
import math
import os
import sys

import bpy
import numpy as np
from mathutils import Vector

argv = sys.argv[sys.argv.index("--") + 1:]
ap = argparse.ArgumentParser()
ap.add_argument("tex_dir")
ap.add_argument("out_dir")
ap.add_argument("--size", type=int, default=1600)
ap.add_argument("--samples", type=int, default=128)
ap.add_argument("--prefix", default="")
ap.add_argument("--views", default="side,front,three_quarter,three_quarter_rear,head")
ap.add_argument("--field", default="", help="show a sim debug field (debug/<field>.png) as flat colour instead")
args = ap.parse_args(argv)
blend = bpy.data.filepath
with open(os.path.splitext(blend)[0] + ".mog.json") as f:
    mog = json.load(f)

for name in ("high", "wear"):
    col = bpy.data.collections.get(name)
    for ob in list(col.objects) if col else []:
        bpy.data.objects.remove(ob)
mech = mog["mech"]
obs = [bpy.data.objects[f"{mech}_a{k}_low"] for k in range(1, mog["atlases"] + 1)]


def atlas_material(part_dir):
    mat = bpy.data.materials.new("weathered")
    nt = mat.node_tree
    bsdf = nt.nodes["Principled BSDF"]

    def tex(name, data):
        n = nt.nodes.new("ShaderNodeTexImage")
        n.image = bpy.data.images.load(os.path.join(part_dir, f"{name}.png"))
        n.image.colorspace_settings.is_data = data
        return n

    if args.field:
        nt.links.new(tex(f"debug/{args.field}", True).outputs["Color"], bsdf.inputs["Base Color"])
        bsdf.inputs["Roughness"].default_value = 0.8
        return mat
    col_node = tex("color", False)
    nt.links.new(col_node.outputs["Color"], bsdf.inputs["Base Color"])
    if os.path.exists(os.path.join(part_dir, "emissive.png")):
        # Mirrors Roblox EmissiveMaskContent: the colour map tints, the mask gates.
        nt.links.new(col_node.outputs["Color"], bsdf.inputs["Emission Color"])
        strength = nt.nodes.new("ShaderNodeMath")
        strength.operation = 'MULTIPLY'
        strength.inputs[1].default_value = 6.0
        nt.links.new(tex("emissive", True).outputs["Color"], strength.inputs[0])
        nt.links.new(strength.outputs["Value"], bsdf.inputs["Emission Strength"])
    nt.links.new(tex("metalness", True).outputs["Color"], bsdf.inputs["Metallic"])
    nt.links.new(tex("roughness", True).outputs["Color"], bsdf.inputs["Roughness"])
    nmap = nt.nodes.new("ShaderNodeNormalMap")
    nt.links.new(tex("normal", True).outputs["Color"], nmap.inputs["Color"])
    nt.links.new(nmap.outputs["Normal"], bsdf.inputs["Normal"])
    return mat


def mesh_co(me):
    co = np.empty(len(me.vertices) * 3)
    me.vertices.foreach_get("co", co)
    vi = np.empty(len(me.polygons) * 3, int)
    me.polygons.foreach_get("vertices", vi)
    return co.reshape(-1, 3), vi.reshape(-1, 3)


for ob in obs:
    mat = atlas_material(os.path.join(os.path.abspath(args.tex_dir), ob.name[:-4]))
    for i in range(len(ob.data.materials)):
        ob.data.materials[i] = mat

co = np.concatenate([mesh_co(ob.data)[0] for ob in obs])
lo, hi = co.min(0), co.max(0)
center = (lo + hi) / 2
height = hi[2] - lo[2]

head_bone = next((b for b in ("Head", "Chest", "Pelvis") if b in mog["face_ranges"]), next(iter(mog["face_ranges"])))
part, a, b = mog["face_ranges"][head_bone]
hco, hvi = mesh_co(bpy.data.objects[f"{part}_low"].data)
head = hco[hvi[a:b].ravel()]
head_c = (head.min(0) + head.max(0)) / 2
head_r = np.linalg.norm(head.max(0) - head.min(0)) / 2

scene = bpy.context.scene
scene.render.engine = 'CYCLES'
scene.cycles.samples = args.samples
scene.cycles.use_denoising = True
prefs = bpy.context.preferences.addons["cycles"].preferences
prefs.compute_device_type = 'OPTIX'
prefs.get_devices()
for d in prefs.devices:
    d.use = d.type == 'OPTIX'
scene.cycles.device = 'GPU'
scene.render.resolution_x = scene.render.resolution_y = args.size
scene.view_settings.view_transform = 'AgX'
scene.view_settings.look = 'AgX - Medium High Contrast'
scene.view_settings.exposure = -1.6

world = bpy.data.worlds.new("sky")
scene.world = world
wn = world.node_tree
sky = wn.nodes.new("ShaderNodeTexSky")
sky.sun_elevation = math.radians(38)
sky.sun_rotation = math.radians(215)
wn.links.new(sky.outputs["Color"], wn.nodes["Background"].inputs["Color"])
wn.nodes["Background"].inputs["Strength"].default_value = 0.35

bpy.ops.mesh.primitive_plane_add(size=height * 40, location=(center[0], center[1], lo[2]))
ground = bpy.context.active_object
gm = bpy.data.materials.new("ground")
gb = gm.node_tree.nodes["Principled BSDF"]
noise = gm.node_tree.nodes.new("ShaderNodeTexNoise")
noise.inputs["Scale"].default_value = 40 / height
ramp = gm.node_tree.nodes.new("ShaderNodeValToRGB")
ramp.color_ramp.elements[0].color = (0.07, 0.068, 0.065, 1)
ramp.color_ramp.elements[1].color = (0.13, 0.125, 0.12, 1)
gm.node_tree.links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
gm.node_tree.links.new(ramp.outputs["Color"], gb.inputs["Base Color"])
gb.inputs["Roughness"].default_value = 0.9
ground.data.materials.append(gm)


def shoot(name, target, direction, dist, lens=60):
    if name not in args.views.split(","):
        return
    cam_d = bpy.data.cameras.new(name)
    cam_d.lens = lens
    cam_d.clip_start = height * 0.01
    cam_d.clip_end = height * 100
    cam = bpy.data.objects.new(name, cam_d)
    scene.collection.objects.link(cam)
    d = Vector(direction).normalized()
    cam.location = Vector(target) + d * dist
    cam.rotation_euler = (-d).to_track_quat('-Z', 'Y').to_euler()
    scene.camera = cam
    scene.render.filepath = os.path.join(os.path.abspath(args.out_dir), f"{args.prefix}{name}{'_' + args.field if args.field else ''}.png")
    bpy.ops.render.render(write_still=True)
    print(f"[render] {scene.render.filepath}")


os.makedirs(args.out_dir, exist_ok=True)
span = max(hi - lo)
# Pipeline space: +Y forward, +Z up, +X is the mech's left (Roblox x right seen from behind).
shoot("side", center, (1, 0, 0.08), span * 2.0)
shoot("front", center, (0, 1, 0.12), span * 2.0)
shoot("three_quarter", center, (0.75, 0.85, 0.45), span * 2.0)
shoot("three_quarter_rear", center, (-0.8, -0.7, 0.4), span * 2.0)
shoot("head", head_c, (0.8, 0.9, 0.35), head_r * 5.0)
legs = co[co[:, 2] < lo[2] + 0.55 * height]
legs_c = (legs.min(0) + legs.max(0)) / 2
shoot("legs", legs_c, (0.9, 0.6, 0.25), np.linalg.norm(legs.max(0) - legs.min(0)) * 1.2)


def bone_box(bone):
    part, a, b = mog["face_ranges"][bone]
    bco, bvi = mesh_co(bpy.data.objects[f"{part}_low"].data)
    p = bco[bvi[a:b].ravel()]
    return (p.min(0) + p.max(0)) / 2, np.linalg.norm(p.max(0) - p.min(0)) / 2


# Rider's-eye shots: 14 in-game studs off the bone's bounding sphere (an install scale enlarges the mech, so
# fewer build studs).
near = 14 * mog["m_per_stud"] / mog["config"].get("density", {}).get("install_scale", 1.0)
if "UpperArmL" in mog["face_ranges"]:
    c, r = bone_box("UpperArmL")
    shoot("pauldron", c, (0.6, 0.5, 0.65), r + near)
if "Chest" in mog["face_ranges"]:
    c, r = bone_box("Chest")
    shoot("chest", c, (0.35, 0.9, 0.45), r + near)
