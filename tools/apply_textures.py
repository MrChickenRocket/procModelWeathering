"""Assign sim output textures to the low-poly parts and export a textured GLB (low only) for the viewer.

blender -b <in.blend> -P tools/apply_textures.py -- <tex_dir> <out.glb>
"""
import os
import sys

import bpy

tex_dir, out = (os.path.abspath(a) for a in sys.argv[sys.argv.index("--") + 1:][:2])


def image(path, data):
    img = bpy.data.images.load(path)
    img.colorspace_settings.is_data = data
    return img


for col_name in ("high", "wear"):
    col = bpy.data.collections.get(col_name)
    for ob in list(col.objects) if col else []:
        bpy.data.objects.remove(ob)

for ob in bpy.data.collections["low"].objects:
    part_dir = os.path.join(tex_dir, ob.name[:-4])
    if not os.path.isdir(part_dir):
        print(f"[tex] no textures for {ob.name}, keeping flat material")
        continue
    mat = bpy.data.materials.new(f"{ob.name}_weathered")
    nt = mat.node_tree
    bsdf = nt.nodes["Principled BSDF"]

    def tex(name, data):
        n = nt.nodes.new("ShaderNodeTexImage")
        n.image = image(os.path.join(part_dir, f"{name}.png"), data)
        return n

    nt.links.new(tex("color", False).outputs["Color"], bsdf.inputs["Base Color"])
    nt.links.new(tex("metalness", True).outputs["Color"], bsdf.inputs["Metallic"])
    nt.links.new(tex("roughness", True).outputs["Color"], bsdf.inputs["Roughness"])
    nmap = nt.nodes.new("ShaderNodeNormalMap")
    nt.links.new(tex("normal", True).outputs["Color"], nmap.inputs["Color"])
    nt.links.new(nmap.outputs["Normal"], bsdf.inputs["Normal"])
    ob.data.materials.clear()
    ob.data.materials.append(mat)

os.makedirs(os.path.dirname(out), exist_ok=True)
bpy.ops.export_scene.gltf(filepath=out, export_format='GLB', export_yup=True, export_apply=True,
                          export_image_format='AUTO')
print(f"[tex] {out}")
