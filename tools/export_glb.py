"""Export a pipeline .blend (low, high, wear collections) to one GLB for the viewer.

blender -b <in.blend> -P tools/export_glb.py -- <out.glb>
"""
import os
import sys

import bpy

out = os.path.abspath(sys.argv[sys.argv.index("--") + 1])
os.makedirs(os.path.dirname(out), exist_ok=True)
bpy.ops.export_scene.gltf(
    filepath=out,
    export_format='GLB',
    export_extras=True,
    export_yup=True,
    export_apply=True,
    use_visible=False,
    use_renderable=False,
)
print(f"[export] {out}")
