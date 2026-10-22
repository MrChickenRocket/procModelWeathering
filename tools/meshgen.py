"""Shared mesh-authoring helpers for the test weapon generators. Geometry is built in millimeters."""
import math
import os

import bmesh
import bpy
from mathutils import Euler, Matrix, Vector

TEX_SIZE = 4096
UV_PAD_PX = 16
HIGH_BEVEL_MM = 0.4
SHARP_ANGLE = math.radians(35)

R = math.radians


def link(ob):
    bpy.context.scene.collection.objects.link(ob)
    return ob


def xf(loc, rot, size, parent):
    m = Matrix.LocRotScale(Vector(loc), Euler([R(a) for a in rot]), Vector(size))
    return parent @ m if parent is not None else m


def rot_x(deg):
    return Matrix.Rotation(R(deg), 4, 'X')


def rot_y(deg):
    return Matrix.Rotation(R(deg), 4, 'Y')


def _object(name, bm):
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    return link(bpy.data.objects.new(name, me))


def box(name, size, loc, rot=(0, 0, 0), parent=None):
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0, matrix=xf(loc, rot, size, parent))
    return _object(name, bm)


def cyl(name, radius, depth, loc, rot=(90, 0, 0), parent=None, segments=48, radius2=None):
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=True, segments=segments, radius1=radius,
                          radius2=radius if radius2 is None else radius2,
                          depth=depth, matrix=xf(loc, rot, (1, 1, 1), parent))
    return _object(name, bm)


def prism(name, pts, a0, a1, axis='X', parent=None):
    """Extrude a closed 2D polygon. axis X: pts are (y, z); Y: (x, z); Z: (x, y)."""
    def v3(p, a):
        return {'X': (a, p[0], p[1]), 'Y': (p[0], a, p[1]), 'Z': (p[0], p[1], a)}[axis]

    bm = bmesh.new()
    lo = [bm.verts.new(v3(p, a0)) for p in pts]
    hi = [bm.verts.new(v3(p, a1)) for p in pts]
    bm.faces.new(lo)
    bm.faces.new(hi)
    n = len(pts)
    for i in range(n):
        bm.faces.new((lo[i], lo[(i + 1) % n], hi[(i + 1) % n], hi[i]))
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    if parent is not None:
        bmesh.ops.transform(bm, matrix=parent, verts=bm.verts)
    return _object(name, bm)


def arc(cy, cz, r, a0, a1, n):
    return [(cy + r * math.cos(R(a)), cz + r * math.sin(R(a)))
            for a in (a0 + (a1 - a0) * i / (n - 1) for i in range(n))]


def stadium(length, width, n=8):
    """Rounded slot outline in (x, y), long along y."""
    r, h = width / 2, length / 2 - width / 2
    return arc(0, h, r, 0, 180, n) + arc(0, -h, r, 180, 360, n)


def text(name, body, size, depth, matrix):
    """Solid text centered on its origin, reaching +-depth along its local Z.
    Edges are rounded: vertical engraving walls have no slope and vanish from tangent normal bakes."""
    cu = bpy.data.curves.new(name, 'FONT')
    cu.body = body
    cu.size = size
    cu.extrude = 0.2 * depth
    cu.bevel_depth = 0.8 * depth
    cu.bevel_resolution = 3
    cu.align_x = 'CENTER'
    cu.align_y = 'CENTER'
    tmp = link(bpy.data.objects.new(name, cu))
    dg = bpy.context.evaluated_depsgraph_get()
    me = bpy.data.meshes.new_from_object(tmp.evaluated_get(dg))
    bpy.data.objects.remove(tmp)
    bpy.data.curves.remove(cu)
    me.transform(matrix)
    return link(bpy.data.objects.new(name, me))


def apply_modifiers(ob):
    dg = bpy.context.evaluated_depsgraph_get()
    me = bpy.data.meshes.new_from_object(ob.evaluated_get(dg))
    old = ob.data
    ob.modifiers.clear()
    ob.data = me
    bpy.data.meshes.remove(old)


def boolean(target, operands, op, self_intersect=False):
    col = bpy.data.collections.new(f"_bool_{target.name}")
    bpy.context.scene.collection.children.link(col)
    for o in operands:
        for uc in o.users_collection:
            uc.objects.unlink(o)
        col.objects.link(o)
    m = target.modifiers.new("bool", 'BOOLEAN')
    m.operation = op
    m.operand_type = 'COLLECTION'
    m.collection = col
    m.use_self = self_intersect
    apply_modifiers(target)
    for o in list(col.objects):
        me = o.data
        bpy.data.objects.remove(o)
        bpy.data.meshes.remove(me)
    bpy.data.collections.remove(col)


def finish_mesh(ob):
    ob.data.transform(Matrix.Scale(0.001, 4))
    ob.data.set_sharp_from_angle(angle=SHARP_ANGLE)
    ob.data.shade_smooth()


def weight_normals(ob):
    # Without this, smooth shading bleeds from bevel strips across large flat faces.
    m = ob.modifiers.new("wn", 'WEIGHTED_NORMAL')
    m.mode = 'FACE_AREA'
    m.keep_sharp = True
    apply_modifiers(ob)


def make_high(low, detail_cuts=None):
    """detail_cuts: callable returning mm-space cutters applied after the bevel (engravings, checkering)."""
    hi = low.copy()
    hi.data = low.data.copy()
    hi.name = low.name.replace("_low", "_high")
    m = hi.modifiers.new("bevel", 'BEVEL')
    m.width = HIGH_BEVEL_MM
    m.segments = 4
    m.limit_method = 'ANGLE'
    m.angle_limit = R(30)
    m.use_clamp_overlap = True
    link(hi)
    apply_modifiers(hi)
    if detail_cuts:
        boolean(hi, detail_cuts(), 'DIFFERENCE', self_intersect=True)
    return hi


def unwrap(ob):
    bpy.ops.object.select_all(action='DESELECT')
    ob.select_set(True)
    bpy.context.view_layer.objects.active = ob
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.uv.smart_project(angle_limit=SHARP_ANGLE, island_margin=0.0)
    bpy.ops.uv.select_all(action='SELECT')
    bpy.ops.uv.pack_islands(rotate=True, margin_method='FRACTION', margin=UV_PAD_PX / TEX_SIZE)
    bpy.ops.object.mode_set(mode='OBJECT')


def tri_count(ob):
    ob.data.calc_loop_triangles()
    return len(ob.data.loop_triangles)


def add_wear_volumes(col, volumes):
    """volumes: (name, center_mm, half_extents_mm, parent_matrix_or_None, weight)."""
    for name, center, half, parent, weight in volumes:
        e = bpy.data.objects.new(name, None)
        e.empty_display_type = 'CUBE'
        e.empty_display_size = 1.0
        local = Matrix.LocRotScale(Vector(center), None, Vector(half))
        e.matrix_world = Matrix.Scale(0.001, 4) @ (parent @ local if parent is not None else local)
        e["weight"] = weight
        col.objects.link(e)


def build_scene(tag, materials, parts, volumes):
    """parts: (name, builder, material, detail_cuts_or_None). Returns (low, high, wear) collections."""
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    cols = [bpy.data.collections.new(n) for n in ("low", "high", "wear")]
    for col in cols:
        scene.collection.children.link(col)
    low_col, high_col, wear_col = cols

    mats = {}
    for name, (color, metallic, roughness) in materials.items():
        mat = bpy.data.materials.new(name)
        mat.diffuse_color = color
        bsdf = mat.node_tree.nodes["Principled BSDF"]
        bsdf.inputs["Base Color"].default_value = color
        bsdf.inputs["Metallic"].default_value = metallic
        bsdf.inputs["Roughness"].default_value = roughness
        mats[name] = mat

    for part, builder, mat_name, detail in parts:
        low = builder()
        low.name = f"{part}_low"
        low.data.name = low.name
        low.data.materials.clear()
        low.data.materials.append(mats[mat_name])
        high = make_high(low, detail)
        for ob in (low, high):
            finish_mesh(ob)
        weight_normals(high)
        for ob in (low, high):
            for uc in ob.users_collection:
                uc.objects.unlink(ob)
        low_col.objects.link(low)
        high_col.objects.link(high)
        unwrap(low)
        print(f"[{tag}] {part}: low {tri_count(low)} tris, high {tri_count(high)} tris, mat {mat_name}")

    add_wear_volumes(wear_col, volumes)
    return low_col, high_col, wear_col


def render_previews(out_dir, high_col, low_col, wear_col, target_loc, ortho_scale, views):
    os.makedirs(out_dir, exist_ok=True)
    scene = bpy.context.scene
    scene.render.engine = 'BLENDER_WORKBENCH'
    scene.render.resolution_x, scene.render.resolution_y = 1600, 1000
    sh = scene.display.shading
    sh.light = 'STUDIO'
    sh.color_type = 'SINGLE'
    sh.single_color = (0.55, 0.55, 0.55)
    sh.show_cavity = True
    sh.cavity_type = 'BOTH'
    sh.show_specular_highlight = True

    target = link(bpy.data.objects.new("cam_target", None))
    target.location = target_loc
    cam = link(bpy.data.objects.new("cam", bpy.data.cameras.new("cam")))
    cam.data.type = 'ORTHO'
    cam.data.ortho_scale = ortho_scale
    c = cam.constraints.new('TRACK_TO')
    c.target = target
    c.track_axis = 'TRACK_NEGATIVE_Z'
    c.up_axis = 'UP_Y'
    scene.camera = cam

    wear_col.hide_render = True
    for col_name, show, hide in (("high", high_col, low_col), ("low", low_col, high_col)):
        show.hide_render, hide.hide_render = False, True
        for view, loc in views.items():
            cam.location = loc
            scene.render.filepath = os.path.join(out_dir, f"{col_name}_{view}.png")
            bpy.ops.render.render(write_still=True)
