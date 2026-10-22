"""Read MogCompany mech exports: rig.json plus one OBJ per bone (Roblox studs, bone space).

Roblox space is x right, y up, -z forward; pipeline space is x right, +y forward, +z up, scaled by m_per_stud.
"""
import json
import os

import numpy as np

RBX_TO_BLENDER = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], np.float64)


def load_rig(src):
    with open(os.path.join(src, "rig.json")) as f:
        rig = json.load(f)
    return rig, {b["name"]: _frame(b["cframe"]) for b in rig["bones"]}


def parse_obj(path):
    """Faces as (slot, (v0, v1, v2), (vt...), (vn...)), indices 0-based; vt is None when the OBJ has no UVs."""
    v, vn, faces, slot = [], [], [], None
    with open(path) as f:
        for line in f:
            tag, _, rest = line.partition(" ")
            if tag == "v":
                v.append([float(x) for x in rest.split()[:3]])
            elif tag == "vn":
                vn.append([float(x) for x in rest.split()[:3]])
            elif tag == "usemtl":
                slot = rest.strip()
            elif tag == "f":
                corners = [c.split("/") for c in rest.split()]
                assert len(corners) == 3, f"{path}: non-triangle face"
                faces.append((slot, tuple(int(c[0]) - 1 for c in corners),
                              tuple(int(c[1]) - 1 for c in corners) if corners[0][1] else None,
                              tuple(int(c[2]) - 1 for c in corners)))
    return dict(v=np.array(v, np.float64), vn=np.array(vn, np.float64), faces=faces)


def _frame(c):
    return np.array(c[3:12], np.float64).reshape(3, 3), np.array(c[:3], np.float64)


def mesh_objs(src, rig):
    """(name, obj path, frame) for every exported mesh: bone meshes in bone space, then piston stages in their
    own stage frames. Order is the import's face order, so export relies on it staying stable."""
    out = [(b["name"], os.path.join(src, f"{b['name']}.obj"), _frame(b["cframe"])) for b in rig["bones"]
           if os.path.exists(os.path.join(src, f"{b['name']}.obj"))]
    for pst in rig.get("pistons", []):
        for st in pst.get("stages", []):
            out.append((os.path.splitext(st["obj"])[0], os.path.join(src, st["obj"]), _frame(st["cframe"])))
    return out


def to_world(points, bone, m_per_stud, normals=False):
    R, p = bone
    w = points @ R.T if normals else points @ R.T + p
    w = w @ RBX_TO_BLENDER.T
    return w if normals else w * m_per_stud
