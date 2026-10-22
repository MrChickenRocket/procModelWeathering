"""ArcRoyale character pieces (Blend JSON parts.json) to and from the mog export layout.

python tools/arcpieces.py export <out_export_dir> <suit> <piece dirs, atlases split by "|">
python tools/arcpieces.py deliver <export_dir> <paint_dir> <deliver_dir>
python tools/arcpieces.py fanout <deliver_dir> <delivered piece> <posed piece_dir>...

export: one "bone" per piece (LOD 0 parts), laid out on a grid so pieces never occlude each other in the AO bake;
every LOD 1 part whose mesh differs from its LOD 0 part gets its own <piece>__lod1 bone in the same atlas. LOD 2
merged meshes are left out, and so are Fabric parts. Each face's slot is its ArcRoyale material key (charlib/suits MATS).
deliver: the re-UV'd bone OBJs split back into the source parts, as parts.json (same names, piece space, uvs and
tris rewritten, "atlas" added; LOD 2 parts get "atlas": null) and one OBJ per part, plus each atlas's maps.
"""
import json
import os
import shutil
import sys

import numpy as np

GRID = 3.0  # studs between pieces; every piece is under 2 studs across
ARC = "C:/Repos/ArcRoyale/tools/blender/character"
PREFIX = {"hazmat": "hz_", "exo": "ex_", "samurai": "sm_", "astronaut": "as_"}


def load_mats(suit):
    """ArcRoyale's material keys {key: (rgb, material, tint)}: charlib's soldier table plus the suit's own keys."""
    import ast
    mats = {}
    for f in ("charlib.py", "suits.py"):
        src = open(os.path.join(ARC, f)).read()
        i = src.index("\nMATS = {") + 1
        mats.update(ast.literal_eval(src[src.index("{", i):src.index("\n}\n", i) + 2]))
    pre = PREFIX.get(suit)
    return {k: v for k, v in mats.items() if not k.startswith("_") and (k[2] != "_" or k.startswith(pre or "--"))}


def slot_of(q, mats):
    """Material key per triangle: Detail parts by their vertex colour among untinted keys, other parts by tint (then
    colour, since pads and pockets share a tint)."""
    def nearest(rgb, keys):
        return min(keys, key=lambda k: sum((a - b) ** 2 for a, b in zip(mats[k][0], rgb)))
    if "tc" in q:
        keys = [k for k, v in mats.items() if v[2] is None] or list(mats)
        per = [nearest(c, keys) for c in q["colors"]]
        return [per[i] for i in q["tc"]]
    keys = [k for k, v in mats.items() if v[2] == q.get("tint")] or list(mats)
    return [nearest(q["color"], keys)] * len(q["tris"])


def lod0_name(part):
    n = part["name"]
    return (n[4:] or "Detail") if n.startswith("LOD1") else n


def same_mesh(a, b):
    # Same topology within a hundredth of a stud (the vest's LOD 1 is its LOD 0 nudged 1 mm) shares the UVs.
    return (len(a["positions"]) == len(b["positions"]) and [t[0::3] for t in a["tris"]] == [t[0::3] for t in b["tris"]]
            and np.abs(np.subtract(a["positions"], b["positions"])).max() < 0.01)


def export(out, mech, groups, skip_fabric=True):
    """groups: lists of piece dirs; a piece's bone is its dir name without char_ and _<mech>. Fabric parts never take
    a SurfaceAppearance in game, so skip_fabric leaves them out of the atlases."""
    os.makedirs(out, exist_ok=True)
    mats = load_mats(mech)
    bones, layout, atlases, slots = [], {}, [], set()
    for group in groups:
        atlas = []
        for d in group:
            p = os.path.basename(os.path.normpath(d)).removeprefix("char_").replace(f"_{mech}", "")
            with open(os.path.join(d, "parts.json")) as f:
                doc = json.load(f)
            keep = [q for q in doc["parts"] if not (skip_fabric and q.get("material") == "Fabric")]
            lod0 = {q["name"]: q for q in keep if q["lod"] == 0}
            lod1 = [q for q in keep if q["lod"] == 1]
            reuse = {q["name"]: lod0_name(q) for q in lod1 if lod0_name(q) in lod0 and same_mesh(q, lod0[lod0_name(q)])}
            own = [q for q in lod1 if q["name"] not in reuse]
            for suffix, parts in (("", list(lod0.values())), ("__lod1", own)):
                if not parts:
                    continue
                name = p + suffix
                v, vn, lines, faces, cur = [], [], [], [], None
                # Degenerate and repeated triangles crash Blender's custom normals; they stay out of the bake and
                # deliver picks their UVs (skip: bone tri index -> None, or [written tri, corner order]).
                seen, skip, ti, written = {}, {}, 0, 0
                for q in parts:
                    for t, slot in zip(q["tris"], slot_of(q, mats)):
                        pis = [t[3 * k] + len(v) for k in range(3)]
                        key = tuple(sorted(pis))
                        ti += 1
                        if len(set(pis)) < 3:
                            skip[ti - 1] = None
                            continue
                        if key in seen:
                            ref, rp = seen[key]
                            skip[ti - 1] = [ref, [rp.index(x) for x in pis]]
                            continue
                        seen[key] = (written, pis)
                        written += 1
                        if slot != cur:
                            lines.append(f"usemtl {slot}")
                            cur = slot
                            slots.add(slot)
                        c = [t[3 * k:3 * k + 3] for k in range(3)]
                        lines.append("f " + " ".join(f"{pi + len(v) + 1}//{ni + len(vn) + 1}" for pi, _, ni in c))
                    faces.append([q["name"], len(q["tris"])])
                    v += q["positions"]
                    vn += q["normals"]
                body = [f"v {x} {y} {z}" for x, y, z in v] + [f"vn {x} {y} {z}" for x, y, z in vn] + lines
                with open(os.path.join(out, f"{name}.obj"), "w") as f:
                    f.write("\n".join(body) + "\n")
                x, z = (len(bones) % 6) * GRID, (len(bones) // 6) * GRID
                bones.append(dict(name=name, parent=None, length=1.0, cframe=[x, 0, z, 1, 0, 0, 0, 1, 0, 0, 0, 1]))
                layout[name] = dict(piece=p, src=os.path.abspath(d), faces=faces, reuse=reuse if not suffix else {},
                                    skip=skip)
                atlas.append(name)
        if atlas:
            atlases.append(atlas)
    with open(os.path.join(out, "rig.json"), "w") as f:
        json.dump(dict(mech=mech, space="Roblox studs, piece space; bones are pieces on a layout grid",
                       slots=sorted(slots), bones=bones), f, indent=1)
    with open(os.path.join(out, "layout.json"), "w") as f:
        json.dump(dict(atlases=atlases, bones=layout, mats={k: mats[k] for k in sorted(slots)}), f, indent=1)
    print(f"[arc] {mech}: {len(bones)} bones in {len(atlases)} atlases, slots {sorted(slots)} -> {out}")


def read_vt(path):
    vt, fvt = [], []
    with open(path) as f:
        for line in f:
            if line.startswith("vt "):
                vt.append([float(x) for x in line.split()[1:3]])
            elif line.startswith("f "):
                fvt.append([int(c.split("/")[1]) - 1 for c in line.split()[1:]])
    return np.array(vt), np.array(fvt)


def deliver(export_dir, paint, out):
    with open(os.path.join(export_dir, "layout.json")) as f:
        layout = json.load(f)
    with open(os.path.join(paint, "atlases.json")) as f:
        atlas_of = {os.path.splitext(k)[0]: v for k, v in json.load(f)["objs"].items()}
    docs, uv_of = {}, {}
    for name, info in layout["bones"].items():
        vt, fvt = read_vt(os.path.join(paint, f"{name}.obj"))
        tri_uv = vt[fvt]
        skip = {int(k): r for k, r in info.get("skip", {}).items()}
        if skip:
            full, w = [], 0
            for ti in range(len(tri_uv) + len(skip)):
                if ti not in skip:
                    full.append(tri_uv[w])
                    w += 1
                elif skip[ti] is None:
                    full.append(np.repeat(tri_uv[0][:1], 3, 0))  # zero area: any point
                else:
                    full.append(tri_uv[skip[ti][0]][skip[ti][1]])
            tri_uv = np.array(full)
        i = 0
        for part, n in info["faces"]:
            uv_of[(info["piece"], part)] = (tri_uv[i:i + n], atlas_of[name])
            i += n
        for lod1, lod0 in info["reuse"].items():
            uv_of[(info["piece"], lod1)] = uv_of[(info["piece"], lod0)]
        if info["piece"] not in docs:
            with open(os.path.join(info["src"], "parts.json")) as f:
                docs[info["piece"]] = json.load(f)
    for piece, doc in docs.items():
        d = os.path.join(out, "meshes", piece)
        os.makedirs(d, exist_ok=True)
        for q in doc["parts"]:
            if (piece, q["name"]) not in uv_of:
                q["atlas"] = None  # LOD 2: no wear maps; keep the plain material
                continue
            uv, k = uv_of[(piece, q["name"])]
            # parts.json UVs are Roblox's (origin top left, v down; tools/blender/lib.py writes 1 - v); OBJs keep v up.
            q["uvs"] = [[round(float(a), 6), round(1 - float(b), 6)] for a, b in uv.reshape(-1, 2)]
            q["tris"] = [[t[0], 3 * j, t[2], t[3], 3 * j + 1, t[5], t[6], 3 * j + 2, t[8]] for j, t in enumerate(q["tris"])]
            q["atlas"] = k
            lines = [f"# {piece}/{q['name']}: Roblox studs, piece space; UVs: atlas a{k}"]
            lines += [f"v {x} {y} {z}" for x, y, z in q["positions"]]
            lines += [f"vt {a} {1 - b:.6f}" for a, b in q["uvs"]]
            lines += [f"vn {x} {y} {z}" for x, y, z in q["normals"]]
            lines += ["f " + " ".join(f"{t[3 * c] + 1}/{t[3 * c + 1] + 1}/{t[3 * c + 2] + 1}" for c in range(3))
                      for t in q["tris"]]
            with open(os.path.join(d, f"{q['name']}.obj"), "w") as f:
                f.write("\n".join(lines) + "\n")
        with open(os.path.join(d, "parts.json"), "w") as f:
            json.dump(doc, f, separators=(",", ":"))
    for tier in ("minimal", "scarred"):
        src = os.path.join(paint, f"textures_{tier}")
        if os.path.isdir(src):
            dst = os.path.join(out, f"textures_{tier}")
            shutil.copytree(src, dst, dirs_exist_ok=True, ignore=shutil.ignore_patterns("emissive.png"))
            # color.png is the composite over the default part colours: a preview, never the in-game colour map.
            for k in os.listdir(dst):
                os.replace(os.path.join(dst, k, "color.png"), os.path.join(dst, k, "preview_color.png"))
    print(f"[arc] {len(docs)} pieces, {len(uv_of)} parts -> {out}")


def fanout(out, src_piece, dirs):
    """Posed copies of one mesh (glove flipbook frames): every part takes src_piece's delivered per-corner UVs and
    atlas, keeping its own position and normal indices, so the wear stays put as the frames swap."""
    with open(os.path.join(out, "meshes", src_piece, "parts.json")) as f:
        src = {q["name"]: q for q in json.load(f)["parts"]}
    for d in dirs:
        piece = os.path.basename(os.path.normpath(d)).removeprefix("char_")
        with open(os.path.join(d, "parts.json")) as f:
            doc = json.load(f)
        for q in doc["parts"]:
            r = src[q["name"]]
            assert [t[0::3] for t in q["tris"]] == [t[0::3] for t in r["tris"]], f"{piece}/{q['name']}: topology differs"
            q["atlas"] = r["atlas"]
            if r["atlas"] is None:
                continue
            q["uvs"] = r["uvs"]
            q["tris"] = [[t[0], u[1], t[2], t[3], u[4], t[5], t[6], u[7], t[8]] for t, u in zip(q["tris"], r["tris"])]
        os.makedirs(os.path.join(out, "meshes", piece), exist_ok=True)
        with open(os.path.join(out, "meshes", piece, "parts.json"), "w") as f:
            json.dump(doc, f, separators=(",", ":"))
    print(f"[arc] {src_piece} UVs -> {len(dirs)} poses")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "export":
        export(sys.argv[2], sys.argv[3], [g.split() for g in " ".join(sys.argv[4:]).split("|")])
    elif cmd == "fanout":
        fanout(sys.argv[2], sys.argv[3], sys.argv[4:])
    else:
        deliver(*sys.argv[2:5])
