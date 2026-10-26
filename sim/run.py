"""python -m sim.run <bake_dir> <out_dir> [--age 0.6 --use 0.6 --humidity 0.3 --dirt 0.4 --dust 0.2 --seed 1]
    [--parts a,b] [--stack part=MAT_painted_steel,...] [--jobs N] [--mog assets/mogs/<mech>.json]
--stack overrides the blend's material slot (keyed by part, or by material on multi-material atlas parts).
Parts run in parallel processes (--jobs, default up to 6). Atlas parts (a matid map) run each material as its
own sub-part and merge them; --mog adds that mech's decals."""
import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, fields

import numpy as np
from scipy.spatial import cKDTree

from . import decals, materials
from .composite import shade, write_outputs
from .noise import fbm
from .wear import FOLD3D_FRAC, Params, Part, curvature3d, folds3d, simulate

MATID_SCALE = 32  # tools/bake.py

TOTAL_THREADS = os.cpu_count() or 8


def _init_worker(threads):
    import numba
    numba.set_num_threads(threads)


def run_atlas(part, stacks, prm, vols, mog, log):
    if mog and "decals" in mog:
        # Several atlas parts share the mog's marking budget by surface area.
        d = mog["decals"]
        d = dict(d, count=round(d["count"] * mog.get("area_frac", {}).get(part.name, 1.0)), m_per_stud=mog["m_per_stud"])
        decals.place(part, d, prm.seed, log=log)
    mid = np.round(part.maps["matid"] * MATID_SCALE - 0.5).astype(int)
    if mog and "glow_through" in mog:
        gt = mog["glow_through"]
        for cfg in gt if isinstance(gt, list) else [gt]:
            mid = glow_through(part, mid, cfg, mog["m_per_stud"], log)
    if mog and "stripes" in mog:
        mid = edge_stripes(part, mid, mog["stripes"], mog["m_per_stud"], log)
    if mog and "patterns" in mog:
        mid = patterns(part, mid, mog["patterns"], mog["m_per_stud"], log)
    N = len(part.idx)
    F, out = {}, {}
    for i, mat in enumerate(part.meta["materials"]):
        sel = mid == i
        if not sel.any():
            continue
        sub = part.subset(sel, stacks.get(mat, mat))
        f = simulate(sub, vols, prm, log=log)
        o = shade(sub, f, prm)
        n_sel = int(sel.sum())
        for src, dst in ((f, F), (o, out)):
            for k, v in src.items():
                if k == "layers" or not isinstance(v, np.ndarray) or v.ndim == 0 or len(v) != n_sel:
                    continue
                key = f"{k}_{mat}" if k.startswith("layer_") else k
                if key not in dst:
                    dst[key] = np.zeros((N,) + v.shape[1:], np.float32)
                dst[key][sel] = v
    return F, out


def glow_through(part, mid, cfg, m_per_stud, log):
    """Texels of cfg["from"] (an opaque lens dome) lying within within_studs of cfg["to"] texels (the glowing iris
    behind it) take the "to" material, so the glow shows through on the dome as a ring around a dark pupil.
    With lateral_studs, a texel needs a "to" texel within that distance measured along the glass surface."""
    mats = part.meta["materials"]
    if cfg["from"] not in mats or cfg["to"] not in mats:
        return mid
    a, b = mats.index(cfg["from"]), mats.index(cfg["to"])
    src, dst = np.flatnonzero(mid == a), np.flatnonzero(mid == b)
    if not len(src) or not len(dst):
        return mid
    reach = cfg["within_studs"] * m_per_stud
    if "lateral_studs" in cfg:
        # Thick glass: measure across the glass surface so rings behind it stay rings instead of filling the disc.
        d, j = cKDTree(part.pos[dst]).query(part.pos[src], k=32, distance_upper_bound=reach)
        ok = np.isfinite(d)
        off = part.pos[dst][np.minimum(j, len(dst) - 1)] - part.pos[src][:, None]
        n = part.maps["normal_obj"][src][:, None].astype(np.float64)
        lat = np.linalg.norm(off - n * (off * n).sum(-1, keepdims=True), axis=-1)
        hit = src[(np.where(ok, lat, np.inf).min(1)) <= cfg["lateral_studs"] * m_per_stud]
    else:
        d, _ = cKDTree(part.pos[dst]).query(part.pos[src], distance_upper_bound=reach)
        hit = src[np.isfinite(d)]
    mid = mid.copy()
    mid[hit] = b
    log(f"[sim] {part.name}: {len(hit):,} of {len(src):,} {cfg['from']} texels glow through as {cfg['to']}")
    return mid


RBX_TO_BLENDER = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], np.float64)


def edge_stripes(part, mid, rules, m_per_stud, log):
    """Per rule {from: [stacks], to, width_studs, toward: Roblox direction}: each "from" stack is one thin arched plate
    (a band); texels within width_studs of its edge on the "toward" side take the "to" material. The plate's thin axis
    is its smallest principal axis."""
    mats = part.meta["materials"]
    mid = mid.copy()
    for r in rules:
        toward = RBX_TO_BLENDER @ np.array(r["toward"], np.float64)
        for src in r["from"]:
            if src not in mats:
                continue
            sel = np.flatnonzero(mid == mats.index(src))
            if len(sel) < 10:
                continue
            c = part.pos[sel] - part.pos[sel].mean(0)
            e = np.linalg.eigh(c.T @ c)[1][:, 0]
            proj = c @ (e if e @ toward > 0 else -e)
            hit = sel[proj > proj.max() - r["width_studs"] * m_per_stud]
            if r["to"] not in mats:
                mats.append(r["to"])  # in-memory only: a stack no face was assigned to
            mid[hit] = mats.index(r["to"])
            log(f"[sim] {part.name}: {len(hit):,} of {len(sel):,} {src} texels striped {r['to']}")
    return mid


def patterns(part, mid, rules, m_per_stud, log):
    """Per rule {from: [stacks], to, mode, period_studs, fill, box?: [lo, hi] Roblox studs, seed?}: paint texels inside
    the box take the "to" material where the pattern is on. mode "bands": noise-warped stripes across axis (Roblox
    direction), fill = stripe fraction, with tapering width. mode "spots": noise blobs covering about fill; octaves 1
    gives round blobs, 3 ragged ones."""
    mats = part.meta["materials"]
    mid = mid.copy()
    P = part.pos
    for r in rules:
        sel = np.isin(mid, [mats.index(m) for m in r["from"] if m in mats])
        if "box" in r:
            a, b = (np.array(r["box"], np.float64) @ RBX_TO_BLENDER.T) * m_per_stud
            sel &= np.all((P >= np.minimum(a, b)) & (P <= np.maximum(a, b)), axis=1)
        sel = np.flatnonzero(sel)
        if not len(sel):
            continue
        q, per, seed = P[sel], r["period_studs"] * m_per_stud, r.get("seed", 7)
        if r.get("mode", "bands") == "bands":
            axis = RBX_TO_BLENDER @ np.array(r["axis"], np.float64)
            t = q @ (axis / np.linalg.norm(axis)) / per + 1.2 * (fbm(q, 0.5 / per, 3, seed) - 0.5)
            width = r["fill"] * (0.5 + fbm(q, 0.7 / per, 2, seed + 1))
            on = (t - np.floor(t)) < width
        else:
            n = fbm(q, 1.0 / per, r.get("octaves", 3), seed)
            on = n > np.quantile(n, 1 - r["fill"])
        if r["to"] not in mats:
            mats.append(r["to"])  # in-memory only: a stack no face was assigned to
        mid[sel[on]] = mats.index(r["to"])
        log(f"[sim] {part.name}: {int(on.sum()):,} of {len(sel):,} texels patterned {r['to']}")
    return mid


def burn_field(P, burns, m_per_stud):
    """Scorch from fixed heat sources: [{"at": [x, y, z] Roblox studs, "radius_studs": r}], full strength within
    a third of r, gone at r."""
    out = np.zeros(len(P), np.float32)
    for b in burns:
        c = (RBX_TO_BLENDER @ np.array(b["at"], np.float64)) * m_per_stud
        d = np.linalg.norm(P - c, axis=1) / (b["radius_studs"] * m_per_stud)
        out = np.maximum(out, b.get("strength", 1.0) * np.clip((1 - d) / (1 - 1 / 3), 0, 1) ** 1.5)
    return out


def run_part(bake_dir, out_dir, name, material, prm, vols, stacks=None, mog=None):
    lines = []
    t = time.time()
    if mog and "stacks" in mog:
        materials.register(mog["stacks"])
    if mog and "soot" in mog:
        # In place: composite imported the SOOT dict by name. Ice mogs turn it into frost.
        materials.SOOT.update(color=tuple(mog["soot"]["color"]), rough=mog["soot"].get("rough", materials.SOOT["rough"]))
    part = Part(os.path.join(bake_dir, name), material)
    if prm.ridge > 0 or prm.cavity > 0:
        # On the whole atlas before it splits by material, so seams between pieces of different slots register.
        n = part.maps["normal_obj"].astype(np.float64)
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
        part.maps["curv3d"], part.maps["occl3d"] = curvature3d(part.pos, n, prm.ridge_mm / 1000)
        part.maps["fold_cvx3d"], part.maps["fold_ccv3d"] = folds3d(part.pos, n, FOLD3D_FRAC * prm.ridge_mm / 1000)
    for key, d in (("grime", materials.GRIME), ("dust", materials.DUST)):
        # In place, like SOOT: composite imported these dicts by name. Ice characters turn grime into rime.
        if mog and key in mog:
            d.update(color=tuple(mog[key]["color"]), rough=mog[key].get("rough", d["rough"]))
    if mog and "haze" in mog:
        # Grime rising from the ground: full at the feet, gone at below_studs (Roblox y).
        h = mog["haze"]
        y = part.pos[:, 2] / mog["m_per_stud"]
        part.maps["haze"] = (h.get("strength", 0.5) * np.clip(1 - y / h["below_studs"], 0, 1)).astype(np.float32)
    if mog and "burns" in mog:
        part.maps["burn"] = burn_field(part.pos, mog["burns"], mog["m_per_stud"])
    if "matid" in part.maps and len(part.meta.get("materials", [])) > 1:
        F, out = run_atlas(part, stacks or {}, prm, vols, mog, lines.append)
    else:
        F = simulate(part, vols, prm, log=lines.append)
        out = shade(part, F, prm)
    t_sim = time.time() - t
    write_outputs(part, F, out, os.path.join(out_dir, name))
    lines.append(f"[sim] {name}: sim {t_sim:.1f}s, total {time.time() - t:.1f}s")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bake_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--parts", default="")
    ap.add_argument("--stack", default="")
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--mog", default="")
    ap.add_argument("--tier", default="", help="parameter preset from the --mog config's tiers; explicit flags still win")
    for f in fields(Params):
        ap.add_argument(f"--{f.name}", type=type(f.default), default=f.default)
    pre, _ = ap.parse_known_args()
    if pre.tier:
        with open(pre.mog) as f:
            ap.set_defaults(**json.load(f)["tiers"][pre.tier])
    args = ap.parse_args()
    prm = Params(**{f.name: getattr(args, f.name) for f in fields(Params)})

    with open(os.path.join(args.bake_dir, "wear_volumes.json")) as f:
        vols = json.load(f)
    stacks = dict(kv.split("=") for kv in args.stack.split(",") if kv)
    mog = None
    if args.mog:
        with open(args.mog) as f:
            mog = json.load(f)
    parts = [p for p in args.parts.split(",") if p] or sorted(
        d for d in os.listdir(args.bake_dir) if os.path.isfile(os.path.join(args.bake_dir, d, "maps.npz")))
    if mog:
        area = {}
        for p in parts:
            with open(os.path.join(args.bake_dir, p, "meta.json")) as f:
                area[p] = json.load(f).get("area_m2", 1.0)
        mog["area_frac"] = {p: a / sum(area.values()) for p, a in area.items()}

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "params.json"), "w") as f:
        json.dump(asdict(prm), f, indent=2)
    t0 = time.time()
    jobs = max(1, min(args.jobs, len(parts)))
    # Each worker runs numba kernels; split the cores so workers don't oversubscribe them.
    with ProcessPoolExecutor(jobs, initializer=_init_worker, initargs=(max(1, TOTAL_THREADS // jobs),)) as ex:
        futs = [ex.submit(run_part, args.bake_dir, args.out_dir, n, stacks.get(n), prm, vols, stacks, mog) for n in parts]
        for fut in futs:
            print(fut.result(), flush=True)
    print(f"[sim] {len(parts)} parts in {time.time() - t0:.1f}s ({jobs} jobs)")


if __name__ == "__main__":
    main()
