"""Time-stepped wear simulation over the covered texels of one baked part.

All spatial queries use texel object-space positions (3D), never UV adjacency, so chips,
scratches and noise run continuously across UV seams.
"""
import copy
import json
import os
import zlib
from dataclasses import dataclass, replace

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from . import texsample
from .materials import BASES, STACKS
from .noise import cell_seeds, fbm, smoothstep

K_ABRASION = 10.0
IMPACTS_PER_M2 = 40_000
SCRATCHES_PER_M2 = 15_000
K_RUST = 3.0
CHIP_R_MIN_M, CHIP_R_MAX_M, CHIP_ALPHA = 0.00012, 0.0025, 1.8
SCRATCH_LEN_MEDIAN_M = 0.008
SCRATCH_WIDTH_M = (0.00004, 0.00012)
BASE_CONTACT = 0.1
# Abrasion is calibrated so a 25 um film with resist 1 wears like the original single-coat model.
REF_FILM_UM = 25.0
SCRATCH_DEPTH_UM = 25.0
VORONOI_CELL = 0.5
VORONOI_REACH = 2.2
# Worn-through areas on voronoi layers are snapped to whole cells of this size, so wear edges flake.
FLAKE_CELL_M = 0.0006
DRAW_AXIS = np.array([0.0, 1.0, 0.0])
FOLD_DEG = (6.0, 25.0)
FOLD_SHARP_DEG = 35.0
FOLD_RADIUS_PX = 3
FOLD_FINE_RADIUS_PX = 1
FOLD3D_FRAC = 0.4  # fold search radius as a fraction of ridge_mm


@dataclass
class Params:
    age: float = 0.6        # 0 = new, 1 = end of service life
    use: float = 0.6        # handling intensity
    humidity: float = 0.3
    dirt: float = 0.4
    dust: float = 0.2
    seed: int = 1
    steps: int = 12
    scratches: float = 0.0     # density multiplier for simulated hairline scratches; off by default
    detail: float = 1.0        # strength of mask-stamped handling detail (scratches, prints, smudges)
    detail_scale: float = 1.0  # multiplies detail mask tile sizes; raise for vehicle-sized assets
    scratch_cut: float = 0.0   # how deep stamped detail scratches cut through paint films; 0 = surface marks only
    scorch: float = 0.0        # fire damage: soot patches, blistered paint, burnt-through spots with heat tint
    streaks: float = 0.0       # grime runs down vertical faces
    crease: float = 0.0        # extra wear on the lips of panel gaps and creases, plus paint lifting inside them
    edge_band: float = 0.0     # broad, patchy wear along hard edges and sharp folds (armour shoulders, chamfers)
    fine_edge: float = 0.0     # hairline wear along shallow folds (faceted shells), where a broad band reads as blotches
    edge_band_mm: float = 4.0  # band half-width; scale with the asset (4 mm at the mech's 40 mm/stud is 0.1 stud)
    ridge: float = 0.0         # crisp worn line on every convex bevel and ridge from 3D curvature, smooth-shaded or not
    cavity: float = 0.0        # dark grime line in every concave crease, panel line and plate-on-plate seam
    ridge_mm: float = 4.0      # curvature measuring radius; about the bevel radius, scale with the asset
    rivets: float = 0.0        # domed rivet rows inset from plate rims (painted layers only); their crowns chip
    rivet_mm: float = 2.6      # rivet head radius
    rivet_pitch_mm: float = 14.0
    rivet_inset_mm: float = 4.5
    seams: float = 0.0         # scribed panel line further in from the rim on plates big enough to hold one
    seam_inset_mm: float = 9.0
    seam_mm: float = 0.6       # groove half-width
    fray: float = 0.0          # fabric bases only: bleached, fuzzy, broken-up threads along torn and folded edges
    min_feature_px: float = 0.0  # chip radius and micro-noise floor in texels, for coarse atlases; a stack's weather may set it


class Part:
    def __init__(self, bake_dir, material=None):
        with open(os.path.join(bake_dir, "meta.json")) as f:
            self.meta = json.load(f)
        self.name = self.meta["part"]
        self.material = material or self.meta["material"]
        self.size = S = self.meta["size"]
        d = np.load(os.path.join(bake_dir, "maps.npz"))
        self.mask = d["mask"] > 0.5
        self.idx = np.flatnonzero(self.mask)
        self.maps = {}
        for k in d.files:
            a = d[k]
            a = a.reshape(S * S, -1) if a.ndim == 3 else a.reshape(S * S)
            self.maps[k] = a[self.idx].astype(np.float32)
        self.pos = self.maps["position"].astype(np.float64)
        # Texel size varies per island (view-weighted packing), so it is tracked per texel.
        self.texel_img = _texel_size_img(d["position"].astype(np.float32), self.mask)
        self.texel = self.texel_img.reshape(-1)[self.idx]
        for k, img in _fold_imgs(d["normal_obj"], d["position"], self.mask).items():
            self.maps[k] = img.reshape(-1)[self.idx]
        self.seed_salt = zlib.crc32(self.name.encode()) % 7919

    def subset(self, sel, material):
        """The texels of one material on a multi-material (atlas) part, simulated as a part of their own."""
        sub = copy.copy(self)
        sub.idx = self.idx[sel]
        sub.maps = {k: v[sel] for k, v in self.maps.items()}
        sub.pos, sub.texel = self.pos[sel], self.texel[sel]
        sub.material = material
        sub.name = f"{self.name}/{material}"
        sub.seed_salt = zlib.crc32(sub.name.encode()) % 7919
        return sub


def _texel_size_img(pos, mask):
    """Per-texel world size in meters: mean of available u and v neighbor distances."""
    total = np.zeros(mask.shape, np.float32)
    count = np.zeros(mask.shape, np.float32)
    for axis in (0, 1):
        for shift in (1, -1):
            nb = np.roll(pos, shift, axis)
            ok = mask & np.roll(mask, shift, axis)
            total += np.where(ok, np.linalg.norm(pos - nb, axis=-1), 0)
            count += ok
    size = np.where(count > 0, total / np.maximum(count, 1), 0)
    fallback = np.median(size[count > 0])
    return np.where(mask & (count == 0), fallback, size).astype(np.float32)


def _fold_imgs(normal, pos, mask):
    """Fold lines the AO bakes are too blunt to see, from normal jumps between UV-adjacent texels. Folds that land
    on island borders are missed; those are hard edges, which convex_small already covers.
      fold_fine     shallow convex folds (faceting), a thin line: these get hairline edge wear only
      fold_sharp    convex folds of FOLD_SHARP_DEG and up: seeds for the broad edge band, like hard corners
      fold_concave  concave folds widened to a band: paint lifting and grime"""
    n = normal.astype(np.float32)
    n /= np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-9)
    p = pos.astype(np.float32)
    convex = np.zeros(mask.shape, np.float32)
    concave = np.zeros(mask.shape, np.float32)
    for axis in (0, 1):
        nn, pn = np.roll(n, -1, axis), np.roll(p, -1, axis)
        ok = mask & np.roll(mask, -1, axis)
        ang = np.where(ok, np.degrees(np.arccos(np.clip((n * nn).sum(-1), -1, 1))), 0).astype(np.float32)
        out = ((nn - n) * (pn - p)).sum(-1) > 0
        convex = np.maximum(convex, np.where(out, ang, 0))
        concave = np.maximum(concave, np.where(out, 0, ang))

    def widen(f, r):
        size = 2 * r + 1
        return np.where(mask, ndimage.uniform_filter(ndimage.grey_dilation(f, size=size), size=size), 0)

    lo, hi = FOLD_DEG
    shallow = smoothstep(lo, lo + 8, convex) * (1 - smoothstep(FOLD_SHARP_DEG - 8, FOLD_SHARP_DEG, convex))
    return dict(fold_fine=widen(shallow.astype(np.float32), FOLD_FINE_RADIUS_PX),
                fold_sharp=widen(smoothstep(FOLD_SHARP_DEG - 8, FOLD_SHARP_DEG, convex).astype(np.float32), 1),
                fold_concave=widen(smoothstep(lo, hi, concave).astype(np.float32), FOLD_RADIUS_PX))


def curvature3d(P, n, r_m):
    """(curv, occl) at scale r_m from texel positions and smooth object normals, across UV seams.
    curv: signed mean curvature times r_m (convex > 0); sees smooth-shaded bevels and hard edges alike. A bevel of
    radius R reads about r_m / (2R).
    occl: share of the surface within r_m that rises in front of the tangent plane, from any piece; creases and the
    gap under an overhanging plate edge."""
    v = r_m / 3
    key = np.floor(P / v).astype(np.int64)
    key -= key.min(0)
    span = key.max(0) + 1
    _, rep = np.unique((key[:, 0] * span[1] + key[:, 1]) * span[2] + key[:, 2], return_index=True)
    S, nS = P[rep], n[rep]
    tree = cKDTree(S)
    d, j = tree.query(S, k=48, distance_upper_bound=r_m, workers=-1)
    ok = np.isfinite(d) & (d > 0)
    jj = np.where(ok, j, 0)
    nj, dp = nS[jj], S[jj] - S[:, None]
    g = np.exp(-(d / (0.6 * r_m)) ** 2)
    rise = np.einsum("ikc,ic->ik", dp, nS) / np.maximum(d, 1e-12)
    oS = np.where(ok, g * (rise > 0.35), 0).sum(1) / np.maximum(np.where(ok, g, 0).sum(1), 1e-12)
    # Back faces of thin plates fall inside the radius and would read as a sharp ridge.
    ok &= np.einsum("ikc,ic->ik", nj, nS) > -0.3
    kij = np.einsum("ikc,ikc->ik", nj - nS[:, None], dp) / np.maximum(d, 1e-12) ** 2
    w = np.where(ok, g, 0)
    cS = (w * kij).sum(1) / np.maximum(w.sum(1), 1e-12) * r_m
    curv = np.empty(len(P), np.float32)
    occl = np.empty(len(P), np.float32)
    for s in range(0, len(P), 1_000_000):
        q = slice(s, s + 1_000_000)
        d, j = tree.query(P[q], k=4, workers=-1)
        w = np.exp(-(d / v) ** 2) * (np.einsum("ikc,ic->ik", nS[j], n[q]) > 0.5) + 1e-9
        curv[q] = (w * cS[j]).sum(1) / w.sum(1)
        occl[q] = (w * oS[j]).sum(1) / w.sum(1)
    return curv, occl


def folds3d(P, n, r_m):
    """Largest convex and concave normal turn in degrees within r_m: hard facet folds and chamfers that the r_m-scale
    curvature averages away, as a line about 2 r_m wide. Seams between separate pieces count when they touch."""
    v = r_m / 2
    key = np.floor(P / v).astype(np.int64)
    key -= key.min(0)
    span = key.max(0) + 1
    _, rep = np.unique((key[:, 0] * span[1] + key[:, 1]) * span[2] + key[:, 2], return_index=True)
    S, nS = P[rep], n[rep]
    tree = cKDTree(S)
    cvx = np.zeros(len(S), np.float32)
    ccv = np.zeros(len(S), np.float32)
    for s in range(0, len(S), 500_000):
        q = slice(s, s + 500_000)
        d, j = tree.query(S[q], k=24, distance_upper_bound=r_m, workers=-1)
        ok = np.isfinite(d)
        jj = np.where(ok, j, 0)
        nj = nS[jj]
        cos = np.einsum("ikc,ic->ik", nj, nS[q])
        # Back faces of thin plates: see curvature3d.
        ok &= cos > -0.3
        ang = np.where(ok, np.degrees(np.arccos(np.clip(cos, -1, 1))), 0)
        out = np.einsum("ikc,ikc->ik", nj - nS[q][:, None], S[jj] - S[q][:, None]) > 0
        cvx[q] = np.where(out, ang, 0).max(1)
        ccv[q] = np.where(out, 0, ang).max(1)
    _, j = tree.query(P, k=1, workers=-1)
    return cvx[j], ccv[j]


def rim_distance(P, n, seeds, reach_m):
    """(distance, seed) to the nearest seed texel on the same side of the part: inf and -1 beyond reach_m."""
    sp = np.flatnonzero(seeds)
    if not len(sp):
        return np.full(len(P), np.inf), np.full(len(P), -1)
    sp = sp[::max(1, len(sp) // 400_000)]
    d, j = cKDTree(P[sp]).query(P, k=4, distance_upper_bound=reach_m, workers=-1)
    ok = np.isfinite(d)
    jj = np.where(ok, j, 0)
    ok &= np.einsum("ikc,ic->ik", n[sp][jj], n) > 0.3
    d = np.where(ok, d, np.inf)
    k = d.argmin(1)
    best = d[np.arange(len(P)), k]
    return best, np.where(np.isfinite(best), sp[jj[np.arange(len(P)), k]], -1)


def _straight(P, idx, r):
    """Per index: how close the seed texels within r of it lie to a line (1 = straight rim, 0 = blob or corner)."""
    tree = cKDTree(P[idx])
    out = np.zeros(len(idx), np.float32)
    for s in range(0, len(idx), 100_000):
        q = slice(s, s + 100_000)
        _, j = tree.query(P[idx[q]], k=24, distance_upper_bound=r, workers=-1)
        ok = j < len(idx)
        Q = P[idx][np.where(ok, j, 0)]
        w = ok[..., None].astype(np.float64)
        mu = (Q * w).sum(1) / np.maximum(w.sum(1), 1)
        X = (Q - mu[:, None]) * w
        ev = np.linalg.eigvalsh(np.einsum("ikc,ikd->icd", X, X))
        out[q] = 1 - np.clip(ev[:, 1] / np.maximum(ev[:, 2], 1e-30) / 0.05, 0, 1)
    return out


def _spread(idx, P, pitch, rng):
    """Greedy pick of idx in random order, keeping points at least pitch apart."""
    keep, cells = [], {}
    for i in rng.permutation(idx):
        c = tuple(np.floor(P[i] / pitch).astype(np.int64))
        if any(np.linalg.norm(P[j] - P[i]) < pitch for dc in np.ndindex(3, 3, 3)
               for j in cells.get((c[0] + dc[0] - 1, c[1] + dc[1] - 1, c[2] + dc[2] - 1), ())):
            continue
        cells.setdefault(c, []).append(i)
        keep.append(i)
    return np.array(keep, np.int64)


def hardware(P, n, m, curv, prm, texel, rng, seed):
    """Rivet rows and panel seams placed from plate rims (3D folds and bevels). Returns (relief_mm, grime, crown):
    crown is the rivet top, where paint chips first."""
    N = len(P)
    relief, grime, crown = (np.zeros(N, np.float32) for _ in range(3))
    # A plate rim turns the surface 60 deg or more within ridge_mm (a chamfered corner is two 45 deg folds); a
    # single 35 deg facet split inside a curved plate is not a rim.
    seeds = folds3d(P, n, prm.ridge_mm / 1000)[0] > 60
    inset_r, inset_s = prm.rivet_inset_mm / 1000, prm.seam_inset_mm / 1000
    d, near_seed = rim_distance(P, n, seeds, 3 * max(inset_r, inset_s))
    flat = (np.abs(curv) < 0.15) & (m["fold_cvx3d"] < 15)
    tol = max(float(np.median(texel)), 0.12 * inset_r)
    if prm.rivets > 0:
        R = prm.rivet_mm / 1000
        cand = np.flatnonzero(flat & (np.abs(d - inset_r) < tol))
        cen = _spread(cand, P, prm.rivet_pitch_mm / 1000, rng) if len(cand) else np.zeros(0, np.int64)
        if len(cen):
            dist, j = cKDTree(P[cen]).query(P, k=1, distance_upper_bound=1.4 * R, workers=-1)
            hit = np.isfinite(dist)
            jj = cen[np.where(hit, j, 0)]
            # Same surface only: not the back of a thin plate or a neighbour piece just behind.
            hit &= (np.einsum("ic,ic->i", n[jj], n) > 0.7) & (np.abs(np.einsum("ic,ic->i", P - P[jj], n[jj])) < 0.5 * R)
            rel = np.where(hit, dist / R, 9.0)
            relief += (prm.rivets * 0.45 * R * 1000 * np.sqrt(np.clip(1 - rel ** 2, 0, 1))).astype(np.float32)
            grime = np.maximum(grime, prm.rivets * 0.7 * smoothstep(1.4, 1.0, rel) * smoothstep(0.8, 1.0, rel))
            worn = smoothstep(0.35, 0.75, fbm(P[jj], 40, 1, seed + 74))
            crown = (smoothstep(0.75, 0.25, rel) * worn * (0.6 + 0.8 * fbm(P, 2500, 2, seed + 75))).astype(np.float32)
    if prm.seams > 0:
        # Only plates deep enough to leave a margin inside the line get one.
        deep = np.flatnonzero(d > 2.2 * inset_s)
        line = flat & (np.abs(d - inset_s) < prm.seam_mm / 1000 + 0.5 * tol)
        if len(deep) and line.any():
            li = np.flatnonzero(line)
            near = np.isfinite(cKDTree(P[deep[::max(1, len(deep) // 300_000)]]).query(
                P[li], k=1, distance_upper_bound=1.6 * inset_s, workers=-1)[0])
            # Only alongside straight stretches of rim: offsets of tabs, slots and corners wobble.
            ns = near_seed[li[near]]
            useq, inv = np.unique(ns, return_inverse=True)
            straight = _straight(P, useq, 2.5 * inset_s)[inv] > 0.5
            near[np.flatnonzero(near)[~straight]] = False
            g = np.zeros(N, np.float32)
            g[li[near]] = smoothstep(prm.seam_mm / 1000 + 0.5 * tol, 0.3 * prm.seam_mm / 1000, np.abs(d[li[near]] - inset_s))
            relief -= (prm.seams * 0.35 * g).astype(np.float32)
            grime = np.maximum(grime, prm.seams * 0.8 * g * (0.7 + 0.3 * fbm(P, 400, 2, seed + 76)))
    return relief, grime.astype(np.float32), crown


def edge_band_field(P, n, seeds, width_m):
    """exp(-(d/w)^2) of the 3D distance to the nearest seed texel on the same side of the part."""
    sp = np.flatnonzero(seeds)
    if not len(sp):
        return np.zeros(len(P), np.float32)
    sp = sp[::max(1, len(sp) // 400_000)]
    tree = cKDTree(P[sp])
    d, j = tree.query(P, k=4, distance_upper_bound=2.5 * width_m, workers=-1)
    ok = np.isfinite(d)
    jj = np.where(ok, j, 0)
    # Seeds on the far side of a thin plate are close in 3D but belong to a different surface.
    ok &= np.einsum("ikc,ic->ik", n[sp][jj], n) > 0.3
    dmin = np.where(ok, d, np.inf).min(1)
    return np.exp(-(dmin / width_m) ** 2).astype(np.float32)


def _tangent(v, n):
    v = v - n * np.einsum("ij,ij->i", v, n)[:, None]
    return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)


def _ball_entries(tree, centers, radii):
    lists = tree.query_ball_point(centers, radii, workers=-1, return_sorted=False)
    lens = np.fromiter((len(l) for l in lists), np.int64, len(lists))
    if lens.sum() == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    return np.concatenate([np.asarray(l, np.int64) for l in lists if l]), np.repeat(np.arange(len(lists)), lens)


def scorch_field(P, n, prm, seed):
    """Burn intensity 0..1. Unsalted seed, so a fire crosses material boundaries on an atlas part."""
    patches = smoothstep(0.52, 0.76, fbm(P, 28, 4, seed + 50))
    # Flame tongues: noise stretched 3x along up (+Z).
    licks = smoothstep(0.35, 0.75, fbm(P * np.array([1.0, 1.0, 0.33]), 80, 3, seed + 51))
    under = np.clip(-n[:, 2], 0, 1)
    burn = patches * (0.55 + 0.6 * licks) + 0.3 * under * smoothstep(0.4, 0.7, fbm(P, 20, 3, seed + 52))
    return np.clip(prm.scorch * burn * (0.6 + 0.8 * fbm(P, 90, 3, seed + 53)), 0, 1).astype(np.float32)


def streak_field(P, n, m, prm, seed):
    runs = smoothstep(0.5, 0.78, fbm(P * np.array([1.0, 1.0, 0.08]), 220, 3, seed + 60))
    vertical = (1 - np.abs(n[:, 2])) ** 2
    return np.clip(prm.streaks * runs * vertical * (0.4 + 0.6 * np.clip(m["ao_assembled"], 0, 1)), 0, 1).astype(np.float32)


def contact_field(P, vols):
    free = np.ones(len(P), np.float32)
    for v in vols:
        M = np.asarray(v["world_to_local"])
        local = P @ M[:3, :3].T + M[:3, 3]
        free *= 1 - v["weight"] * smoothstep(1.25, 0.85, np.abs(local).max(1)).astype(np.float32)
    return BASE_CONTACT + (1 - BASE_CONTACT) * (1 - free)


def _pick(rng, cdf, count):
    """Weighted texel sampling from a precomputed CDF; rng.choice(p=...) rebuilds it on every call."""
    return np.minimum(np.searchsorted(cdf, rng.random(count) * cdf[-1], side="right"), len(cdf) - 1)


def _chip_r2(r_min):
    a = CHIP_ALPHA
    return a * r_min ** a / (2 - a) * (CHIP_R_MAX_M ** (2 - a) - r_min ** (2 - a))


def _consume(T, idx, amount_um, layers, use_resist=True):
    """Remove amount_um of film from the top down at texels idx (unique); hard layers absorb more."""
    for k in range(len(layers) - 1, -1, -1):
        r = layers[k]["resist"] if use_resist else 1.0
        t = T[idx, k]
        used = np.minimum(amount_um, t * r)
        T[idx, k] = t - used / r
        amount_um = amount_um - used


def _impacts(F, ctx, count):
    P, n, tree, rng, layers, base = ctx["P"], ctx["n"], ctx["tree"], ctx["rng"], ctx["layers"], ctx["base"]
    sites = _pick(rng, ctx["impact_cdf"], count)
    r = np.minimum(ctx["chip_r_min"] * (1 - rng.random(count)) ** (-1 / CHIP_ALPHA), CHIP_R_MAX_M)
    reach_r = VORONOI_REACH if any(l["profile"] == "voronoi" for l in layers) else 1.6
    idx, cid = _ball_entries(tree, P[sites], r * reach_r)
    if not len(idx):
        return
    center = P[sites][cid]
    d = np.linalg.norm(P[idx] - center, axis=1)
    rr = r[cid] * (0.45 + 1.1 * fbm(P[idx], 1.5 / r[cid], 3, ctx["seed"] + 10))
    facing = np.einsum("ij,ij->i", n[idx], n[sites][cid]) > -0.2
    hit = facing & (d < rr)
    ragged = np.where(hit, 1 - (d / rr) ** 2, 0).astype(np.float32)
    np.maximum.at(F["impact"], idx[hit], ragged[hit])
    np.minimum.at(F["dent_mm"], idx[hit], -base["dent_mm"] * ragged[hit])
    if not layers:
        return
    prof = {"ragged": ragged}
    if any(l["profile"] == "voronoi" for l in layers):
        seeds = cell_seeds(P[idx], r[cid] * VORONOI_CELL, ctx["seed"] + 20)
        ds = np.linalg.norm(seeds - center, axis=1)
        rv = r[cid] * VORONOI_REACH
        prof["voronoi"] = np.where(facing & (ds < rv), 1 - (ds / rv) ** 2, 0).astype(np.float32)
    # A stronger hit reaches deeper; a layer flakes where the hit reaches its mid-depth and it lets go.
    reach = rng.uniform(0.6, 1.6, count)[cid]
    T, total = F["T"], ctx["total_um"]
    depth = 0.0
    above_gone = np.ones(len(idx), bool)
    for k in range(len(layers) - 1, -1, -1):
        lay = layers[k]
        mid = 0.8 * (depth + lay["thickness_um"] / 2) / total
        depth += lay["thickness_um"]
        was_gone = T[idx, k] <= 0
        lets_go = (rng.random(count) < lay["chip"])[cid]
        flake = above_gone & lets_go & (reach * prof[lay["profile"]] > mid)
        T[idx[flake], k] = 0
        above_gone &= flake | was_gone


def _scratches(F, ctx, count):
    P, n, tree, rng, texel = ctx["P"], ctx["n"], ctx["tree"], ctx["rng"], ctx["texel"]
    start = _pick(rng, ctx["scratch_cdf"], count)
    length = np.exp(rng.normal(np.log(SCRATCH_LEN_MEDIAN_M), 0.7, count))
    width = rng.uniform(SCRATCH_WIDTH_M[0], SCRATCH_WIDTH_M[1], count)
    depth = rng.uniform(0.2, 1.0, count).astype(np.float32)
    along_draw = rng.random(count) < 0.6
    dirs = np.where(along_draw[:, None], DRAW_AXIS * rng.choice([-1.0, 1.0], count)[:, None],
                    rng.normal(size=(count, 3)))
    cur, cn = P[start].copy(), n[start].astype(np.float64)
    dirs = _tangent(dirs, cn)
    step = np.maximum(texel[start], 0.6 * width) * 1.5
    alive = np.ones(count, bool)
    traveled = np.zeros(count)
    pts, sids = [start], [np.arange(count)]
    while True:
        alive &= traveled < length
        a = np.flatnonzero(alive)
        if not len(a):
            break
        dist, j = tree.query(cur[a] + dirs[a] * step[a, None], workers=-1)
        traveled[a] += step[a]
        # Stop at edges (normal turns) or where the path leaves the surface.
        ok = (dist < step[a]) & (np.einsum("ij,ij->i", n[j], cn[a]) > 0.9)
        alive[a[~ok]] = False
        a, j = a[ok], j[ok]
        cur[a], cn[a] = P[j], n[j]
        dirs[a] = _tangent(dirs[a] + rng.normal(scale=0.03, size=(len(a), 3)), cn[a])
        pts.append(j)
        sids.append(a)
    j, sid = np.concatenate(pts), np.concatenate(sids)
    # A scratch narrower than a texel still marks the nearest texel; texture res limits what survives.
    idx, e = _ball_entries(tree, P[j], np.maximum(width[sid], 0.6 * texel[j]))
    if not len(idx):
        return
    s = sid[e]
    np.maximum.at(F["scratch"], idx, depth[s])
    if ctx["layers"]:
        cut = np.zeros(len(P), np.float32)
        np.maximum.at(cut, idx, depth[s] * SCRATCH_DEPTH_UM)
        u = np.flatnonzero(cut)
        _consume(F["T"], u, cut[u], ctx["layers"], use_resist=False)


SET_KEYS = ("min_feature_px",)


def scale_params(prm, weather):
    """A stack's weather multipliers applied to the matching Params fields (SET_KEYS replace instead); other keys
    are stack-only knobs."""
    scaled = {k: getattr(prm, k) * v for k, v in weather.items() if hasattr(prm, k) and k not in SET_KEYS}
    return replace(prm, **scaled, **{k: v for k, v in weather.items() if k in SET_KEYS})


def simulate(part, vols, prm, log=print):
    P, m = part.pos, part.maps
    n = m["normal_obj"].astype(np.float64)
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    N = len(P)
    stack = STACKS[part.material]
    base, layers = BASES[stack["base"]], stack["layers"]
    weather = stack.get("weather", {})
    prm = scale_params(prm, weather)
    thick = np.array([l["thickness_um"] for l in layers], np.float32)
    seed = prm.seed * 7919 + part.seed_salt
    rng = np.random.default_rng([prm.seed, part.seed_salt])

    contact = contact_field(P, vols)
    expo = np.clip(m["ao_assembled"], 0, 1) ** 1.5
    # Narrow shells (a wedge head) read as convex_large across whole panels; light paints scale it down.
    cl = m["convex_large"] * weather.get("large_edges", 1.0)
    edge = m["convex_small"] + 1.2 * cl ** 1.5
    adhesion = 0.5 + smoothstep(0.35, 0.65, fbm(P, 180, 4, seed + 1))
    # min_feature_px: coarse atlases (big mogs) floor chip radius and micro-noise cell size in texels, so wear
    # isn't single-texel speckle. Chip count drops to keep the chipped area unchanged.
    feat_m = prm.min_feature_px * float(np.median(part.texel))
    chip_r_min = min(max(CHIP_R_MIN_M, feat_m / 2), CHIP_R_MAX_M / 2)
    chip_rate = _chip_r2(CHIP_R_MIN_M) / _chip_r2(chip_r_min)
    micro = 0.4 + 1.2 * smoothstep(0.3, 0.7, fbm(P, min(2500, 1 / max(feat_m, 1e-9)), 3, seed + 2))
    rust_noise = smoothstep(0.35, 0.7, fbm(P, 800, 4, seed + 3))
    rust_site = (0.25 + m["concave_large"] + 0.6 * (1 - m["ao_assembled"])) * (1 - 0.85 * contact)

    # Lips of creases and panel gaps: convex edges next to a cavity. AO darkens them, so the exposure terms alone
    # would shield exactly the edges that flex, collect grit and get pried at.
    lip = (smoothstep(0.2, 0.6, m["convex_small"] + 0.5 * m["convex_large"])
           * smoothstep(0.15, 0.5, m["concave_large"] + 0.5 * (1 - m["ao_assembled"]))).astype(np.float32)
    impact_w = (expo ** 2 * (0.15 * weather.get("flat_impacts", 1.0) + cl + 0.5 * m["convex_small"]) * (0.4 + contact)
                + prm.crease * 0.6 * lip).astype(np.float64)
    scratch_w = (expo * (0.3 + contact)).astype(np.float64)
    ctx = dict(P=P, n=n, tree=cKDTree(P), rng=rng, layers=layers, base=base, seed=seed, texel=part.texel,
               chip_r_min=chip_r_min,
               total_um=float(thick.sum()) if layers else 1.0,
               impact_cdf=np.cumsum(impact_w), scratch_cdf=np.cumsum(scratch_w))
    area = float((part.texel.astype(np.float64) ** 2).sum())

    zeros = lambda: np.zeros(N, np.float32)
    band = np.zeros(N, np.float32)
    if prm.edge_band > 0:
        band = edge_band_field(P, n, (m["fold_sharp"] > 0.5) | (m["convex_small"] > 0.75), prm.edge_band_mm / 1000)
        band = band * smoothstep(0.35, 0.65, fbm(P, 250, 4, seed + 56)) * np.clip(m["ao_assembled"], 0, 1)
    fine = m["fold_fine"] * smoothstep(0.3, 0.6, fbm(P, 600, 3, seed + 57)) * np.clip(m["ao_assembled"], 0, 1)
    abrasion = prm.use * (contact * expo * (0.04 + edge) + prm.crease * 0.12 * lip + prm.edge_band * 0.25 * band
                          + prm.fine_edge * 0.45 * fine) * micro
    F = dict(contact=contact, lip=lip, edge_band=band, T=np.tile(thick, (N, 1)),
             wear=zeros(), rust=zeros(), impact=zeros(), dent_mm=zeros(), scratch=zeros())
    dt = prm.age / prm.steps
    n_imp = n_scr = 0
    everything = np.arange(N)
    for _ in range(prm.steps):
        F["wear"] += dt * abrasion
        if layers:
            _consume(F["T"], everything, dt * abrasion * K_ABRASION * REF_FILM_UM / adhesion, layers)
        c = rng.poisson(dt * prm.use * IMPACTS_PER_M2 * chip_rate * area)
        if c:
            _impacts(F, ctx, c)
        n_imp += c
        c = rng.poisson(dt * prm.use * prm.scratches * SCRATCHES_PER_M2 * area)
        if c:
            _scratches(F, ctx, c)
        n_scr += c
        np.clip(F["T"], 0, None, out=F["T"])
        if base["rusts"]:
            # Rust only grows on metal that is already bare, so early chips end up rustier than late ones.
            bare = 1 - smoothstep(0, 0.2, F["T"][:, 0] / thick[0]) if layers else 1.0
            if base.get("corrodes_under_layers"):
                bare = 1.0
            F["rust"] += dt * prm.humidity * K_RUST * base.get("corrosion_rate", 1.0) * bare * rust_noise * rust_site

    gseed = prm.seed * 7919
    if layers and prm.scratch_cut > 0 and prm.detail > 0:
        # Same mask, seed and weighting as the composite's surface scratches, so marks and cuts line up.
        sc = texsample.detail("scratches", P, m["normal_obj"], prm.detail_scale, seed + 40)
        sc = sc * np.clip(1.5 * contact + 0.15, 0, 1) * expo * prm.detail
        cut = (prm.scratch_cut * 2.5 * SCRATCH_DEPTH_UM * smoothstep(0.15, 0.7, sc)).astype(np.float32)
        u = np.flatnonzero(cut > 0)
        _consume(F["T"], u, cut[u], layers, use_resist=False)
    if layers and prm.crease > 0:
        # Paint lifts and flakes inside creases where grit and moisture sit.
        crease = np.maximum(smoothstep(0.35, 0.75, m["concave_large"]), m["fold_concave"])
        crease = crease * smoothstep(0.45, 0.7, fbm(P, 500, 3, seed + 55))
        F["T"][:, -1] *= 1 - prm.crease * 0.9 * smoothstep(0.2, 0.6, crease)
        if len(layers) > 1:
            F["T"][:, -2] *= 1 - prm.crease * 0.6 * smoothstep(0.45, 0.8, crease)
    F["scorch"] = scorch_field(P, n, prm, gseed) if prm.scorch > 0 else zeros()
    if "burn" in m:
        F["scorch"] = np.maximum(F["scorch"], m["burn"] * (0.75 + 0.5 * fbm(P, 150, 3, gseed + 58))).clip(0, 1)
    F["streaks"] = streak_field(P, n, m, prm, gseed) if prm.streaks > 0 else zeros()
    if layers and prm.scorch > 0:
        # Paint blisters off the hottest spots first; the primer only goes where it burnt hardest.
        blister = 0.25 * (fbm(P, 900, 3, gseed + 54) - 0.5)
        F["T"][:, -1] *= 1 - smoothstep(0.8, 1.0, F["scorch"] + blister)
        if len(layers) > 1:
            F["T"][:, -2] *= 1 - smoothstep(0.95, 1.05, F["scorch"] + blister)

    curv = None
    if prm.ridge > 0 or prm.cavity > 0:
        if "curv3d" not in m:
            m = dict(m)
            m["curv3d"], m["occl3d"] = curvature3d(P, n, prm.ridge_mm / 1000)
            m["fold_cvx3d"], m["fold_ccv3d"] = folds3d(P, n, FOLD3D_FRAC * prm.ridge_mm / 1000)
        curv = m["curv3d"]
    F["ridge"] = zeros()
    if layers and prm.ridge > 0:
        # Hidden faces keep their paint; ao_assembled is not used as a factor, it damps every edge next to another plate.
        seen = smoothstep(0.1, 0.4, m["ao_assembled"])
        jag = 0.6 + 0.8 * fbm(P, 1600, 2, seed + 70)
        runs = smoothstep(0.38, 0.6, fbm(P, 150, 3, seed + 71))
        edge = np.maximum(smoothstep(0.2, 0.5, curv), smoothstep(FOLD_SHARP_DEG - 13, FOLD_SHARP_DEG + 3, m["fold_cvx3d"]))
        s = edge * jag * (0.3 + 0.7 * runs) * seen
        F["ridge"] = s.astype(np.float32)
        F["T"][:, -1] *= 1 - prm.ridge * smoothstep(0.4, 0.6, s)
        if len(layers) > 1:
            F["T"][:, :-1] *= (1 - prm.ridge * smoothstep(0.6, 0.8, s))[:, None]

    F["fray"] = zeros()
    if stack["base"] == "fabric" and prm.fray > 0 and curv is not None:
        edge = np.maximum(smoothstep(0.15, 0.45, curv), smoothstep(FOLD_SHARP_DEG - 18, FOLD_SHARP_DEG, m["fold_cvx3d"]))
        tufts = smoothstep(0.35, 0.65, fbm(P, 900, 3, seed + 74))
        F["fray"] = np.clip(prm.fray * edge * (0.4 + 0.6 * tufts) * smoothstep(0.1, 0.4, m["ao_assembled"]), 0, 1).astype(np.float32)

    F["relief_mm"] = zeros()
    hw_grime = None
    if layers and curv is not None and (prm.rivets > 0 or prm.seams > 0):
        F["relief_mm"], hw_grime, crown = hardware(P, n, m, curv, prm, part.texel, rng, seed)
        F["T"][:, -1] *= 1 - 0.9 * smoothstep(0.3, 0.6, crown)
        if len(layers) > 1:
            F["T"][:, :-1] *= (1 - 0.9 * smoothstep(0.75, 0.95, crown))[:, None]

    F["layers"] = F.pop("T") / np.maximum(thick, 1e-6) if layers else np.zeros((N, 0), np.float32)
    flaky = [k for k, l in enumerate(layers) if l["profile"] == "voronoi"]
    if flaky:
        seed_texel = np.concatenate([
            ctx["tree"].query(cell_seeds(P[s:s + 1_000_000], np.full(min(1_000_000, N - s), FLAKE_CELL_M),
                                          seed + 30), workers=-1)[1]
            for s in range(0, N, 1_000_000)])
        for k in flaky:
            F["layers"][:, k] = np.minimum(F["layers"][:, k], F["layers"][seed_texel, k])
    F["coat"] = F["layers"].mean(1) if layers else zeros()
    for k, lay in enumerate(layers):
        F[f"layer_{k}_{lay['name']}"] = F["layers"][:, k]
    F["rust"] = np.clip(F["rust"], 0, 1)
    F["wear"] = np.clip(F["wear"] * K_ABRASION, 0, 1)
    grime_site = 0.9 * m["concave_large"] + 0.5 * m["concave_small"] + 0.6 * (1 - m["ao_assembled"]) + prm.crease * 0.6 * m["fold_concave"]
    F["grime"] = np.clip(prm.dirt * prm.age * 2.0 * grime_site * (1 - 0.8 * contact)
                         * smoothstep(0.3, 0.7, fbm(P, 300, 4, seed + 4)), 0, 1)
    if "haze" in m:
        F["grime"] = np.maximum(F["grime"], m["haze"] * smoothstep(0.25, 0.75, fbm(P, 120, 4, seed + 73)))
    if curv is not None and prm.cavity > 0:
        # Contact lines where another piece sits on or overhangs this one.
        under = smoothstep(0.15, 0.4, m["concave_assembled"] - m["concave_large"]) if "concave_assembled" in m else zeros()
        cav = np.maximum.reduce([smoothstep(0.06, 0.25, -curv), smoothstep(0.05, 0.2, m["occl3d"]), smoothstep(FOLD_SHARP_DEG - 13, FOLD_SHARP_DEG + 5, m["fold_ccv3d"]),
                                 0.8 * under]) * (0.75 + 0.25 * fbm(P, 400, 3, seed + 72))
        F["grime"] = np.maximum(F["grime"], np.clip(prm.cavity * cav, 0, 1))
    if hw_grime is not None:
        F["grime"] = np.maximum(F["grime"], hw_grime)
    up = np.clip(n[:, 2], 0, 1) ** 2
    F["dust"] = np.clip(prm.dust * prm.age * 2.5 * up * m["ao_assembled"] * (1 - contact)
                        * smoothstep(0.3, 0.7, fbm(P, 600, 4, seed + 5)), 0, 1).astype(np.float32)
    if layers:
        exposed = m["ao_assembled"] > 0.6
        bottom, top = F["layers"][:, 0], F["layers"][:, -1]
        for label, sel in (("edges", exposed & (m["convex_small"] > 0.5)),
                           ("edge band", exposed & (m["convex_large"] > 0.4))):
            if not sel.any():
                continue
            log(f"[sim] {part.name}: exposed {label}: {(bottom[sel] < 0.12).mean():.1%} bare, "
                f"{(top[sel] < 0.6).mean():.1%} top layer thinned ({sel.sum():,} texels)")
    log(f"[sim] {part.name}: {part.material}, {N:,} texels, {area * 1e6:,.0f} mm2, "
        f"texel median {np.median(part.texel) * 1000:.3f} mm, {n_imp} impacts, {n_scr} scratches")
    return F
