"""Composed factory livery (Gunpla-style decal sheet) for a pristine paint: hazard stripes along the long straight
bottom rims of plates, big caution panels centred on the largest flat plates, and small supporting marks aligned
to them and clustered at the joints. Laid out on the -x half and mirrored, except one unit number on the -x side.

Writes part.maps decal_alpha / decal_color like decals.place, so composite lays it over the topcoat.
cfg: {targets: {material: weight}, light_paint: [materials], m_per_stud, pinch: [{at, mirror_x}] (Roblox studs),
      panels, stripes, scale, panel_min_d, anchors}.
panel_min_d: smallest rim distance (studs) a panel centre may have, default 0.16; lower it for small faceted plates.
anchors: [{part, at: [x, y, z] part-space metres on the -x half, h: panel height m, tol, plan, cover}] force a panel
onto a plate the flat-plate search misses; it is mirrored like the rest.
"""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from .decals import BLACK, WHITE, YELLOW, RESCUE_RED, _glyphs, _random_glyphs, _rect, _shape
from .wear import _straight, folds3d, rim_distance

GREY = (0.32, 0.33, 0.34)


def _tri(u, v, s=1.0, cx=0.5, cy=0.37):
    x, y = (u - cx) / s + cx, (v - cy) / s + cy
    return (y >= 0.06) & (y <= 0.94) & (np.abs(x - 0.5) <= 0.47 * (0.94 - y) / 0.88)


def _panel(u, v, p):
    """Caution panel: framed ground, header bar with text, a warning triangle and text rows."""
    a = np.ones(len(u), np.float32)
    ink, ground, accent = (np.asarray(x, np.float32) for x in (p["ink"], p["ground"], p["accent"]))
    c = np.tile(ground, (len(u), 1))
    asp = p["aspect"]
    m = 0.05
    frame = ~_rect(u, v, (m / asp, m, 1 - m / asp, 1 - m))
    inner_line = ~_rect(u, v, (1.6 * m / asp, 1.6 * m, 1 - 1.6 * m / asp, 1 - 1.6 * m)) & ~frame
    header = _rect(u, v, (m / asp, 0.74, 1 - m / asp, 1 - m))
    hu, hv = (u - 0.08) / 0.62, (v - 0.78) / 0.14
    htext = _glyphs(np.clip(hu, 0, 1), hv, p["head"], stencil=False) & (hu >= 0) & (hu <= 1)
    # Triangle sits in a square on the left of the body; rows fill the rest.
    tw = 0.58 / asp
    tu, tv = (u - 0.06 / asp * 1.8) / tw, (v - 0.1) / 0.58
    t_out, t_in = _tri(tu, tv), _tri(tu, tv, 0.72)
    mark = _rect(tu, tv, (0.455, 0.36, 0.545, 0.7)) | _rect(tu, tv, (0.455, 0.2, 0.545, 0.29))
    rows = np.zeros(len(u), bool)
    x0 = 0.06 / asp * 1.8 + tw + 0.05
    for k, (y0, frac) in enumerate(p["rows"]):
        rows |= _rect(u, v, (x0, y0, x0 + (0.92 - x0) * frac, y0 + 0.07))
    c[inner_line | rows] = ink
    c[frame] = ink
    c[header] = ink
    c[header & htext] = accent
    c[t_out & ~t_in] = ink
    c[t_in & ~mark] = accent
    c[t_in & mark] = ink
    return a, c


def _stripe_band(u, v, p):
    a = np.ones(len(u), np.float32)
    s = ((u * p["aspect"] + v) / 0.9) % 1 < 0.5
    return a, np.where(s[:, None], np.asarray(YELLOW, np.float32), np.asarray(BLACK, np.float32))


def _text(u, v, p):
    a = _glyphs(u, v, p["glyphs"], stencil=p.get("stencil", True)).astype(np.float32)
    return a, np.tile(np.asarray(p["color"], np.float32), (len(u), 1))


def _bar_text(u, v, p):
    """Thin underline with text above it: a caution line."""
    a = np.zeros(len(u), np.float32)
    line = _rect(u, v, (0.0, 0.0, 1.0, 0.14))
    txt = _glyphs(u / 0.8, (v - 0.28) / 0.72, p["glyphs"], stencil=False) & (u <= 0.8)
    a[line | txt] = 1
    c = np.tile(np.asarray(p["color"], np.float32), (len(u), 1))
    c[line] = p.get("line", p["color"])
    return a, c


def place(part, cfg, seed, log=print):
    m, P = part.maps, part.pos.astype(np.float64)
    n = m["normal_obj"].astype(np.float64)
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    N = len(P)
    unit = cfg["m_per_stud"] * cfg.get("scale", 1.0)
    texel = float(np.median(part.texel))
    mats = part.meta["materials"]
    mid = np.round(m["matid"] * 32 - 0.5).astype(int)
    target = np.zeros(N, bool)
    light = np.zeros(N, bool)
    for mat in cfg["targets"]:
        if mat in mats:
            target |= mid == mats.index(mat)
            if mat in cfg.get("light_paint", []):
                light |= mid == mats.index(mat)
    rng = np.random.default_rng([seed, 777])
    alpha = np.zeros(N, np.float32)
    color = np.zeros((N, 3), np.float32)
    half = P[:, 0] <= 0
    tree = cKDTree(P)
    taken = np.zeros(N, bool)
    counts = {}

    seeds = folds3d(P, n, 0.004)[0] > 60
    d, near = rim_distance(P, n, seeds, 0.7 * unit)
    curv = m.get("curv3d", np.zeros(N))
    flat = target & (np.abs(curv) < 0.15) & (m["ao_assembled"] > 0.8)

    def stamp(c, n0, right, up, w, h, fn, p, kind, need_target=True, tol=0.9, plan=0.12, cover=0.9):
        r = 0.5 * np.hypot(w, h)
        idx = np.asarray(tree.query_ball_point(c, r), np.int64)
        if not len(idx):
            return False
        dd = P[idx] - c
        du, dv, dn = dd @ right, dd @ up, dd @ n0
        inside = (np.abs(du) < w / 2) & (np.abs(dv) < h / 2)
        ok = inside & (np.abs(dn) < plan * h) & (n[idx] @ n0 > tol)
        if need_target:
            ok &= target[idx]
        if inside.sum() == 0 or ok.sum() < cover * inside.sum() or taken[idx[ok]].mean() > 0.02:
            return False
        idx, u, v = idx[ok], du[ok] / w + 0.5, dv[ok] / h + 0.5
        tx = part.texel[idx] / h
        acc_a = np.zeros(len(idx), np.float32)
        acc_c = np.zeros((len(idx), 3), np.float32)
        for ou, ov in ((-0.25, -0.25), (0.25, -0.25), (-0.25, 0.25), (0.25, 0.25)):
            a, col = fn(u + ou * tx * h / w, v + ov * tx, p)
            acc_a += a / 4
            acc_c += col * a[:, None] / 4
        col = acc_c / np.maximum(acc_a, 1e-6)[:, None]
        color[idx] = color[idx] + (col - color[idx]) * acc_a[:, None]
        alpha[idx] = acc_a + alpha[idx] * (1 - acc_a)
        taken[idx] = True
        counts[kind] = counts.get(kind, 0) + 1
        return True

    def frame_up(n0):
        up = np.array([0.0, 0.0, 1.0]) - n0 * n0[2]
        up /= max(np.linalg.norm(up), 1e-9)
        return np.cross(up, n0), up

    def ink_for(i):
        return BLACK if light[i] else WHITE

    # 1. Big caution panels at the centres of the largest flat vertical plates.
    panels = []
    cand = np.flatnonzero(flat & half & (np.abs(n[:, 2]) < 0.55) & np.isfinite(d) & (d > cfg.get("panel_min_d", 0.16) * unit))
    # Faces a viewer sees first: forward (+y) and outward (-x on this half).
    facing = 0.4 + 0.6 * np.clip(np.maximum(n[cand, 1], -n[cand, 0]), 0, 1)
    cand = cand[np.argsort(-d[cand] * facing)]
    blocked = np.zeros(N, bool)
    for i in cand[:: max(1, len(cand) // 4000)]:
        if len(panels) >= cfg.get("panels", 3) or blocked[i]:
            continue
        right, up = frame_up(n[i])
        h = float(np.clip(1.7 * d[i], 0.24 * unit, 0.6 * unit))
        aspect = float(rng.uniform(1.5, 1.9))
        w = h * aspect
        dark = light[i]
        p = dict(aspect=aspect, head=_random_glyphs(rng, int(rng.integers(5, 8))),
                 rows=[(0.5, rng.uniform(0.7, 1)), (0.34, rng.uniform(0.5, 0.95)), (0.18, rng.uniform(0.4, 0.8))],
                 ground=YELLOW if not dark else WHITE, ink=BLACK, accent=YELLOW if dark else YELLOW)
        if len(panels) % 2 == 1:
            p.update(ground=WHITE, accent=RESCUE_RED)
        for shrink in (1.0, 0.85, 0.72, 0.6, 0.5, 0.42):
            if stamp(P[i], n[i], right, up, w * shrink, h * shrink, _panel, p, "panel", tol=0.97, plan=0.04, cover=0.98):
                panels.append((i, w * shrink, h * shrink, right, up))
                blocked[tree.query_ball_point(P[i], 0.6 * w * shrink + 0.1 * unit)] = True
                break
    # Caution line under each panel, number pair at its upper right.
    for i, w, h, right, up in panels:
        ink = ink_for(i)
        c = P[i] - up * (0.5 * h + 0.09 * unit)
        stamp(c, n[i], right, up, 0.75 * w, 0.06 * unit, _bar_text, dict(glyphs=_random_glyphs(rng, 9), color=ink, line=RESCUE_RED),
              "caution_line")
        c = P[i] + right * (0.5 * w + 0.08 * unit) + up * (0.35 * h)
        stamp(c, n[i], right, up, 0.12 * unit, 0.08 * unit, _text, dict(glyphs=_random_glyphs(rng, 2), color=ink), "number")

    # Anchored panels for plates the flatness search misses (small faceted plates). at: part-space metres on the -x
    # half; the frontmost (+y) target texel within 1 cm of at's x/z is the centre.
    for a in cfg.get("anchors", []):
        if a["part"] != part.name:
            continue
        x, z = a["at"][0], a["at"][2]
        near_xz = np.flatnonzero(target & (np.hypot(P[:, 0] - x, P[:, 2] - z) < 0.01))
        if not len(near_xz):
            log(f"[livery] {part.name}: anchor {a['at']} has no target texel within 1 cm")
            continue
        i = near_xz[np.argmax(P[near_xz, 1])]
        n0 = n[tree.query_ball_point(P[i], 0.4 * a["h"])].mean(0)
        n0 /= np.linalg.norm(n0)
        right, up = frame_up(n0)
        p = dict(aspect=a.get("aspect", 1.6), head=_random_glyphs(rng, 6),
                 rows=[(0.5, rng.uniform(0.7, 1)), (0.34, rng.uniform(0.5, 0.95)), (0.18, rng.uniform(0.4, 0.8))],
                 ground=WHITE if light[i] else YELLOW, ink=BLACK, accent=YELLOW if light[i] else RESCUE_RED)
        if not stamp(P[i], n0, right, up, a["h"] * p["aspect"], a["h"], _panel, p, "anchor_panel",
                     tol=a.get("tol", 0.85), plan=a.get("plan", 0.15), cover=a.get("cover", 0.9)):
            log(f"[livery] {part.name}: anchor at {np.round(P[i], 3)} n {np.round(n0, 2)} did not fit")

    # 2. Hazard stripes along long straight rims below a flat band (the bottom edge of a plate).
    bw = 0.1 * unit
    d0 = 0.006 * unit + 0.004
    gdir = np.zeros((N, 3))
    okn = near >= 0
    gdir[okn] = P[okn] - P[near[okn]]
    gdir[okn] -= n[okn] * np.einsum("ij,ij->i", gdir[okn], n[okn])[:, None]
    gdir /= np.maximum(np.linalg.norm(gdir, axis=1, keepdims=True), 1e-12)
    side_ok = target & okn & (d > d0) & (d < d0 + bw) & (gdir[:, 2] > 0.75) & (np.abs(n[:, 2]) < 0.5) & (m["ao_assembled"] > 0.7)

    def comps_for(side):
        band = np.flatnonzero(side_ok & side & ~taken)
        if len(band):
            band = band[_straight(P, near[band], 0.25 * unit) > 0.7]
        out = []
        if len(band) <= 50:
            return out
        pairs = cKDTree(P[band]).query_pairs(2.5 * texel, output_type="ndarray")
        g = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(band), len(band)))
        k, lab = connected_components(g, directed=False)
        for c in range(k):
            sel = band[lab == c]
            if len(sel) < 30:
                continue
            Q = P[sel] - P[sel].mean(0)
            t = np.linalg.eigh(Q.T @ Q)[1][:, 2]
            extent = np.ptp(Q @ t)
            # One plane only: a band wrapped round a faceted cuff breaks up, since the stripe phase runs along one axis.
            nm = n[sel].mean(0)
            nm /= max(np.linalg.norm(nm), 1e-9)
            if extent > 0.45 * unit and np.percentile(n[sel] @ nm, 5) > 0.93:
                out.append((extent, sel, t, P[sel].mean(0)))
        return sorted(out, key=lambda x: -x[0])

    pitch = 0.12 * unit
    stripe_px = np.zeros(N, bool)

    def paint_stripe(sel, t):
        s = (P[sel] @ t) + (d[sel] - d0)
        yel = ((s / pitch) % 1) < 0.5
        # Keep a margin at both ends so the band stops square, not ragged.
        along = P[sel] @ t
        keep = (along > along.min() + 0.02 * unit) & (along < along.max() - 0.02 * unit)
        sel, yel = sel[keep], yel[keep]
        alpha[sel] = 1
        color[sel] = np.where(yel[:, None], np.asarray(YELLOW, np.float32), np.asarray(BLACK, np.float32))
        taken[sel] = True
        stripe_px[sel] = True
        return sel

    # Each side's stripe comes from its own plate edge; a left stripe without a matching right edge (the mirror
    # would have left patches) is dropped on both sides.
    right_comps = comps_for(~half)
    pending = []
    stripes = 0
    for extent, sel, t, cen in comps_for(half):
        if stripes >= cfg.get("stripes", 4):
            break
        mcen = cen * np.array([-1.0, 1, 1])
        match = [rc for rc in right_comps if np.linalg.norm(rc[3] - mcen) < 0.12 * unit and rc[0] > 0.7 * extent]
        if not match:
            continue
        rc = min(match, key=lambda x: np.linalg.norm(x[3] - mcen))
        right_comps = [x for x in right_comps if x is not rc]
        sel = paint_stripe(sel, t)
        pending.append((rc[1], t * np.array([-1.0, 1, 1])))
        stripes += 1
        # A small warning triangle above one end of the stripe.
        e = sel[np.argmin(P[sel] @ t)]
        right, up = frame_up(n[e])
        c = P[e] + up * (bw + 0.1 * unit) + t * np.sign(t @ right) * 0.08 * unit
        stamp(c, n[e], right, up, 0.13 * unit, 0.13 * unit, lambda u, v, p: _shape("warning", u, v, p),
              dict(color=BLACK, aspect=1.0), "stripe_tag")
    counts["stripe"] = stripes

    # 3. Joint markers: a small warning with a stencil line beside it, aligned upright.
    for pt in cfg.get("pinch", []):
        at = np.asarray(pt["at"], np.float64)
        if at[0] > 0:
            at[0] = -at[0]
        c0 = np.array([at[0], -at[2], at[1]]) * cfg["m_per_stud"]
        near_j = np.flatnonzero(flat & half & (np.linalg.norm(P - c0, axis=1) < 0.7 * unit) & ~taken)
        near_j = near_j[np.argsort(np.linalg.norm(P[near_j] - c0, axis=1))][:: max(1, len(near_j) // 80)]
        for i in near_j:
            right, up = frame_up(n[i])
            ink = ink_for(i)
            if stamp(P[i], n[i], right, up, 0.15 * unit, 0.15 * unit, lambda u, v, p: _shape("warning", u, v, p),
                     dict(color=BLACK, aspect=1.0), "pinch"):
                stamp(P[i] + right * 0.2 * unit, n[i], right, up, 0.22 * unit, 0.05 * unit, _text,
                      dict(glyphs=_random_glyphs(rng, 5), color=ink, stencil=False), "pinch_text")
                break

    # Mirror the -x half onto +x.
    src = np.flatnonzero(half & (alpha > 0))
    dst = np.flatnonzero(~half)
    if len(src) and len(dst):
        hp = np.flatnonzero(half)
        mt = cKDTree(P[hp])
        Pm = P[dst] * np.array([-1.0, 1, 1])
        dist, j = mt.query(Pm, distance_upper_bound=2.5 * texel, workers=-1)
        ok = np.isfinite(dist)
        jj = hp[np.where(ok, j, 0)]
        nm = n[dst] * np.array([-1.0, 1, 1])
        ok &= (np.einsum("ij,ij->i", n[jj], nm) > 0.8) & ~stripe_px[jj]
        alpha[dst[ok]] = alpha[jj[ok]]
        color[dst[ok]] = color[jj[ok]]
        taken[dst[ok]] = taken[jj[ok]]
    for sel, t in pending:
        paint_stripe(sel, t)

    # 4. One unit number, -x side only: the deliberate asymmetry.
    if cfg.get("unit_number", True):
        cand = np.flatnonzero(flat & half & (np.abs(n[:, 2]) < 0.5) & np.isfinite(d) & (d > 0.12 * unit) & ~taken)
        cand = cand[np.argsort(-d[cand])][:: max(1, len(cand) // 500)]
        glyphs = _random_glyphs(rng, 2)
        for i in cand[:200]:
            right, up = frame_up(n[i])
            h = float(np.clip(1.2 * d[i], 0.16 * unit, 0.3 * unit))
            if stamp(P[i], n[i], right, up, 1.6 * h * 0.8, h, _text, dict(glyphs=glyphs, color=ink_for(i), stencil=True),
                     "unit_number"):
                break

    m["decal_alpha"] = alpha
    m["decal_color"] = color
    log(f"[livery] {part.name}: {counts}")
