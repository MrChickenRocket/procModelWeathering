"""Procedural markings (hazard stripes, warning signs, fake stencil lettering, data plates) projected onto
flat panels of an atlas part. Lettering is random strokes on purpose: it reads as text without being legible.

Placement runs on the whole part before it is split per material, so a marking can only land on one panel
but the result is continuous in 3D. Output goes into part.maps as decal_alpha (N,) and decal_color (N, 3);
composite.shade lays it over the topcoat.
"""
import numpy as np
from scipy.spatial import cKDTree

YELLOW = (0.80, 0.50, 0.012)
BLACK = (0.016, 0.016, 0.016)
WHITE = (0.70, 0.70, 0.66)

# Sizes are in studs (heights; width = height * aspect). Lettering width comes from its glyph count.
KINDS = {
    "hazard": dict(h=(0.12, 0.24), aspect=(2.5, 5.0)),
    "warning": dict(h=(0.18, 0.34), aspect=(1.12, 1.12)),
    "stencil": dict(h=(0.07, 0.14), glyphs=(3, 7)),
    "plate": dict(h=(0.18, 0.3), aspect=(1.4, 2.0)),
    "triangle": dict(h=(0.12, 0.24), aspect=(1.0, 1.0)),
    "roundel": dict(h=(0.12, 0.24), aspect=(1.0, 1.0)),
    "numbers": dict(h=(0.22, 0.38), glyphs=(2, 2)),
    "stripe": dict(h=(0.05, 0.1), aspect=(5.0, 10.0)),
    # Pinch-point only (never in MIX): name them in a pinch point's kinds.
    "biohazard": dict(h=(0.28, 0.4), aspect=(1.0, 1.0)),
    "corrosive": dict(h=(0.3, 0.42), aspect=(1.0, 1.0)),
    "laser": dict(h=(0.24, 0.34), aspect=(1.12, 1.12)),
    "gear": dict(h=(0.24, 0.34), aspect=(1.12, 1.12)),
    "chevron": dict(h=(0.14, 0.22), aspect=(1.6, 2.4)),
    "voltage": dict(h=(0.24, 0.34), aspect=(1.12, 1.12)),
    "arcflash": dict(h=(0.26, 0.36), aspect=(1.4, 1.6)),
    "cold": dict(h=(0.24, 0.34), aspect=(1.12, 1.12)),
    "radiation": dict(h=(0.24, 0.34), aspect=(1.0, 1.0)),
    "diamond": dict(h=(0.26, 0.36), aspect=(1.0, 1.0)),
    "rf": dict(h=(0.24, 0.34), aspect=(1.12, 1.12)),
    "refband": dict(h=(0.1, 0.14), aspect=(4.0, 5.0)),
    "coldstripe": dict(h=(0.12, 0.18), aspect=(4.0, 6.0)),
    "flame": dict(h=(0.24, 0.34), aspect=(1.12, 1.12)),
    "hotsurface": dict(h=(0.24, 0.34), aspect=(1.12, 1.12)),
    "redhatch": dict(h=(0.12, 0.18), aspect=(4.0, 6.0)),
    "phase": dict(h=(0.1, 0.14), aspect=(3.0, 3.6)),
    "earth": dict(h=(0.2, 0.28), aspect=(1.0, 1.0)),
    "papertape": dict(h=(0.16, 0.22), aspect=(2.6, 3.2)),
    # Aircraft stencils; only placed when a mog's decals.mix names them.
    "nostep": dict(h=(0.1, 0.16), aspect=(2.4, 3.2)),
    "arrow": dict(h=(0.12, 0.2), aspect=(2.6, 3.6)),
}
RESCUE_RED = (0.5, 0.03, 0.02)
ANSI_ORANGE = (0.85, 0.2, 0.01)
NAVY = (0.012, 0.03, 0.12)
EARTH_GREEN = (0.02, 0.25, 0.06)
TAPE_CREAM = (0.78, 0.72, 0.52)
# Old-style three-phase marking, L1 L2 L3.
PHASES = ((0.5, 0.03, 0.02), (0.8, 0.5, 0.02), (0.02, 0.08, 0.4))
MIX = {
    "paint": {"stencil": 0.3, "numbers": 0.12, "plate": 0.12, "warning": 0.13, "hazard": 0.13, "triangle": 0.12,
              "roundel": 0.08},
    "frame": {"hazard": 0.45, "warning": 0.2, "stencil": 0.25, "plate": 0.1},
}
# Glyph strokes as rects (x0, y0, x1, y1) in a unit glyph box, stroke width 0.22.
_W = 0.22
STROKES = [(0, 0, _W, 1), (1 - _W, 0, 1, 1), (0.5 - _W / 2, 0, 0.5 + _W / 2, 1), (0, 1 - _W, 1, 1),
           (0, 0.5 - _W / 2, 1, 0.5 + _W / 2), (0, 0, 1, _W), (0, 0.5, _W, 1), (0, 0, _W, 0.5),
           (1 - _W, 0.5, 1, 1), (1 - _W, 0, 1, 0.5), (0.5 - _W / 2, 0.5, 0.5 + _W / 2, 1), (0, 0.5 - _W / 2, 0.5, 0.5 + _W / 2)]


def _rect(x, y, r):
    return (x >= r[0]) & (x <= r[2]) & (y >= r[1]) & (y <= r[3])


def _glyphs(u, v, strokes_per_glyph, stencil=True):
    """Row of glyphs filling the unit box; each glyph is a list of stroke indices."""
    n = len(strokes_per_glyph)
    cell = np.clip((u * n).astype(int), 0, n - 1)
    gx = (u * n - cell - 0.1) / 0.8
    gy = (v - 0.05) / 0.9
    out = np.zeros(len(u), bool)
    for g, strokes in enumerate(strokes_per_glyph):
        s = cell == g
        for k in strokes:
            out[s] |= _rect(gx[s], gy[s], STROKES[k])
    if stencil:
        # Stencil bridges: thin breaks so the letters look sprayed through a cut sheet.
        out &= ~((np.abs(gy - 0.7) < 0.04) | (np.abs(gx - 0.3) < 0.035))
    return out & (gx >= 0) & (gx <= 1) & (gy >= 0) & (gy <= 1)


def _random_glyphs(rng, n):
    return [list(rng.choice(len(STROKES), rng.integers(3, 6), replace=False)) for _ in range(n)]


# Vinyl kinds that can be cut with a white border; stencils are sprayed and plates already have a ground.
STICKERS = ("warning", "triangle", "roundel", "numbers", "hazard", "biohazard", "corrosive", "laser", "gear", "voltage",
            "arcflash", "cold", "refband", "radiation", "rf", "diamond", "flame", "hotsurface", "earth")
BACKING_MARGIN = 0.09


def _shape(kind, u, v, p):
    """(alpha (M,), color (M, 3)) for decal-local coordinates in [0, 1]."""
    if p.get("backing"):
        # Inner art shrunk into a white rounded-rect sticker; margin in decal heights.
        mv, mu = BACKING_MARGIN, BACKING_MARGIN / p["aspect"]
        iu, iv = (u - mu) / (1 - 2 * mu), (v - mv) / (1 - 2 * mv)
        a, c = _shape(kind, iu, iv, dict(p, backing=False))
        a = a * ((iu >= 0) & (iu <= 1) & (iv >= 0) & (iv <= 1))
        r = 0.6 * mv
        cu, cv = np.clip(u, r / p["aspect"], 1 - r / p["aspect"]), np.clip(v, r, 1 - r)
        inside = np.hypot((u - cu) * p["aspect"], v - cv) <= r
        c = c * a[:, None] + np.asarray(WHITE, np.float32) * (1 - a[:, None])
        return inside.astype(np.float32), c.astype(np.float32)
    a = np.zeros(len(u), np.float32)
    c = np.tile(np.asarray(p["color"], np.float32), (len(u), 1))
    if kind == "hazard":
        a[:] = 1
        stripe = ((u * p["aspect"] + v) / 0.8) % 1 < 0.5
        c[:] = np.where(stripe[:, None], YELLOW, BLACK)
    elif kind == "warning":
        def tri(s):
            cx, cy = 0.5, 0.37
            x, y = (u - cx) / s + cx, (v - cy) / s + cy
            return (y >= 0.06) & (y <= 0.94) & (np.abs(x - 0.5) <= 0.47 * (0.94 - y) / 0.88)
        outer, inner = tri(1.0), tri(0.74)
        mark = (_rect(u, v, (0.455, 0.36, 0.545, 0.7)) | _rect(u, v, (0.455, 0.2, 0.545, 0.29)))
        a[outer] = 1
        c[:] = BLACK
        c[inner & ~mark] = YELLOW
    elif kind == "laser":
        def tri(sc):
            cx, cy = 0.5, 0.37
            x, y = (u - cx) / sc + cx, (v - cy) / sc + cy
            return (y >= 0.06) & (y <= 0.94) & (np.abs(x - 0.5) <= 0.47 * (0.94 - y) / 0.88)
        outer, inner = tri(1.0), tri(0.8)
        # Starburst: a dot with rays, one long ray to the right (IEC 60825 laser symbol).
        x, y = u - 0.43, v - 0.36
        r, t = np.hypot(x, y), np.arctan2(y, x)
        rays = (np.abs(((t / (2 * np.pi / 16)) + 0.5) % 1 - 0.5) < 0.12) & (r > 0.07) & (r < 0.17)
        beam = _rect(u, v, (0.43, 0.345, 0.78, 0.375))
        mark = (r <= 0.05) | rays | beam
        a[outer] = 1
        c[:] = BLACK
        c[inner & ~mark] = YELLOW
    elif kind == "gear":
        def tri(sc):
            cx, cy = 0.5, 0.37
            x, y = (u - cx) / sc + cx, (v - cy) / sc + cy
            return (y >= 0.06) & (y <= 0.94) & (np.abs(x - 0.5) <= 0.47 * (0.94 - y) / 0.88)
        outer, inner = tri(1.0), tri(0.8)
        # Two meshing gears (ISO 7010 W024 crushing hazard); teeth offset half a pitch so they interlock.
        mark = np.zeros(len(u), bool)
        for cx, cy, r0, ph in ((0.375, 0.3, 0.095, 0.0), (0.625, 0.3, 0.095, 0.5)):
            x, y = u - cx, v - cy
            r, t = np.hypot(x, y), np.arctan2(y, x)
            tooth = ((t / (2 * np.pi / 8) + ph) % 1) < 0.5
            mark |= ((r <= r0) | (tooth & (r <= r0 + 0.035))) & (r > 0.035)
        a[outer] = 1
        c[:] = BLACK
        c[inner & ~mark] = YELLOW
    elif kind == "voltage":
        def tri(sc):
            cx, cy = 0.5, 0.37
            x, y = (u - cx) / sc + cx, (v - cy) / sc + cy
            return (y >= 0.06) & (y <= 0.94) & (np.abs(x - 0.5) <= 0.47 * (0.94 - y) / 0.88)
        outer, inner = tri(1.0), tri(0.8)
        # Lightning bolt: two slanted bars offset at a step, the lower one ending in a point.
        top = (v >= 0.4) & (v <= 0.68) & (np.abs(u - (0.5 + 0.35 * (v - 0.54))) < 0.045)
        step = _rect(u, v, (0.44, 0.37, 0.56, 0.41))
        low = (v >= 0.14) & (v <= 0.4) & (np.abs(u - (0.5 + 0.35 * (v - 0.27))) < 0.045 * (v - 0.14) / 0.26)
        a[outer] = 1
        c[:] = BLACK
        c[inner & ~(top | step | low)] = YELLOW
    elif kind == "arcflash":
        a[:] = 1
        c[:] = WHITE
        border = ~_rect(u, v, (0.03, 0.04, 0.97, 0.96))
        header = _rect(u, v, (0.03, 0.72, 0.97, 0.96))
        hu, hv = (u - 0.2) / 0.6, (v - 0.76) / 0.16
        head_text = _glyphs(np.clip(hu, 0, 1), hv, p["glyphs"], stencil=False) & (hu >= 0) & (hu <= 1)
        rows = np.zeros(len(u), bool)
        for y0, segs in p["rows"]:
            for x0, x1 in segs:
                rows |= _rect(u, v, (x0, y0, x1, y0 + 0.05))
        c[header & ~head_text] = ANSI_ORANGE
        c[border | rows] = BLACK
    elif kind == "cold":
        def tri(sc):
            cx, cy = 0.5, 0.37
            x, y = (u - cx) / sc + cx, (v - cy) / sc + cy
            return (y >= 0.06) & (y <= 0.94) & (np.abs(x - 0.5) <= 0.47 * (0.94 - y) / 0.88)
        outer, inner = tri(1.0), tri(0.8)
        # Snowflake (ISO 7010 W010 low temperature): three crossed bars with a V barb near each end.
        x, y = u - 0.5, v - 0.36
        mark = np.hypot(x, y) < 0.03
        for k in range(3):
            t = np.pi / 2 + np.pi * k / 3
            along, across = x * np.cos(t) + y * np.sin(t), -x * np.sin(t) + y * np.cos(t)
            mark |= (np.abs(along) < 0.21) & (np.abs(across) < 0.02)
            mark |= (np.abs(np.abs(along) - 0.14 - 0.6 * np.abs(across)) < 0.02) & (np.abs(across) < 0.07)
        a[outer] = 1
        c[:] = BLACK
        c[inner & ~mark] = YELLOW
    elif kind == "refband":
        a[:] = 1
        c[:] = YELLOW
        text = _glyphs((u - 0.06) / 0.6, (v - 0.25) / 0.5, p["glyphs"], stencil=False)
        arrow = (u >= 0.72) & (u <= 0.94) & (np.abs(v - 0.5) <= 0.38 * (0.94 - u) / 0.22)
        edge = (v < 0.08) | (v > 0.92)
        c[text | arrow | edge] = BLACK
    elif kind == "phase":
        a[:] = 1
        k = np.clip((u * 3).astype(int), 0, 2)
        c[:] = np.asarray(PHASES, np.float32)[k]
        c[(np.abs(u * 3 - np.round(u * 3)) < 0.04 * p["aspect"] / 3) | (v < 0.08) | (v > 0.92)] = BLACK
    elif kind == "papertape":
        # Punched 5-hole paper tape across a white label: data holes in five tracks, a smaller sprocket track between.
        a[:] = 1
        c[:] = WHITE
        tape = _rect(u, v, (0.04, 0.22, 0.96, 0.78))
        c[tape] = TAPE_CREAM
        bits = p["bits"]
        col = np.clip(((u - 0.04) / 0.92 * len(bits)).astype(int), 0, len(bits) - 1)
        cu = (u - 0.04) / 0.92 * len(bits) - col - 0.5
        tracks = (0.3, 0.4, 0.56, 0.64, 0.72)
        holes = np.zeros(len(u), bool)
        for k, tv in enumerate(tracks):
            holes |= bits[col, k] & (np.hypot(cu * 0.92 / len(bits) * p["aspect"], v - tv) < 0.035)
        holes |= tape & (np.hypot(cu * 0.92 / len(bits) * p["aspect"], v - 0.48) < 0.018)
        text = _glyphs((u - 0.06) / 0.5, (v - 0.04) / 0.14, p["glyphs"], stencil=False) & (u < 0.56)
        c[(holes & tape) | text | ~_rect(u, v, (0.02, 0.02, 0.98, 0.98))] = BLACK
    elif kind == "earth":
        # IEC 60417-5017 protective earth in a green/yellow striped ring.
        x, y = u - 0.5, v - 0.5
        r = np.hypot(x, y)
        a[r <= 0.48] = 1
        ring = r > 0.34
        stripe = ((x + y) / 0.12) % 1 < 0.5
        c[:] = WHITE
        c[ring] = np.where(stripe[ring, None], EARTH_GREEN, YELLOW)
        mark = (np.abs(x) < 0.025) & (y > -0.04) & (y < 0.22)
        for w, yy in ((0.2, -0.04), (0.13, -0.11), (0.06, -0.18)):
            mark |= (np.abs(x) < w / 2 + 0.01) & (np.abs(y - yy) < 0.02)
        c[mark] = BLACK
    elif kind == "coldstripe" or kind == "redhatch":
        a[:] = 1
        stripe = ((u * p["aspect"] + v) / 0.8) % 1 < 0.5
        c[:] = np.where(stripe[:, None], NAVY if kind == "coldstripe" else RESCUE_RED, WHITE)
    elif kind == "flame" or kind == "hotsurface":
        def tri(sc):
            cx, cy = 0.5, 0.37
            x, y = (u - cx) / sc + cx, (v - cy) / sc + cy
            return (y >= 0.06) & (y <= 0.94) & (np.abs(x - 0.5) <= 0.47 * (0.94 - y) / 0.88)
        outer, inner = tri(1.0), tri(0.8)
        base = _rect(u, v, (0.3, 0.15, 0.7, 0.2))
        if kind == "flame":
            # ISO 7010 W021: a flame over a bar, with a hollow inner tongue.
            def tongue(y0, y1, w):
                t = np.clip((v - y0) / (y1 - y0), 0, 1)
                half = w * (1 - t) ** 0.8 * np.minimum(1, 0.5 + 3 * t)
                return (v >= y0) & (v <= y1) & (np.abs(u - 0.5 - 0.03 * t * t) <= half)
            body, notch = tongue(0.22, 0.64, 0.14), tongue(0.22, 0.44, 0.055)
            mark = base | (body & ~notch)
        else:
            # ISO 7010 W017: three wavy heat lines rising from a bar.
            mark = base.copy()
            for x0 in (0.4, 0.5, 0.6):
                mark |= (v >= 0.23) & (v <= 0.56) & (np.abs(u - x0 - 0.025 * np.sin((v - 0.23) * 28)) < 0.02)
        a[outer] = 1
        c[:] = BLACK
        c[inner & ~mark] = YELLOW
    elif kind == "diamond":
        # NFPA 704 fire diamond: health (blue) left, fire (red) top, reactivity (yellow) right, special (white) bottom.
        x, y = u - 0.5, v - 0.5
        d = np.abs(x) + np.abs(y)
        a[d <= 0.49] = 1
        top, bottom = (y > np.abs(x)), (-y > np.abs(x))
        c[:] = np.where((x < 0)[:, None], np.asarray((0.02, 0.08, 0.45), np.float32), np.asarray(YELLOW, np.float32))
        c[top] = (0.6, 0.02, 0.015)
        c[bottom] = WHITE
        digit = np.zeros(len(u), bool)
        for cx, cy in ((0.0, 0.24), (-0.24, 0.0), (0.24, 0.0)):
            digit |= _rect(x, y, (cx - 0.03, cy - 0.07, cx + 0.03, cy + 0.07))
        lines = (np.abs(np.abs(x) - np.abs(y)) < 0.012) | (d > 0.45)
        c[(digit | lines) & (d <= 0.49)] = BLACK
    elif kind == "radiation":
        # Classic trefoil: black on yellow, three 60-degree blades around a dot.
        x, y = u - 0.5, v - 0.5
        r, t = np.hypot(x, y), np.degrees(np.arctan2(y, x))
        blade = (np.abs(((t - 90) % 120) - 60) > 30) & (r >= 0.11) & (r <= 0.36)
        mark = (r <= 0.07) | blade
        border = (np.abs(x) > 0.45) | (np.abs(y) > 0.45)
        a[:] = 1
        c[:] = YELLOW
        c[mark | border] = BLACK
    elif kind == "rf":
        def tri(sc):
            cx, cy = 0.5, 0.37
            x, y = (u - cx) / sc + cx, (v - cy) / sc + cy
            return (y >= 0.06) & (y <= 0.94) & (np.abs(x - 0.5) <= 0.47 * (0.94 - y) / 0.88)
        outer, inner = tri(1.0), tri(0.8)
        # ISO 7010 W005 non-ionising radiation: an antenna mast with radiating arcs.
        x, y = u - 0.5, v - 0.46
        r, t = np.hypot(x, y), np.arctan2(y, x)
        mast = _rect(u, v, (0.48, 0.14, 0.52, 0.46)) | (np.hypot(x, y) < 0.035)
        arcs = np.zeros(len(u), bool)
        for rr in (0.09, 0.15):
            arcs |= (np.abs(r - rr) < 0.017) & (np.abs(np.abs(t) - np.pi / 2) > np.pi / 4)
        mark = mast | arcs
        a[outer] = 1
        c[:] = BLACK
        c[inner & ~mark] = YELLOW
    elif kind == "chevron":
        a[:] = 1
        stripe = ((np.abs(u - 0.5) * p["aspect"] + v) / 0.7) % 1 < 0.5
        c[:] = np.where(stripe[:, None], YELLOW, BLACK)
    elif kind == "biohazard":
        border = _rect(u, v, (0.0, 0.0, 1.0, 1.0)) & ~_rect(u, v, (0.05, 0.05, 0.95, 0.95))
        x, y = u - 0.5, v - 0.47
        mark = np.zeros(len(u), bool)
        for k in range(3):
            t = np.pi / 2 + 2 * np.pi * k / 3
            dx, dy = np.cos(t), np.sin(t)
            lobe = np.hypot(x - 0.13 * dx, y - 0.13 * dy) <= 0.2
            bite = np.hypot(x - 0.2 * dx, y - 0.2 * dy) <= 0.14
            mark |= lobe & ~bite
        r = np.hypot(x, y)
        mark |= (r >= 0.12) & (r <= 0.16)
        mark &= r >= 0.05
        a[:] = 1
        c[:] = YELLOW
        c[mark | border] = BLACK
    elif kind == "corrosive":
        x, y = np.abs(u - 0.5), np.abs(v - 0.5)
        diamond = x + y <= 0.49
        a[diamond] = 1
        c[:] = WHITE
        c[diamond & (x + y >= 0.44)] = BLACK
        c[diamond & (v < 0.42)] = BLACK
        for sx in (-1, 1):
            # Tilted test tube pouring inward, two drops, and the surface it eats into.
            cx = 0.5 + sx * 0.15
            tube = (np.abs((u - cx) + sx * 0.45 * (v - 0.7)) < 0.035) & (v > 0.6) & (v < 0.8)
            drop = (np.hypot(u - (cx - sx * 0.07), v - 0.53) < 0.025) | (np.hypot(u - (cx - sx * 0.07), v - 0.47) < 0.018)
            c[tube | drop] = BLACK
            c[_rect(u, v, (cx - 0.13, 0.3, cx + 0.07, 0.35) if sx < 0 else (cx - 0.07, 0.3, cx + 0.13, 0.35))] = WHITE
    elif kind == "stencil" or kind == "numbers":
        a[_glyphs(u, v, p["glyphs"], stencil=True)] = 1
    elif kind == "plate":
        a[:] = 1
        c[:] = p["color"]
        border = ~_rect(u, v, (0.04, 0.06, 0.96, 0.94))
        header = _rect(u, v, (0.04, 0.72, 0.96, 0.94))
        hu, hv = (u - 0.08) / 0.5, (v - 0.75) / 0.16
        head_text = _glyphs(np.clip(hu, 0, 1), hv, p["glyphs"], stencil=False) & (hu >= 0) & (hu <= 1)
        rows = np.zeros(len(u), bool)
        for r, (y0, segs) in enumerate(p["rows"]):
            for x0, x1 in segs:
                rows |= _rect(u, v, (x0, y0, x1, y0 + 0.055))
        dark = (border | header | rows) & ~head_text
        c[dark] = BLACK
    elif kind == "nostep":
        frame = ~_rect(u, v, (0.04 / p["aspect"] * 2, 0.1, 1 - 0.04 / p["aspect"] * 2, 0.9))
        text = _glyphs((u - 0.12) / 0.76, (v - 0.28) / 0.44, p["glyphs"], stencil=False)
        a[frame | text] = 1
    elif kind == "arrow":
        head = (u >= 0.72) & (np.abs(v - 0.5) <= 0.5 * (1 - u) / 0.28)
        shaft = _rect(u, v, (0.0, 0.28, 0.72, 0.72))
        text = _glyphs((u - 0.06) / 0.6, (v - 0.36) / 0.28, p["glyphs"], stencil=False)
        a[head | shaft] = 1
        c[:] = RESCUE_RED
        c[text & shaft] = WHITE
    elif kind == "stripe":
        a[:] = 1
    elif kind == "triangle":
        a[(v >= 0.08) & (v <= 0.92) & (np.abs(u - 0.5) <= 0.45 * (0.92 - v) / 0.84)] = 1
    elif kind == "roundel":
        d = np.hypot(u - 0.5, v - 0.5)
        ring = (d <= 0.48) & (d >= 0.36)
        a[ring | (d <= 0.24) | _rect(u, v, (0.12, 0.45, 0.88, 0.55)) & (d <= 0.48)] = 1
    return a, c


def _params(kind, rng, family, scale=1.0, accent=None, light=False):
    spec = KINDS[kind]
    h = scale * float(np.exp(rng.uniform(*np.log(spec["h"]))))
    p = dict(h=h)
    if "glyphs" in spec:
        n = int(rng.integers(spec["glyphs"][0], spec["glyphs"][1] + 1))
        p["glyphs"] = _random_glyphs(rng, n)
        p["aspect"] = 0.8 * n
    else:
        p["aspect"] = float(rng.uniform(*spec["aspect"]))
    if family == "paint":
        p["color"] = WHITE if rng.random() < 0.65 and not light else BLACK
    else:
        p["color"] = YELLOW if rng.random() < 0.6 else WHITE
    if kind in ("plate", "arcflash"):
        p["color"] = WHITE if rng.random() < 0.6 else YELLOW
        p["glyphs"] = _random_glyphs(rng, int(rng.integers(3, 6)))
        rows = []
        for y0 in (0.52, 0.38, 0.24, 0.1):
            x, segs = 0.08, []
            while x < 0.85:
                w = rng.uniform(0.06, 0.3)
                segs.append((x, min(x + w, 0.92)))
                x += w + rng.uniform(0.03, 0.06)
            rows.append((y0, segs))
        p["rows"] = rows
    if kind == "roundel":
        p["color"] = WHITE if rng.random() < 0.5 else YELLOW
    if kind == "papertape":
        p["bits"] = rng.random((22, 5)) < 0.45
        p["glyphs"] = _random_glyphs(rng, 5)
    if kind in ("nostep", "arrow", "refband"):
        p["glyphs"] = _random_glyphs(rng, 6)
    if accent is not None and (kind == "stripe" or (family == "paint" and rng.random() < 0.5)):
        p["color"] = accent
    if kind in STICKERS and rng.random() < 0.35:
        p["backing"] = True
        if np.allclose(p["color"], WHITE):
            p["color"] = BLACK
    return p


def _frame(n0):
    up = np.array([0.0, 0.0, 1.0]) if abs(n0[2]) < 0.85 else np.array([0.0, 1.0, 0.0])
    up = up - n0 * (up @ n0)
    up /= np.linalg.norm(up)
    return np.cross(up, n0), up


def place(part, cfg, seed, log=print):
    """cfg: {count, mirror, targets: {material: weight}, scale, mix: {family: {kind: weight}}, accent: rgb,
    light_paint: [materials that get dark markings]}, plus m_per_stud. scale multiplies every marking size and the pinch radius. Writes part.maps decal_*."""
    m, P = part.maps, part.pos
    n = m["normal_obj"].astype(np.float64)
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    N = len(P)
    unit = cfg["m_per_stud"]
    scale = cfg.get("scale", 1.0)
    accent = cfg.get("accent")
    mix = {f: cfg.get("mix", {}).get(f, MIX[f]) for f in MIX}
    mats = part.meta["materials"]
    mid = np.round(m["matid"] * 32 - 0.5).astype(int)
    weight = np.zeros(N)
    family_of = {}
    for mat, w in cfg["targets"].items():
        if mat in mats:
            weight[mid == mats.index(mat)] = w
            family_of[mats.index(mat)] = "frame" if "gunmetal" in mat else "paint"
    flat = (m["ao_assembled"] > 0.85) & (m["convex_large"] < 0.12) & (m["convex_small"] < 0.05)
    cand = np.flatnonzero((weight > 0) & flat)
    # Joints are cluttered and partly shadowed; their markers may sit in less open spots than random ones.
    # convex_large is skipped: thin pressed panels over inner shells read convex across their whole face, and fits()
    # checks planarity itself.
    cand_pinch = np.flatnonzero((weight > 0) & (m["ao_assembled"] > 0.6) & (m["convex_small"] < 0.05))
    cdf = np.cumsum(weight[cand])
    tree = cKDTree(P)
    rng = np.random.default_rng([seed, 4242])
    alpha = np.zeros(N, np.float32)
    color = np.zeros((N, 3), np.float32)
    placed = []

    def fits(c, n0, w, h):
        right, up = _frame(n0)
        r = 0.5 * np.hypot(w, h)
        if any(np.linalg.norm(c - pc) < r + pr for pc, pr in placed):
            return None
        idx = np.asarray(tree.query_ball_point(c, r), np.int64)
        d = P[idx] - c
        du, dv, dn = d @ right, d @ up, d @ n0
        inside = (np.abs(du) < w / 2) & (np.abs(dv) < h / 2)
        planar = inside & (np.abs(dn) < 0.05 * h) & (n[idx] @ n0 > 0.97)
        if (part.texel[idx[planar]].astype(np.float64) ** 2).sum() < 0.93 * w * h:
            return None
        return idx[planar], du[planar] / w + 0.5, dv[planar] / h + 0.5, r

    def stamp(kind, p, hit, c):
        idx, u, v, r = hit
        tx = part.texel[idx] / (p["h"] * unit)
        acc_a = np.zeros(len(idx), np.float32)
        acc_c = np.zeros((len(idx), 3), np.float32)
        for ou, ov in ((-0.25, -0.25), (0.25, -0.25), (-0.25, 0.25), (0.25, 0.25)):
            a, col = _shape(kind, u + ou * tx / p["aspect"], v + ov * tx, p)
            acc_a += a / 4
            acc_c += col * a[:, None] / 4
        col = acc_c / np.maximum(acc_a, 1e-6)[:, None]
        color[idx] = color[idx] + (col - color[idx]) * acc_a[:, None]
        alpha[idx] = acc_a + alpha[idx] * (1 - acc_a)
        placed.append((c, r))

    counts = {}
    pinch = cfg.get("pinch", {})
    pinch_kinds = set()
    for pt in pinch.get("points", []):
        kinds = pt.get("kinds", pinch.get("kinds", ["hazard", "warning"]))
        pinch_kinds |= set(kinds)
        for sx in ((1, -1) if pt.get("mirror_x") else (1,)):
            # Roblox build space (studs, x right, y up, -z forward) to sim space.
            at = np.array(pt["at"], np.float64) * np.array([sx, 1, 1])
            c0 = np.array([at[0], -at[2], at[1]]) * unit
            near = cand_pinch[np.linalg.norm(P[cand_pinch] - c0, axis=1) < pinch.get("radius_studs", 0.35) * scale * unit]
            near = near[np.argsort(np.linalg.norm(P[near] - c0, axis=1) + rng.random(len(near)) * 0.1 * scale * unit)]
            want, k = pt.get("count", pinch.get("per_point", 2)), 0
            for i in near[::max(1, len(near) // 60)]:
                if k >= want:
                    break
                kind = kinds[k % len(kinds)]
                p = _params(kind, rng, family_of[mid[i]], scale)
                for shrink in (1.0, 0.75, 0.55, 0.4):
                    p["h"] = p["h"] * shrink if shrink < 1 else p["h"]
                    h, w = p["h"] * unit, p["h"] * unit * p["aspect"]
                    hit = fits(P[i], n[i], w, h)
                    if hit is not None:
                        break
                if hit is None:
                    continue
                stamp(kind, p, hit, P[i])
                counts[kind] = counts.get(kind, 0) + 1
                k += 1
            if k == 0 and len(near):
                log(f"[decals] no flat spot for a pinch marker near {pt['at']} (x{sx:+d})")
    n_pinch = len(placed)

    tries = 0
    # An atlas of only non-target stacks (a walnut frill) has nowhere to put random markings.
    while len(cand) and len(placed) - n_pinch < cfg["count"] and tries < cfg["count"] * 80:
        tries += 1
        i = cand[min(np.searchsorted(cdf, rng.random() * cdf[-1]), len(cand) - 1)]
        family = family_of[mid[i]]
        # With pinch points given, hazard markings only go where they mean something.
        kinds = [k for k in mix[family] if k not in pinch_kinds]
        wk = np.array([mix[family][k] for k in kinds])
        kind = kinds[rng.choice(len(kinds), p=wk / wk.sum())]
        p = _params(kind, rng, family, scale, accent, mats[mid[i]] in cfg.get("light_paint", []))
        for shrink in (1.0, 0.75, 0.55):
            p["h"] = p["h"] * shrink if shrink < 1 else p["h"]
            h, w = p["h"] * unit, p["h"] * unit * p["aspect"]
            hit = fits(P[i], n[i], w, h)
            if hit is not None:
                break
        if hit is None:
            continue
        stamp(kind, p, hit, P[i])
        counts[kind] = counts.get(kind, 0) + 1
        if rng.random() < cfg.get("mirror", 0) and abs(P[i, 0]) > 4 * w:
            cm = P[i] * np.array([-1.0, 1.0, 1.0])
            dist, j = tree.query(cm)
            if dist < 3 * part.texel[j] and mid[j] == mid[i]:
                hit = fits(P[j], n[j], w, h)
                if hit is not None:
                    stamp(kind, p, hit, P[j])
                    counts[kind] += 1
    m["decal_alpha"] = alpha
    m["decal_color"] = color
    log(f"[decals] {len(placed)} placed ({n_pinch} at pinch points) in {tries} tries: {counts}")

