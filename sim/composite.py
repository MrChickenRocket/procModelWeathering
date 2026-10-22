"""Turn simulated layer state into Roblox SurfaceAppearance maps."""
import copy
import os

import numpy as np
from PIL import Image

from . import texsample
from .materials import BASES, DUST, GRIME, HEAT_TINT, RUST, SOOT, STACKS
from .noise import fbm, smoothstep
from .wear import scale_params

AO_STRENGTH = 0.75
DEBUG_FIELDS = ("contact", "coat", "wear", "rust", "impact", "scratch", "grime", "dust", "polymer_wear", "rounding",
                "scorch", "streaks", "decal_alpha", "lip", "edge_band")
DEBUG_SIZE = 1024


def _mix(a, b, t):
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    if a.ndim == 2 or b.ndim == 2:
        t = t[:, None]
    return a + (b - a) * t


def _detail(family, part, prm, seed):
    return texsample.detail(family, part.pos, part.maps["normal_obj"], prm.detail_scale, seed)


def _weave(P, n, cfg):
    """Plain-weave height in mm, triplanar from the surface position: warp and weft threads of round section that
    alternate over and under every cell_mm. Projections blend by |n|^4, so axis changes soften instead of tearing."""
    q = P / (cfg["cell_mm"] / 1000.0)
    w = np.abs(n) ** 4
    w /= np.maximum(w.sum(1, keepdims=True), 1e-9)
    h = np.zeros(len(P), np.float32)
    for ax, (i, j) in enumerate(((1, 2), (0, 2), (0, 1))):
        a, b = q[:, i], q[:, j]
        warp = np.sqrt(np.clip(1 - (2 * (b % 1) - 1) ** 2, 0, 1)) * (0.6 + 0.4 * np.cos(np.pi * (a + np.floor(b))))
        weft = np.sqrt(np.clip(1 - (2 * (a % 1) - 1) ** 2, 0, 1)) * (0.6 - 0.4 * np.cos(np.pi * (b + np.floor(a))))
        h += w[:, ax] * np.maximum(warp, weft)
    return h * cfg["height_um"] / 1000.0


def shade(part, F, prm):
    """Maps from the sim fields. An overlay stack also gets alpha and overlay_rgb: the colour pass is linear in the
    part Color C (col = A C + B), so two passes with C at 0 and 0.5 give Overlay's col = (1 - a) C + a W."""
    name = part.material
    stack = STACKS[name]
    out = _shade(part, F, prm)
    N = len(part.idx)
    key = stack.get("overlay")
    if not key:
        out.update(alpha=np.ones(N, np.float32), overlay_rgb=out["color"])
        return out
    cols = []
    for c in (0.0, 0.5):
        s = copy.deepcopy(stack)
        if key == "base":
            s["color"] = (c, c, c)
        else:
            next(lay for lay in s["layers"] if lay["name"] == key)["color"] = (c, c, c)
        STACKS[name] = s
        try:
            cols.append(_shade(part, dict(F), prm, raw=True)["color"])
        finally:
            STACKS[name] = stack
    b, h = cols
    a = np.clip(1 - 2 * (h - b).mean(1), 0, 1)
    # Below a few percent the overlay only carries rounding noise of the topcoat itself.
    a = np.where(a < 0.03, 0, a).astype(np.float32)
    w = np.where(a[:, None] > 0, b / np.maximum(a, 1e-6)[:, None], 0)
    out.update(alpha=a, overlay_rgb=np.clip(w, 0, 1).astype(np.float32))
    return out


def _shade(part, F, prm, raw=False):
    m, P = part.maps, part.pos
    n = m["normal_obj"]
    N = len(part.idx)
    stack = STACKS[part.material]
    prm = scale_params(prm, stack.get("weather", {}))
    base, layers = BASES[stack["base"]], stack["layers"]
    overlay = bool(stack.get("overlay"))
    seed = prm.seed * 7919 + part.seed_salt
    # Lenses stay clean: the emissive mask comes from the slot, and wear or grime on the glow tint would dim it.
    clean = base.get("clean", False)

    def tex(ref):
        return texsample.sample(ref, P, n) if ref else dict(color_mul=1.0, rough_add=0.0, height_mm=0.0)

    t = tex(base.get("tex"))
    # An overlaid base is the part's flat Color in game, so its texture keeps roughness and height but not tone.
    cmul = 1.0 if stack.get("overlay") == "base" else t["color_mul"]
    col = np.asarray(stack.get("color", base["color"]), np.float32)[None, :] * np.broadcast_to(cmul, (N,))[:, None]
    metal = np.full(N, base["metal"], np.float32)
    rough = stack.get("rough", base["rough"]) + t["rough_add"] + np.zeros(N, np.float32)
    detail = t["height_mm"] + np.zeros(N, np.float32)
    if "weave" in stack:
        detail = detail + _weave(P, n, stack["weave"])
    level = np.zeros(N, np.float32)
    polish, scratch = F["wear"], F["scratch"]
    flatten = np.zeros(N, np.float32)

    if clean:
        pass
    elif base["metal"] > 0.5:
        rough = rough - 0.2 * polish + 0.12 * scratch
        col = col * (1 + 0.12 * polish + 0.15 * scratch)[:, None]
    else:
        # Handled polymer polishes in broad patches: darker, glossier, moulded texture worn flat.
        # Driven by smooth fields only; the abrasion field carries fine noise that reads as speckle here.
        patches = smoothstep(0.3, 0.7, fbm(P, 45, 3, seed + 7))
        edge = np.clip(m["convex_small"] + m["convex_large"], 0, 1)
        drive = F["contact"] * np.clip(m["ao_assembled"], 0, 1) ** 1.5 * (0.35 + 0.8 * edge)
        pw = smoothstep(0.2, 0.6, drive * (0.25 + 1.5 * patches) * 2 * prm.age * prm.use)
        # Corners take every knock regardless of grip patches; the sharpest edge line abrades matte grey.
        corner = smoothstep(0.3, 0.8, m["convex_small"]) * np.clip(m["ao_assembled"], 0, 1) * (0.5 + F["contact"])
        pw = np.maximum(pw, smoothstep(0.1, 0.4, corner * 2 * prm.age * prm.use))
        scuff = smoothstep(0.5, 0.9, m["convex_small"]) * smoothstep(0.2, 0.6, corner * 2 * prm.age * prm.use)
        # Over a player-picked Color the polish darkening reads as camo, so overlay keeps it mostly in roughness.
        col = col * (1 - (0.15 if overlay else 0.55) * pw)[:, None] + (0.05 * np.clip(scratch + 0.6 * F["impact"], 0, 1))[:, None]
        # Overlay can only blend toward a fixed colour, so scuffs lighten toward dust instead of scaling the paint.
        col = _mix(col, DUST["color"] if overlay else col * 2.5 + 0.035, (0.3 if overlay else 0.7) * scuff)
        rough = rough - 0.3 * pw + 0.1 * scratch + 0.4 * scuff
        detail = detail * (1 - 0.8 * pw)
        flatten = 0.75 * pw

    r = smoothstep(0.05, 0.8, F["rust"])
    rust_var = 0.65 + 0.7 * fbm(P, 1500, 3, seed + 6)
    corr = base.get("corrosion", RUST)

    def corrode(col, metal, rough, detail):
        col = _mix(col, np.asarray(corr["color"], np.float32) * rust_var[:, None], r)
        return col, _mix(metal, corr["metal"], r), _mix(rough, corr["rough"], r), detail + corr["height_mm"] * r * rust_var

    # Rust only forms on bare steel, under the coatings; verdigris grows through copper's tarnish film.
    over = base.get("corrodes_under_layers", False)
    if not over:
        col, metal, rough, detail = corrode(col, metal, rough, detail)
    for k, lay in enumerate(layers):
        f = F["layers"][:, k]
        a = smoothstep(0.0, 0.12, f)
        thin = 1 - smoothstep(0.12, 0.6, f)
        t = tex(lay["tex"])
        lcol = np.asarray(lay["color"], np.float32)[None, :] * (
            np.broadcast_to(t["color_mul"], (N,)) * (1 + 0.2 * thin + 0.4 * scratch))[:, None]
        # Worn-thin film turns translucent and shows what is underneath.
        lcol = _mix(lcol, col, 0.5 * thin)
        col = _mix(col, lcol, a)
        metal = _mix(metal, lay["metal"], a)
        rough = _mix(rough, lay["rough"] + t["rough_add"] - 0.15 * thin + 0.08 * scratch, a)
        level += lay["step_um"] / 1000.0 * f * a
        # Film levels most of the texture below it; its own surface detail fades as it wears thin.
        detail = _mix(detail, 0.25 * detail + t["height_mm"] * (1 - thin), a)
    if over:
        col, metal, rough, detail = corrode(col, metal, rough, detail)
    if "decal_alpha" in m:
        # Stencils and stickers sit on the topcoat: they chip with it and fade unevenly.
        da = m["decal_alpha"] * (smoothstep(0.0, 0.12, F["layers"][:, -1]) if layers else 1.0)
        da = da * (1 - 0.6 * prm.age * smoothstep(0.45, 0.8, fbm(P, 700, 3, seed + 8)))
        col = _mix(col, m["decal_color"], da)
        rough = _mix(rough, 0.5, da)
        metal = _mix(metal, 0.0, da)
        level += 0.004 * da
        F["decal_alpha"] = da
    contact = F["contact"]
    expo = np.clip(m["ao_assembled"], 0, 1) ** 1.5
    if prm.detail > 0 and not clean:
        # Micro-scratches haze gloss and burnish matte finishes, so roughness moves toward the middle.
        sc = _detail("scratches", part, prm, seed + 40) * np.clip(1.5 * contact + 0.15, 0, 1) * expo * prm.detail
        rough = rough + (0.45 - rough) * 0.5 * sc
        col = col * (1 + sc * (0.2 * (1 - metal) + 0.1 * metal))[:, None]
        detail = detail - 0.003 * sc
        im = _detail("imperfections", part, prm, seed + 41) * (0.3 + 0.7 * expo) * prm.detail
        rough = rough + (0.6 - rough) * 0.25 * im
        fp = _detail("fingerprints", part, prm, seed + 42) * smoothstep(0.25, 0.8, contact) * prm.detail
        rough = rough + (0.5 - rough) * 0.5 * fp
        col = col * (1 - 0.08 * fp)[:, None]
        sm = _detail("smears", part, prm, seed + 43) * expo * (1 - contact) * prm.dirt * prm.detail
        col = _mix(col, GRIME["color"], 0.35 * sm)
        rough = rough + (0.7 - rough) * 0.4 * sm
        F.update(detail_scratches=sc, detail_imperfections=im, detail_fingerprints=fp, detail_smears=sm)
    sc = F["scorch"] * (not clean)
    if sc.any():
        bare = 1 - smoothstep(0.0, 0.1, F["layers"][:, 0]) if layers else float(base["metal"] > 0.5)
        t = np.clip((sc - 0.75) * 2 + 0.4 * (fbm(P, 250, 3, seed + 9) - 0.5), 0, 1) * (len(HEAT_TINT) - 1)
        i0 = np.minimum(t.astype(int), len(HEAT_TINT) - 2)
        tint = _mix(np.asarray(HEAT_TINT, np.float32)[i0], np.asarray(HEAT_TINT, np.float32)[i0 + 1], t - i0)
        col = _mix(col, tint * col.mean(1, keepdims=True) / 0.57, 0.35 * bare * smoothstep(0.75, 1.0, sc) * (base["metal"] > 0.5))
        soot = smoothstep(0.08, 0.45, sc) * (0.75 + 0.25 * fbm(P, 300, 3, seed + 10))
        col = _mix(col, SOOT["color"], 0.92 * soot)
        rough = _mix(rough, SOOT["rough"], 0.8 * soot)
        metal = metal * (1 - soot)
        blister = smoothstep(0.3, 0.55, sc) * (1 - smoothstep(0.6, 0.8, sc))
        detail = detail + 0.02 * blister * smoothstep(0.55, 0.75, fbm(P, 2500, 2, seed + 11))
    st = F["streaks"] * 0.7 * (not clean)
    col = _mix(col, GRIME["color"], st)
    rough = _mix(rough, GRIME["rough"], 0.6 * st)
    height = level + detail + F["dent_mm"] - 0.008 * scratch

    g = F["grime"] * 0.85 * (not clean)
    col = _mix(col, GRIME["color"], g)
    rough = _mix(rough, GRIME["rough"], g)
    metal = metal * (1 - 0.6 * g)
    du = F["dust"] * 0.8 * (not clean)
    col = _mix(col, DUST["color"], du)
    rough = _mix(rough, DUST["rough"], du)
    metal = metal * (1 - du)

    if not overlay:
        # Overlay maps carry wear only; the part's flat Color has no baked occlusion either.
        col = col * (1 - AO_STRENGTH * (1 - np.clip(m["ao"], 0, 1)))[:, None]
    F["polymer_wear"] = flatten / 0.75
    corner = smoothstep(0.3, 0.8, m["convex_small"]) * np.clip(m["ao_assembled"], 0, 1) * (0.5 + F["contact"])
    worn_round = smoothstep(0.1, 0.4, corner * 2 * prm.age * prm.use)
    rounding = np.clip(base["round"] + (1 - base["round"]) * worn_round, 0, 1).astype(np.float32)
    F["rounding"] = rounding
    return dict(color=col if raw else np.clip(col, 0, 1), metal=np.clip(metal, 0, 1),
                rough=np.clip(rough, 0.02, 1), height_mm=height, flatten=flatten, rounding=rounding,
                emissive=np.full(N, float(base.get("emissive", False)), np.float32))


def scatter(part, v, fill=0.0):
    S = part.size
    tail = v.shape[1:]
    img = np.full((S * S,) + tail, fill, np.float32)
    img[part.idx] = v
    return img.reshape((S, S) + tail)


def height_to_normal(part, height_mm, normal_ts):
    """Detail normal from a height field, UDN-blended onto the baked tangent normal.
    Rows run +v and columns +u, which are the OpenGL tangent-space Y and X."""
    mask = part.mask
    h = scatter(part, height_mm) / 1000.0
    texel = np.maximum(part.texel_img, 1e-9)

    def deriv(axis):
        f, b = np.roll(h, -1, axis), np.roll(h, 1, axis)
        mf, mb = np.roll(mask, -1, axis), np.roll(mask, 1, axis)
        d = np.where(mf & mb, (f - b) / 2, np.where(mf, f - h, np.where(mb, h - b, 0)))
        return d / texel

    nb = scatter(part, normal_ts)
    n = np.stack([nb[..., 0] - deriv(1), nb[..., 1] - deriv(0), nb[..., 2]], -1)
    n /= np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-9)
    n[~mask] = (0, 0, 1)
    return n


def push_pull(img, valid):
    """Fill invalid texels from progressively coarser averages so every mip level stays clean."""
    if img.shape[0] <= 1:
        return img
    S = img.shape[0]
    C = img.shape[2]
    w = valid.astype(np.float32)
    acc = (img * w[..., None]).reshape(S // 2, 2, S // 2, 2, C).sum((1, 3))
    w2 = w.reshape(S // 2, 2, S // 2, 2).sum((1, 3))
    coarse = np.where(w2[..., None] > 0, acc / np.maximum(w2, 1e-9)[..., None], 0)
    coarse = push_pull(coarse, w2 > 0)
    up = coarse.repeat(2, 0).repeat(2, 1)
    return np.where(valid[..., None], img, up)


def _srgb(c):
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * np.power(c, 1 / 2.4) - 0.055)


def _save(path, arr):
    a = (np.clip(arr, 0, 1) * 255 + 0.5).astype(np.uint8)
    if a.ndim == 3 and a.shape[2] == 1:
        a = a[..., 0]
    # Arrays are stored v-up (row 0 = v 0); image files are top-down.
    Image.fromarray(np.flipud(a)).save(path)


def write_outputs(part, F, out, out_dir):
    os.makedirs(os.path.join(out_dir, "debug"), exist_ok=True)
    mask = part.mask
    color = push_pull(scatter(part, out["color"]), mask)
    metal = push_pull(scatter(part, out["metal"][:, None]), mask)
    rough = push_pull(scatter(part, out["rough"][:, None]), mask)
    # Baked normals flatten where wear has polished the surface down.
    baked = part.maps["normal_ts"] * (1 - out["flatten"])[:, None] + np.array([0, 0, 1], np.float32) * out["flatten"][:, None]
    if "normal_ts_round" in part.maps:
        # Whiteout blend: crisp high-poly detail on top of the low-poly's large-radius edge rounding.
        w = out["rounding"][:, None]
        r = part.maps["normal_ts_round"] * w + np.array([0, 0, 1], np.float32) * (1 - w)
        baked = np.concatenate([baked[:, :2] + r[:, :2], (baked[:, 2] * r[:, 2])[:, None]], 1)
        baked /= np.maximum(np.linalg.norm(baked, axis=1, keepdims=True), 1e-9)
    normal = push_pull(height_to_normal(part, out["height_mm"], baked), mask)
    normal /= np.maximum(np.linalg.norm(normal, axis=-1, keepdims=True), 1e-9)

    _save(os.path.join(out_dir, "color.png"), _srgb(color))
    if (out["alpha"] < 1).any():
        # Straight alpha; texels under alpha 0 take the nearest wear colour so filtering never pulls in the paint.
        a = scatter(part, out["alpha"][:, None])
        valid = mask & (a[..., 0] > 0)
        rgb = push_pull(scatter(part, out["overlay_rgb"]), valid) if valid.any() else np.zeros_like(color)
        _save(os.path.join(out_dir, "overlay.png"), np.concatenate([_srgb(rgb), a], -1))
    _save(os.path.join(out_dir, "metalness.png"), metal)
    _save(os.path.join(out_dir, "roughness.png"), rough)
    _save(os.path.join(out_dir, "normal.png"), normal * 0.5 + 0.5)
    if out["emissive"].any() or "matid" in part.maps:
        # Atlas parts always get one, black when the atlas holds no lens, so every atlas has the same map set.
        # SurfaceAppearance EmissiveMaskContent: greyscale, white glows with the colour map as tint.
        _save(os.path.join(out_dir, "emissive.png"), push_pull(scatter(part, out["emissive"][:, None]), mask))

    step = max(1, part.size // DEBUG_SIZE)
    debug = {k: F[k] for k in F if k in DEBUG_FIELDS or k.startswith(("detail_", "layer_"))}
    for k, v in debug.items():
        img = scatter(part, v / max(float(v.max()), 1e-9) if k == "contact" else v)
        _save(os.path.join(out_dir, "debug", f"{k}.png"), img[::step, ::step])
