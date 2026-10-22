"""Triplanar sampling of material texture sets at texel object-space positions (no UVs, so no seams)."""
import os
from functools import lru_cache

import numpy as np
from numba import njit, prange
from PIL import Image

from .materials import DETAIL

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(_ROOT, "assets", "materials")
MASKS = os.path.join(_ROOT, "assets", "masks")
# Each projection plane is shifted so the three projections do not line up on axis-aligned corners.
PLANE_OFFSETS = np.array([[0.0, 0.0], [0.37, 0.61], [0.73, 0.19]], np.float32)


def _gray(path):
    a = np.asarray(Image.open(path))
    if a.dtype == bool:
        return a.astype(np.float32)
    if a.ndim == 3:
        a = a[..., 0]
    return a.astype(np.float32) / (65535.0 if a.max() > 255 else 255.0)


@lru_cache(maxsize=None)
def load(tex_id):
    """Stacked (H, W, 3) float32: luminance variation (mean 1), roughness offset (mean 0), height (std 1)."""
    d = os.path.join(ASSETS, tex_id)
    c = np.asarray(Image.open(os.path.join(d, "color.png")).convert("RGB")).astype(np.float32) / 255
    c = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    luma = c @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    luma = luma / max(float(luma.mean()), 1e-6)
    rough = _gray(os.path.join(d, "roughness.png"))
    rough -= rough.mean()
    h = _gray(os.path.join(d, "height.png")) if os.path.exists(os.path.join(d, "height.png")) else np.zeros_like(rough)
    h = (h - h.mean()) / max(float(h.std()), 1e-6)
    return np.ascontiguousarray(np.stack([luma, rough, h], -1).astype(np.float32))


@lru_cache(maxsize=None)
def load_mask(mask_id):
    """Mask normalized so its 99th percentile is 1; the CC0 sets range from 0.7% to 55% mean coverage."""
    m = _gray(os.path.join(MASKS, mask_id, "mask.png"))
    m = np.clip(m / max(float(np.percentile(m, 99)), 1e-3), 0, 1)
    return np.ascontiguousarray(m[..., None].astype(np.float32))


@njit(parallel=True, cache=True)
def _triplanar_kernel(img, P, n, scale, offsets):
    H, W, C = img.shape
    out = np.zeros((P.shape[0], C), np.float32)
    for i in prange(P.shape[0]):
        w0, w1, w2 = abs(n[i, 0]) ** 4, abs(n[i, 1]) ** 4, abs(n[i, 2]) ** 4
        ws = max(w0 + w1 + w2, 1e-9)
        for axis in range(3):
            w = (w0, w1, w2)[axis] / ws
            if w < 1e-4:
                continue
            a, b = (1, 2) if axis == 0 else ((0, 2) if axis == 1 else (0, 1))
            x = (P[i, a] * scale + offsets[axis, 0]) * W - 0.5
            y = (P[i, b] * scale + offsets[axis, 1]) * H - 0.5
            fx0, fy0 = np.floor(x), np.floor(y)
            fx, fy = np.float32(x - fx0), np.float32(y - fy0)
            x0, y0 = np.int64(fx0) % W, np.int64(fy0) % H
            x1, y1 = (x0 + 1) % W, (y0 + 1) % H
            for c in range(C):
                top = img[y0, x0, c] * (1 - fx) + img[y0, x1, c] * fx
                bot = img[y1, x0, c] * (1 - fx) + img[y1, x1, c] * fx
                out[i, c] += np.float32(w) * (top * (1 - fy) + bot * fy)
    return out


def _triplanar(img, P, n, tile_mm, offsets=PLANE_OFFSETS):
    return _triplanar_kernel(img, np.ascontiguousarray(P, np.float64), np.ascontiguousarray(n, np.float32),
                             1000.0 / tile_mm, np.ascontiguousarray(offsets, np.float64))


def sample_mask(mask_id, tile_mm, P, n, seed=0):
    """(N,) mask value in [0, 1]; seed shifts the projections so repeated masks don't line up."""
    offsets = (PLANE_OFFSETS + np.random.default_rng(seed).random((3, 2))).astype(np.float32)
    return _triplanar(load_mask(mask_id), P, n, tile_mm, offsets)[:, 0]


def sample(ref, P, n):
    """ref: texture reference from materials.tex(). P: (N, 3) meters, n: (N, 3) unit normals.
    Returns dict(color_mul (N,), rough_add (N,), height_mm (N,))."""
    out = _triplanar(load(ref["id"]), P, n, ref["tile_mm"])
    return dict(color_mul=1 + ref["color_var"] * (out[:, 0] - 1),
                rough_add=ref["rough_var"] * out[:, 1],
                height_mm=ref["height_um"] / 1000.0 * out[:, 2])


def detail(family, P, n, detail_scale, seed):
    """Max over the family's picked CC0 masks (materials.DETAIL), (N,) in [0, 1]."""
    cfg = DETAIL[family]
    rng = np.random.default_rng([seed, sum(map(ord, family))])
    v = np.zeros(len(P), np.float32)
    for i, mask_id in enumerate(rng.choice(cfg["ids"], cfg["pick"], replace=False)):
        tile = cfg["tile_mm"] * detail_scale * rng.uniform(0.8, 1.25)
        v = np.maximum(v, sample_mask(mask_id, tile, P, n, seed + 97 * i))
    return v
