"""3D value noise evaluated at texel positions, so patterns are continuous across UV seams.

fbm runs as a parallel numba kernel; it must stay bit-compatible with _hash (used directly by wear.py)."""
import numpy as np
from numba import njit, prange

_P = np.array([73856093, 19349663, 83492791], np.int64)


def _hash(ix, iy, iz, seed):
    h = (ix * _P[0]) ^ (iy * _P[1]) ^ (iz * _P[2]) ^ np.int64(seed * 2654435761 & 0x7FFFFFFF)
    h = (h ^ (h >> 13)) * np.int64(1274126177)
    h = h ^ (h >> 16)
    return (h & 0xFFFF).astype(np.float32) * np.float32(1.0 / 65535.0)


@njit(inline="always")
def _hash1(ix, iy, iz, s):
    h = (ix * 73856093) ^ (iy * 19349663) ^ (iz * 83492791) ^ s
    h = (h ^ (h >> 13)) * 1274126177
    h = h ^ (h >> 16)
    return np.float32(h & 0xFFFF) * np.float32(1.0 / 65535.0)


@njit(inline="always")
def _value(x, y, z, s):
    fx, fy, fz = np.floor(x), np.floor(y), np.floor(z)
    x0, y0, z0 = np.int64(fx), np.int64(fy), np.int64(fz)
    ux, uy, uz = np.float32(x - fx), np.float32(y - fy), np.float32(z - fz)
    ux = ux * ux * (3 - 2 * ux)
    uy = uy * uy * (3 - 2 * uy)
    uz = uz * uz * (3 - 2 * uz)
    a = _hash1(x0, y0, z0, s)
    c00 = a + (_hash1(x0 + 1, y0, z0, s) - a) * ux
    a = _hash1(x0, y0 + 1, z0, s)
    c10 = a + (_hash1(x0 + 1, y0 + 1, z0, s) - a) * ux
    a = _hash1(x0, y0, z0 + 1, s)
    c01 = a + (_hash1(x0 + 1, y0, z0 + 1, s) - a) * ux
    a = _hash1(x0, y0 + 1, z0 + 1, s)
    c11 = a + (_hash1(x0 + 1, y0 + 1, z0 + 1, s) - a) * ux
    c0 = c00 + (c10 - c00) * uy
    c1 = c01 + (c11 - c01) * uy
    return c0 + (c1 - c0) * uz


@njit(parallel=True, cache=True)
def _fbm(p, freq, stride, octaves, seeds, gain):
    out = np.empty(p.shape[0], np.float32)
    for i in prange(p.shape[0]):
        f = freq[i * stride]
        acc, amp, norm = np.float32(0), 1.0, 0.0
        for o in range(octaves):
            acc += np.float32(amp) * _value(p[i, 0] * f, p[i, 1] * f, p[i, 2] * f, seeds[o])
            norm += amp
            amp *= gain
            f *= 2.0
        out[i] = acc / np.float32(norm)
    return out


def fbm(p, freq, octaves=4, seed=0, gain=0.5):
    """Fractal value noise, (N,) in [0, 1] with mean ~0.5. Contrast is low; remap with smoothstep.
    p: (N, 3) meters. freq: cycles per meter, scalar or (N,)."""
    seeds = np.array([(seed * 31 + o * 1013) * 2654435761 & 0x7FFFFFFF for o in range(octaves)], np.int64)
    freq = np.atleast_1d(np.asarray(freq, np.float64))
    return _fbm(np.ascontiguousarray(p, np.float64), freq, int(len(freq) > 1), octaves, seeds, gain)


@njit(parallel=True, cache=True)
def _cell_seeds(p, cell, seeds):
    out = np.empty_like(p)
    for i in prange(p.shape[0]):
        c = cell[i]
        qx, qy, qz = p[i, 0] / c, p[i, 1] / c, p[i, 2] / c
        bx, by, bz = np.int64(np.floor(qx)), np.int64(np.floor(qy)), np.int64(np.floor(qz))
        best = np.inf
        sx = sy = sz = 0.0
        for ox in range(-1, 2):
            for oy in range(-1, 2):
                for oz in range(-1, 2):
                    cx, cy, cz = bx + ox, by + oy, bz + oz
                    px = cx + 0.1 + 0.8 * _hash1(cx, cy, cz, seeds[0])
                    py = cy + 0.1 + 0.8 * _hash1(cx, cy, cz, seeds[1])
                    pz = cz + 0.1 + 0.8 * _hash1(cx, cy, cz, seeds[2])
                    d = (qx - px) ** 2 + (qy - py) ** 2 + (qz - pz) ** 2
                    if d < best:
                        best, sx, sy, sz = d, px, py, pz
        out[i, 0], out[i, 1], out[i, 2] = sx * c, sy * c, sz * c
    return out


def cell_seeds(p, cell, seed):
    """Nearest jittered-grid seed (3D Voronoi) for each point. p: (M, 3) meters, cell: (M,) meters."""
    seeds = np.array([(seed + i) * 2654435761 & 0x7FFFFFFF for i in range(3)], np.int64)
    return _cell_seeds(np.ascontiguousarray(p, np.float64), np.ascontiguousarray(cell, np.float64), seeds)


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)
