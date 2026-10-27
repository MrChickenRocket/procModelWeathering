"""Per-panel fields for the model-kit finish, from the baked piece-id map.

tone: -1, 0 or +1 per texel, a graph colouring of the pieces so touching or nearby panels differ.
piece_seam: 0..1 line where a piece meets another one, for panel lining between pieces.
"""
import numpy as np
from scipy.spatial import cKDTree

from .noise import smoothstep

VARIANTS = (0, 1, -1)


def _colour(ids, edges):
    """Greedy in id order. Each piece starts its preference at id % 3, so the first piece in a cluster doesn't
    always take the base tone; the result depends only on the ids and the adjacency."""
    nbrs = {i: set() for i in ids}
    for a, b in edges:
        nbrs[a].add(b)
        nbrs[b].add(a)
    tone = {}
    for i in sorted(ids):
        used = [tone[j] for j in nbrs[i] if j in tone]
        order = VARIANTS[i % 3:] + VARIANTS[:i % 3]
        free = [v for v in order if v not in used]
        tone[i] = free[0] if free else min(order, key=used.count)
    return tone


def panel_fields(P, piece, texel, near_m, log=print):
    ids = np.rint(piece).astype(np.int64)
    N = len(P)
    sub = np.arange(0, N, max(1, N // 400_000))
    tree = cKDTree(P[sub])
    d, j = tree.query(P[sub], k=12, distance_upper_bound=near_m, workers=-1)
    ok = np.isfinite(d)
    a = np.broadcast_to(ids[sub][:, None], j.shape)[ok]
    b = ids[sub][np.where(ok, j, 0)][ok]
    diff = (a != b) & (a > 0) & (b > 0)
    edges = set(zip(np.minimum(a[diff], b[diff]).tolist(), np.maximum(a[diff], b[diff]).tolist()))
    present = np.unique(ids[ids > 0])
    # Mirror twins (left and right of the same panel) share one node, so the kit stays symmetric.
    cnt = np.bincount(ids[sub], minlength=ids.max() + 1).astype(np.float64)
    cen = np.stack([np.bincount(ids[sub], P[sub, c], minlength=len(cnt)) for c in range(3)], 1) / np.maximum(cnt, 1)[:, None]
    node = {i: i for i in present.tolist()}
    ctr = cKDTree(cen[present] * [-1, 1, 1])
    for i in present.tolist():
        if cen[i, 0] > 0:
            continue
        d_, k_ = ctr.query(cen[i], k=4, distance_upper_bound=2 * near_m)
        for dk, kk in zip(np.atleast_1d(d_), np.atleast_1d(k_)):
            j = int(present[kk]) if np.isfinite(dk) else None
            if j is not None and j != i and 0.7 < cnt[j] / max(cnt[i], 1) < 1.43:
                node[i] = node[j] = min(i, j)
                break
    medges = {(min(node[a], node[b]), max(node[a], node[b])) for a, b in edges if node[a] != node[b]}
    tone_node = _colour(sorted(set(node.values())), medges)
    tone_of = {i: tone_node[node[i]] for i in present.tolist()}
    lut = np.zeros(ids.max() + 1, np.float32)
    for i, v in tone_of.items():
        lut[i] = v
    tone = lut[np.maximum(ids, 0)]

    # Seam: distance to the nearest texel of another piece, on the texels that have one within a few mm.
    seam_m = 2.0 * float(np.median(texel))
    dc, jc = tree.query(P, k=8, distance_upper_bound=max(4 * seam_m, 0.003), workers=-1)
    okc = np.isfinite(dc)
    cand = (okc & (ids[sub][np.where(okc, jc, 0)] != ids[:, None])).any(1) & (ids > 0)
    seam = np.zeros(N, np.float32)
    ci = np.flatnonzero(cand)
    if len(ci):
        ctree = cKDTree(P[ci])
        dd, jj = ctree.query(P[ci], k=16, distance_upper_bound=2 * seam_m, workers=-1)
        okd = np.isfinite(dd)
        other = okd & (ids[ci][np.where(okd, jj, 0)] != ids[ci][:, None])
        dist = np.where(other, dd, np.inf).min(1)
        seam[ci] = 1 - smoothstep(0.5 * seam_m, seam_m, dist)
    log(f"[finish] {len(present)} pieces ({len(set(node.values()))} after mirror pairing), {len(edges)} adjacencies, tones "
        f"{ {v: sum(t == v for t in tone_of.values()) for v in VARIANTS} }, seam texels {(seam > 0.5).mean():.2%}")
    return tone, seam
