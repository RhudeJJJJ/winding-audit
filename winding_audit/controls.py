"""Planted sheet switches built from two human-traced adjacent windings.

A block of vertices of surface A is moved onto surface B, the human-traced
neighbouring winding, by walking each vertex along A's normal to its first
crossing of B. The result is a real-data surface that is known, without
using the Lasagna field, to sit one winding off inside the block. ``ramp``
blends the move over a band of that many vertices so the switch is gradual
instead of a single cliff edge.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree

from .tifxyz import Surface, grid_normals


def hits_for_vertices(src: Surface, dst: Surface, rc: np.ndarray, max_dist: float, step: float = 1.0,
                      hit_tol: float = 12.0):
    nrm = grid_normals(src.points, src.valid)
    p = src.points[rc[:, 0], rc[:, 1]]
    m = nrm[rc[:, 0], rc[:, 1]]
    ok = np.isfinite(p).all(axis=1) & np.isfinite(m).all(axis=1)
    tree = cKDTree(dst.points[dst.valid])
    ts = np.arange(-max_dist, max_dist + step / 2, step)
    D = np.full((len(p), len(ts)), np.inf)
    for j, t in enumerate(ts):
        q = p + t * m
        q[~ok] = 0
        D[:, j], _ = tree.query(q, k=1, distance_upper_bound=hit_tol)
    D[~ok] = np.inf
    left = np.concatenate([np.full((len(p), 1), np.inf), D[:, :-1]], axis=1)
    right = np.concatenate([D[:, 1:], np.full((len(p), 1), np.inf)], axis=1)
    is_min = np.isfinite(D) & (D <= left) & (D <= right)
    absd = np.where(is_min, np.abs(ts)[None, :], np.inf)
    j = absd.argmin(axis=1)
    hit = np.isfinite(absd[np.arange(len(p)), j])
    t = np.where(hit, ts[j], np.nan)
    return p + t[:, None] * m, hit, t


def plant_block(src: Surface, dst: Surface, r0: int, r1: int, c0: int, c1: int, ramp: int = 0,
                max_dist: float = 160.0, side: int | None = None):
    """Surface A with the block [r0:r1, c0:c1] moved onto B (optionally ramped)."""
    rows, cols = src.shape
    rr0, rr1 = max(0, r0 - ramp), min(rows, r1 + ramp)
    cc0, cc1 = max(0, c0 - ramp), min(cols, c1 + ramp)
    rc = np.argwhere(src.valid[rr0:rr1, cc0:cc1]) + [rr0, cc0]
    hitp, hit, t = hits_for_vertices(src, dst, rc, max_dist=max_dist)
    if side is not None:
        hit &= np.sign(t) == side
    core = np.zeros(src.shape, bool)
    core[r0:r1, c0:c1] = True
    if ramp > 0:
        dist = distance_transform_edt(~core)
        alpha_full = np.clip(1.0 - dist / float(ramp + 1), 0.0, 1.0)
    else:
        alpha_full = core.astype(float)
    pts = src.points.copy()
    valid = src.valid.copy()
    alpha = alpha_full[rc[:, 0], rc[:, 1]]
    moved = alpha > 0
    # a vertex that should move but has no crossing of B is dropped rather
    # than left on A, so the truth map stays exact
    drop = moved & ~hit
    keep = moved & hit
    pts[rc[keep, 0], rc[keep, 1]] = (1 - alpha[keep, None]) * pts[rc[keep, 0], rc[keep, 1]] + alpha[keep, None] * hitp[keep]
    valid[rc[drop, 0], rc[drop, 1]] = False
    pts[~valid] = np.nan
    truth = np.zeros(src.shape)
    truth[rc[keep, 0], rc[keep, 1]] = alpha[keep] * np.sign(t[keep])
    truth[~valid] = np.nan
    return Surface(points=pts, valid=valid, meta=dict(src.meta), source=src.source + "#planted"), truth
