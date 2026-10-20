"""Calibrate the winding-density field against independently traced surfaces.

Two human-traced surfaces that lie on known windings k and k + d give an
external yardstick: walking from a vertex of the first along its surface
normal to the second must cross exactly |d| windings. The integral of the
Lasagna density along that walk, projected on the sheet normal, is the
field's own count of those windings. Its distribution over many vertices
calibrates the field (scale bias, spread) without using the field to define
the truth.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .lasagna import LasagnaFields
from .tifxyz import Surface, grid_normals


def normal_hits(src: Surface, dst: Surface, n_samples: int, max_dist: float, step: float = 2.0,
                hit_tol: float = 12.0, seed: int = 0):
    """Walk from random src vertices along +-normal; return start, hit, and signed distance."""
    rng = np.random.default_rng(seed)
    nrm = grid_normals(src.points, src.valid)
    ok = src.valid & np.isfinite(nrm).all(axis=-1)
    cand = np.argwhere(ok)
    pick = cand[rng.choice(len(cand), size=min(n_samples, len(cand)), replace=False)]
    p = src.points[pick[:, 0], pick[:, 1]]
    m = nrm[pick[:, 0], pick[:, 1]]
    tree = cKDTree(dst.points[dst.valid])
    ts = np.arange(-max_dist, max_dist + step / 2, step)
    D = np.empty((len(p), len(ts)))
    for j, t in enumerate(ts):
        D[:, j], _ = tree.query(p + t * m, k=1, distance_upper_bound=hit_tol)
    # local minima of the distance to dst along the walk = crossings of dst;
    # keep the crossing nearest the start (the first sheet of dst reached)
    left = np.concatenate([np.full((len(p), 1), np.inf), D[:, :-1]], axis=1)
    right = np.concatenate([D[:, 1:], np.full((len(p), 1), np.inf)], axis=1)
    is_min = np.isfinite(D) & (D <= left) & (D <= right)
    absd = np.where(is_min, np.abs(ts)[None, :], np.inf)
    j = absd.argmin(axis=1)
    hit = np.isfinite(absd[np.arange(len(p)), j])
    t_hit = ts[j]
    return pick[hit], p[hit], m[hit], t_hit[hit]


def self_hits(surf: Surface, n_samples: int, max_dist: float, min_grid_sep: int, step: float = 2.0,
              hit_tol: float = 12.0, seed: int = 0):
    """Walk from vertices of a multi-turn surface to its own next turn.

    Crossings of the surface itself whose nearest vertex lies within
    ``min_grid_sep`` grid cells of the start (same local sheet) are ignored,
    so each hit is the adjacent turn of the same traced spiral: exactly one
    winding away if the trace is correct.
    """
    rng = np.random.default_rng(seed)
    nrm = grid_normals(surf.points, surf.valid)
    ok = surf.valid & np.isfinite(nrm).all(axis=-1)
    cand = np.argwhere(ok)
    pick = cand[rng.choice(len(cand), size=min(n_samples, len(cand)), replace=False)]
    p = surf.points[pick[:, 0], pick[:, 1]]
    m = nrm[pick[:, 0], pick[:, 1]]
    vrc = np.argwhere(surf.valid)
    tree = cKDTree(surf.points[surf.valid])
    ts = np.arange(-max_dist, max_dist + step / 2, step)
    D = np.empty((len(p), len(ts)))
    for j, t in enumerate(ts):
        d, idx = tree.query(p + t * m, k=1, distance_upper_bound=hit_tol)
        far = np.zeros(len(p), bool)
        fin = np.isfinite(d)
        g = vrc[idx[fin]]
        far[fin] = np.abs(g - pick[fin]).max(axis=1) > min_grid_sep
        D[:, j] = np.where(far, d, np.inf)
    left = np.concatenate([np.full((len(p), 1), np.inf), D[:, :-1]], axis=1)
    right = np.concatenate([D[:, 1:], np.full((len(p), 1), np.inf)], axis=1)
    is_min = np.isfinite(D) & (D <= left) & (D <= right)
    absd = np.where(is_min, np.abs(ts)[None, :], np.inf)
    j = absd.argmin(axis=1)
    hit = np.isfinite(absd[np.arange(len(p)), j])
    return pick[hit], p[hit], m[hit], ts[j][hit]


def integrate_segment(fields: LasagnaFields, a: np.ndarray, b: np.ndarray, step: float = 4.0):
    """Windings crossed from a to b: integral of density * |n . direction|, midpoint rule."""
    if step <= 0:
        raise ValueError("integration step must be positive")
    if len(a) == 0:
        return np.empty(0), np.empty(0), np.empty(0, dtype=bool)
    d = b - a
    length = np.linalg.norm(d, axis=1)
    k = max(2, int(np.ceil(length.max() / step)))
    ts = (np.arange(k) + 0.5) / k
    pts = (a[:, None, :] + ts[None, :, None] * d[:, None, :]).reshape(-1, 3)
    fields.prefetch(pts)
    rho, n, valid = fields.sample(pts)
    rho = rho.reshape(len(a), k)
    n = n.reshape(len(a), k, 3)
    valid = valid.reshape(len(a), k).all(axis=1)
    e = d / np.where(length > 0, length, 1.0)[:, None]
    proj = np.abs(np.einsum("ikj,ij->ik", n, e))
    windings = (rho * proj).mean(axis=1) * length
    raw = rho.mean(axis=1) * length
    return windings, raw, valid
