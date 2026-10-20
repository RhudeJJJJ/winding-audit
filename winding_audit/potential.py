"""Integrate per-edge winding crossings into a relative winding map.

The crossings c_e are differences of one scalar (the winding position) along
grid edges, so a least-squares potential w with w[b] - w[a] ~ c_e over all
valid edges recovers, up to a constant per connected component, which
winding each vertex lies on relative to the rest of the surface. A surface
that never leaves its winding has w ~ constant; a sheet switch appears as a
step of +-1 across the switch.

The residual r_e = c_e - (w[b] - w[a]) measures where the crossings are not
integrable (a non-zero sum around a grid cell), which flags field or surface
inconsistency rather than a clean step.

Not used by the audit. On real traces the integrated potential drifts by
several windings across a surface (1st-99th percentile -3.2 to +2.3 windings
on PHerc0814 46527_2um_try2, two thirds of the vertices off the zero level),
because small per-edge field errors accumulate along long paths. The audit
therefore counts crossings in short windows only (steps.py). Kept for
reference; needs the optional dependency pyamg.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.ndimage import label

from .edges import EdgeField


def _incidence(shape, eh: EdgeField, ev: EdgeField, weights=None):
    rows, cols = shape
    vid = np.arange(rows * cols).reshape(shape)
    a_h, b_h = vid[:, :-1][eh.valid], vid[:, 1:][eh.valid]
    a_v, b_v = vid[:-1, :][ev.valid], vid[1:, :][ev.valid]
    a = np.concatenate([a_h, a_v])
    b = np.concatenate([b_h, b_v])
    c = np.concatenate([eh.crossing[eh.valid], ev.crossing[ev.valid]])
    w = np.ones_like(c) if weights is None else weights
    return a, b, c, w


def solve_potential(shape, eh: EdgeField, ev: EdgeField, tol: float = 1e-8):
    """Least-squares winding potential (NaN off the edge graph) and edge residuals."""
    a, b, c, w = _incidence(shape, eh, ev)
    n = shape[0] * shape[1]
    m = len(c)
    B = sp.csr_matrix(
        (np.concatenate([-w, w]), (np.concatenate([np.arange(m), np.arange(m)]), np.concatenate([a, b]))),
        shape=(m, n),
    )
    used = np.zeros(n, bool)
    used[a] = True
    used[b] = True
    idx = np.flatnonzero(used)
    Bu = B[:, idx]
    L = (Bu.T @ Bu).tocsr()
    rhs = Bu.T @ (w * c)
    # connected components of the edge graph: pin each one's mean to zero
    adj = (L != 0).astype(np.int8)
    ncomp, comp = sp.csgraph.connected_components(adj, directed=False)
    # a tiny Tikhonov term makes L definite without moving the solution by
    # more than ~eps * |w|; each component's mean is removed afterwards
    eps = 1e-9 * max(1.0, L.diagonal().mean())
    Lr = L + eps * sp.identity(L.shape[0], format="csr")
    try:
        import pyamg

        ml = pyamg.smoothed_aggregation_solver(Lr, symmetry="symmetric")
        x = ml.solve(rhs, tol=tol, accel="cg", maxiter=500)
    except ImportError:  # pragma: no cover - fallback
        x, _ = sp.linalg.cg(Lr, rhs, rtol=tol, maxiter=5000)
    for k in range(ncomp):
        sel = comp == k
        x[sel] -= np.median(x[sel])
    pot = np.full(n, np.nan)
    pot[idx] = x
    pot = pot.reshape(shape)
    full = np.full(n, np.nan)
    full[idx] = x
    resid = c - (full[b] - full[a])
    comp_map = np.full(n, -1)
    comp_map[idx] = comp
    return pot, resid, comp_map.reshape(shape)


def winding_levels(pot: np.ndarray, valid: np.ndarray, comp_map: np.ndarray, min_vertices: int = 0):
    """Integer winding offset per vertex relative to each component's dominant level.

    Returns the offset map (0 = dominant winding, NaN off-surface) and a list
    of connected off-winding regions with their size and mean potential.
    """
    lev = np.full(pot.shape, np.nan)
    regions = []
    for k in np.unique(comp_map[comp_map >= 0]):
        sel = (comp_map == k) & np.isfinite(pot)
        r = np.rint(pot[sel])
        vals, counts = np.unique(r, return_counts=True)
        dom = vals[np.argmax(counts)]
        lev[sel] = np.rint(pot[sel] - dom)
    off = np.isfinite(lev) & (lev != 0)
    lab, nlab = label(off)
    for i in range(1, nlab + 1):
        cells = lab == i
        size = int(cells.sum())
        if size < min_vertices:
            continue
        rr, cc = np.nonzero(cells)
        regions.append({
            "label": i,
            "vertices": size,
            "offset": float(np.median(lev[cells])),
            "mean_potential": float(np.nanmean(pot[cells])),
            "row_range": [int(rr.min()), int(rr.max())],
            "col_range": [int(cc.min()), int(cc.max())],
            "centroid_rc": [float(rr.mean()), float(cc.mean())],
        })
    regions.sort(key=lambda r: -r["vertices"])
    return lev, regions
