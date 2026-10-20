"""Per-edge winding crossings of a quad surface against a Lasagna field.

For an edge from vertex p to vertex q the number of papyrus windings it
crosses is the line integral of the winding-position gradient,

    c(p, q) = integral_0^1  rho(x(t)) * n(x(t)) . (q - p)  dt,   x(t) = p + t (q - p),

where rho is the published winding density (windings per voxel) and n the
unit sheet normal, oriented to agree with the surface's own normal so that c
is signed consistently over the surface. On a surface that stays on one
winding every edge is tangent to the sheets and c ~ 0; an edge, or a run of
edges, that climbs to the neighbouring winding integrates to +-1.

The integral is a midpoint rule with ``k`` samples per edge.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .lasagna import LasagnaFields
from .tifxyz import Surface, grid_normals


@dataclass
class EdgeField:
    """Edge quantities on one grid direction.

    Arrays have the surface's shape minus one along the edge axis; edge
    [i, j] joins vertex [i, j] to [i, j + 1] (axis 1) or [i + 1, j] (axis 0).
    """

    axis: int
    crossing: np.ndarray  # signed windings crossed along the edge
    tilt: np.ndarray  # |n . e| / |e|: sine of the edge's angle to the sheet plane
    density: np.ndarray  # mean winding density on the edge (windings / voxel)
    length: np.ndarray  # edge length (voxels)
    valid: np.ndarray  # both vertices valid and every sample carried a prediction


def _edge_pairs(points: np.ndarray, valid: np.ndarray, axis: int):
    if axis == 1:
        return points[:, :-1], points[:, 1:], valid[:, :-1] & valid[:, 1:]
    return points[:-1, :], points[1:, :], valid[:-1, :] & valid[1:, :]


def edge_fields(surface: Surface, fields: LasagnaFields, k: int | None = None,
                batch: int = 2_000_000, normals: np.ndarray | None = None) -> tuple[EdgeField, EdgeField]:
    pts = surface.points
    valid = surface.valid
    if normals is None:
        normals = grid_normals(pts, valid)
    nvalid = np.isfinite(normals).all(axis=-1)
    out = []
    for axis in (1, 0):
        p, q, ev = _edge_pairs(pts, valid & nvalid, axis)
        n_p, n_q, _ = _edge_pairs(normals, nvalid, axis)
        d = q - p
        length = np.linalg.norm(d, axis=-1)
        kk = k or max(2, int(np.ceil(np.nanpercentile(length[ev], 95) / (0.5 * fields.spacing)))) if ev.any() else 2
        orient = n_p + n_q
        orient /= np.linalg.norm(orient, axis=-1, keepdims=True).clip(1e-12)
        idx = np.flatnonzero(ev.ravel())
        cross = np.full(ev.size, np.nan)
        tilt = np.full(ev.size, np.nan)
        dens_mean = np.full(ev.size, np.nan)
        ok_all = np.zeros(ev.size, bool)
        P = p.reshape(-1, 3)
        D = d.reshape(-1, 3)
        O = orient.reshape(-1, 3)
        L = length.reshape(-1)
        ts = (np.arange(kk) + 0.5) / kk
        fields.prefetch((P[idx][:, None, :] + ts[None, :, None] * D[idx][:, None, :]).reshape(-1, 3))
        for s in range(0, len(idx), max(1, batch // kk)):
            ii = idx[s:s + max(1, batch // kk)]
            c_acc = np.zeros(len(ii))
            t_acc = np.zeros(len(ii))
            r_acc = np.zeros(len(ii))
            ok = np.ones(len(ii), bool)
            for j in range(kk):
                t = (j + 0.5) / kk
                x = P[ii] + t * D[ii]
                rho, n, v = fields.sample(x, O[ii])
                dot = np.einsum("ij,ij->i", n, D[ii])
                c_acc += rho * dot
                t_acc += np.abs(dot)
                r_acc += rho
                ok &= v
            cross[ii] = c_acc / kk
            tilt[ii] = t_acc / kk / np.where(L[ii] > 0, L[ii], 1.0)
            dens_mean[ii] = r_acc / kk
            ok_all[ii] = ok
        shp = ev.shape
        okm = ok_all.reshape(shp) & ev
        out.append(EdgeField(axis, cross.reshape(shp), tilt.reshape(shp), dens_mean.reshape(shp), length, okm))
    return out[0], out[1]
