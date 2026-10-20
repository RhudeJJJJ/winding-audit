"""One-call audit of a tifxyz surface against a Lasagna field.

Produces, for every vertex, the quantities a reviewer needs to find where a
traced surface leaves its winding:

* ``tilt_deg``   angle between the surface and the predicted sheet plane;
* ``cos``        predicted winding phase at the vertex (0.5 + 0.5 cos 2 pi F);
* ``step_h``     signed net windings crossed within +-h edges along the grid
                 line (rows or columns, whichever is stronger), for several h;

and a list of candidate sites: connected groups of vertices whose net
crossing saturates near one calibrated winding across scales (a step), and
separately groups where the crossing keeps growing with the window
(field/surface disagreement that is not a single switch).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import label

from .edges import EdgeField, edge_fields
from .lasagna import LasagnaFields
from .steps import StepMap, step_map
from .tifxyz import Surface, grid_normals


@dataclass
class AuditParams:
    # thresholds in field windings; one traced winding reads 1.06-1.16 at the sheet
    # spacings of the 2.4 um scans (README, Finding 5)
    scales: tuple = (2, 4, 8, 16)
    step_lo: float = 0.6  # |D_small| must reach this
    step_hi: float = 1.8  # |D_large| must stay below this (more is not one winding)
    plateau_tol: float = 0.35  # |D_large - D_small| for a saturated step
    implausible: float = 2.0  # |D_large| above this: not a single switch
    min_site_vertices: int = 12


@dataclass
class AuditResult:
    surface: Surface
    eh: EdgeField
    ev: EdgeField
    steps: StepMap
    tilt_deg: np.ndarray
    cos: np.ndarray | None
    step_mask: np.ndarray
    disagree_mask: np.ndarray
    sites: list = field(default_factory=list)
    disagreements: list = field(default_factory=list)
    params: AuditParams = field(default_factory=AuditParams)


def _vertex_tilt(eh: EdgeField, ev: EdgeField, shape) -> np.ndarray:
    t = np.full(shape, np.nan)
    th = np.degrees(np.arcsin(np.clip(np.nan_to_num(eh.tilt, nan=0.0), 0, 1)))
    tv = np.degrees(np.arcsin(np.clip(np.nan_to_num(ev.tilt, nan=0.0), 0, 1)))
    t[:, :-1] = np.where(eh.valid, th, np.nan)
    t[:, 1:] = np.fmax(t[:, 1:], np.where(eh.valid, th, np.nan))
    t[:-1, :] = np.fmax(t[:-1, :], np.where(ev.valid, tv, np.nan))
    t[1:, :] = np.fmax(t[1:, :], np.where(ev.valid, tv, np.nan))
    return t


def _small_index(steps: StepMap) -> int:
    """Scale used as the 'small' window: the second when three or more are computed."""
    return 1 if len(steps.scales) > 2 else 0


def _groups(mask: np.ndarray, surf: Surface, steps: StepMap, min_vertices: int, kind: str):
    lab, n = label(mask, structure=np.ones((3, 3)))
    out = []
    ks = _small_index(steps)
    small = steps.step[ks]
    large = steps.step[-1]
    for i in range(1, n + 1):
        cells = lab == i
        size = int(cells.sum())
        if size < min_vertices:
            continue
        rr, cc = np.nonzero(cells)
        pts = surf.points[rr, cc]
        ok = np.isfinite(pts).all(axis=1)
        k = int(np.argmax(np.abs(np.nan_to_num(small[rr, cc]))))
        out.append({
            "kind": kind,
            "vertices": size,
            "row_range": [int(rr.min()), int(rr.max())],
            "col_range": [int(cc.min()), int(cc.max())],
            "peak_rc": [int(rr[k]), int(cc[k])],
            "peak_xyz": [float(v) for v in surf.points[rr[k], cc[k]]],
            "centroid_xyz": [float(v) for v in pts[ok].mean(axis=0)] if ok.any() else None,
            "median_step_small": float(np.nanmedian(small[rr, cc])),
            "median_step_large": float(np.nanmedian(large[rr, cc])),
            "direction": "rows" if np.nanmedian(steps.direction[ks][rr, cc]) < 0.5 else "columns",
        })
    out.sort(key=lambda d: -d["vertices"])
    for j, d in enumerate(out):
        d["id"] = f"{kind}-{j:03d}"
    return out


def audit_surface(surf: Surface, fields: LasagnaFields, params: AuditParams | None = None,
                  with_cos: bool = True) -> AuditResult:
    params = params or AuditParams()
    normals = grid_normals(surf.points, surf.valid)
    eh, ev = edge_fields(surf, fields, normals=normals)
    sm = step_map(eh, ev, scales=params.scales)
    tilt = _vertex_tilt(eh, ev, surf.shape)
    cos = None
    if with_cos and "cos" in fields.channels:
        pts = np.where(surf.valid[..., None], surf.points, np.nan).reshape(-1, 3)
        fields.prefetch(pts[np.isfinite(pts).all(axis=1)], channels=("cos",))
        c, inside = fields.sample_channel("cos", pts)
        cos = np.where(inside, c / 255.0, np.nan).reshape(surf.shape)
    s_small = sm.step[_small_index(sm)]
    s_large = sm.step[-1]
    ok = np.isfinite(s_small) & np.isfinite(s_large)
    a_small, a_large = np.abs(s_small), np.abs(s_large)
    step_mask = (ok & (a_small >= params.step_lo) & (a_large <= params.step_hi)
                 & (np.abs(s_large - s_small) <= params.plateau_tol) & (np.sign(s_small) == np.sign(s_large)))
    disagree_mask = np.isfinite(s_large) & (a_large > params.implausible)
    sites = _groups(step_mask, surf, sm, params.min_site_vertices, "step")
    dis = _groups(disagree_mask, surf, sm, params.min_site_vertices, "disagreement")
    return AuditResult(surf, eh, ev, sm, tilt, cos, step_mask, disagree_mask, sites, dis, params)


def summarize(res: AuditResult) -> dict:
    v = res.surface.valid
    nv = int(v.sum())
    def frac(m):
        return float((m & v).sum() / max(1, nv))
    tilt = res.tilt_deg[v & np.isfinite(res.tilt_deg)]
    out = {
        "shape": list(res.surface.shape),
        "valid_vertices": nv,
        "edges_scored": int(res.eh.valid.sum() + res.ev.valid.sum()),
        "median_edge_length_vox": float(np.nanmedian(np.concatenate([res.eh.length[res.eh.valid], res.ev.length[res.ev.valid]]))) if nv else None,
        "median_winding_spacing_vox": float(1.0 / np.nanmedian(np.concatenate([res.eh.density[res.eh.valid], res.ev.density[res.ev.valid]]))) if nv else None,
        "tilt_deg_percentiles": {str(p): float(np.percentile(tilt, p)) for p in (50, 90, 99)} if tilt.size else None,
        "step_vertex_fraction": frac(res.step_mask),
        "disagreement_vertex_fraction": frac(res.disagree_mask),
        "step_sites": len(res.sites),
        "disagreement_zones": len(res.disagreements),
        "params": res.params.__dict__,
    }
    if res.cos is not None:
        c = res.cos[v & np.isfinite(res.cos)]
        out["cos_percentiles"] = {str(p): float(np.percentile(c, p)) for p in (5, 25, 50, 75)} if c.size else None
    return out
