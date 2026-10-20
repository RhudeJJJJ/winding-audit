"""Does a surface lie on papyrus in the volume it is attributed to?

A traced recto surface sits on papyrus: the CT at its vertices is material,
and the mean intensity profile along the surface normal peaks near the
surface. A surface registered to the wrong volume, or carried over from an
older reconstruction, lands off the volume, on empty or masked voxels, or
inside the scroll between layers, and shows a flat profile. Sampled vertices
are classified as outside the volume (no CT there), empty (CT at or below
``empty_level``: background or masked away), or on material.

Given the volume's Lasagna fields, the check also reports the angle between
the surface normal and the predicted sheet normal at the sampled vertices: a
few degrees for a surface that follows the sheets of this volume, tens of
degrees for one that was traced in another frame.
"""

from __future__ import annotations

import numpy as np

from .ct import CTVolume
from .lasagna import LasagnaFields
from .tifxyz import Surface, grid_normals


def placement_check(surf: Surface, ct: CTVolume, volume_shape_zyx, samples: int = 1500, half_depth: float = 12.0,
                    seed: int = 0, empty_level: float = 1.0, fields: LasagnaFields | None = None,
                    peak_tolerance: float = 3.0) -> dict:
    if samples <= 0 or half_depth < 2 or peak_tolerance < 0:
        raise ValueError("samples must be positive, half_depth >= 2 and peak_tolerance >= 0")
    rng = np.random.default_rng(seed)
    N = grid_normals(surf.points, surf.valid)
    ok = surf.valid & np.isfinite(N).all(axis=-1)
    rc = np.argwhere(ok)
    if not len(rc):
        return {"samples": 0, "verdict": "insufficient_data", "reason": "no vertices with usable normals"}
    rc = rc[rng.choice(len(rc), size=min(samples, len(rc)), replace=False)]
    p = surf.points[rc[:, 0], rc[:, 1]]
    m = N[rc[:, 0], rc[:, 1]]
    ts = np.arange(-int(np.ceil(half_depth)), int(np.ceil(half_depth)) + 1, dtype=float)
    prof = ct.sample((p[:, None, :] + ts[None, :, None] * m[:, None, :]).reshape(-1, 3)).reshape(len(p), len(ts))
    at0 = prof[:, len(ts) // 2]
    inside = np.isfinite(at0)
    outside_fraction = float(1.0 - inside.mean())
    empty_fraction = float(np.mean(at0[inside] <= empty_level)) if inside.any() else float("nan")
    # the layer profile is taken over vertices on material only
    mat = inside & (at0 > empty_level)
    if mat.any():
        mean_prof = np.nanmean(prof[mat], axis=0)
        edge = np.nanmean(mean_prof[[0, 1, -2, -1]])
    else:
        mean_prof, edge = np.full(len(ts), np.nan), float("nan")
    peak_ratio = float(np.nanmax(mean_prof) / edge) if np.isfinite(edge) and edge > 0 else float("nan")
    peak_offset = float(ts[int(np.nanargmax(mean_prof))]) if np.isfinite(mean_prof).any() else float("nan")
    central = np.abs(ts) <= peak_tolerance
    center_ratio = float(np.nanmax(mean_prof[central]) / edge) if np.isfinite(edge) and edge > 0 else float("nan")
    P = surf.points[surf.valid]
    lo, hi = np.nanmin(P, axis=0), np.nanmax(P, axis=0)
    Z, Y, X = volume_shape_zyx
    bbox_outside = bool(lo[0] < 0 or lo[1] < 0 or lo[2] < 0 or hi[0] > X - 1 or hi[1] > Y - 1 or hi[2] > Z - 1)
    if outside_fraction > 0.5:
        verdict = "outside_volume"
    elif empty_fraction > 0.5:
        verdict = "on_empty_voxels"
    elif not peak_ratio >= 1.04:
        verdict = "no_layer_at_surface"
    elif abs(peak_offset) > peak_tolerance:
        verdict = "layer_offset"
    else:
        verdict = "on_papyrus"
    tilt = float("nan")
    if fields is not None:
        fields.prefetch(p)
        _, n, fvalid = fields.sample(p)
        mu = m / np.linalg.norm(m, axis=1, keepdims=True)
        cosang = np.clip(np.abs(np.einsum("ij,ij->i", n, mu)), 0.0, 1.0)
        if fvalid.any():
            tilt = float(np.degrees(np.median(np.arccos(cosang[fvalid]))))
    ct_verdict = verdict
    # Intensity alone can pass even if a surface cuts across the layers.
    # Keep both observations rather than upgrading that pass to ground truth.
    if verdict == "on_papyrus" and np.isfinite(tilt) and tilt > 20.0:
        verdict = "check_orientation"
    return {
        "samples": int(len(p)),
        "field_tilt_median_deg": tilt,
        "ct_at_surface_median": float(np.nanmedian(at0)) if inside.any() else float("nan"),
        "outside_fraction": outside_fraction,
        "empty_fraction": empty_fraction,
        "empty_fraction_of_inside": empty_fraction,
        "empty_fraction_of_all": float(np.mean(inside & (at0 <= empty_level))),
        "inside_samples": int(inside.sum()),
        "material_samples": int(mat.sum()),
        "field_valid_samples": int(fvalid.sum()) if fields is not None else 0,
        "profile_peak_offset_vox": peak_offset,
        "profile_peak_to_edge": peak_ratio,
        "profile_center_to_edge": center_ratio,
        "peak_tolerance_vox": peak_tolerance,
        "profile_offsets_vox": ts.tolist(),
        "profile_mean_on_material": mean_prof.tolist(),
        "fraction_basis": "uniform sample of valid grid vertices with usable normals; not surface area",
        "ct_screen_verdict": ct_verdict,
        "interpretation": "heuristic screening result, not an anatomical ground-truth label",
        "bbox_xyz": [[float(v) for v in lo], [float(v) for v in hi]],
        "volume_shape_zyx": [int(v) for v in volume_shape_zyx],
        "bbox_outside_volume": bbox_outside,
        "verdict": verdict,
    }
