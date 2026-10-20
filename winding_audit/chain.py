"""Check that a set of per-winding segments steps out one winding at a time.

For consecutive segments k -> k+1 two independent measures should agree on a
step of one winding: the field's winding count along k's surface normals to
the first crossing of k+1 (see calibrate.py), and the geometric distance of
that crossing compared with the chain's typical sheet spacing. Each pair gets:

* ``windings``  median field count (with quartiles and sample count);
* ``distance``  median normal distance to k+1, also relative to the median
                over the chain (``spacing_ratio``);
* ``coincide``  fraction of k+1's vertices within ``coincide_vox`` of k;
* ``verdict``   "duplicate" (k+1 mostly lies on k, or field and geometry both
                put it less than half a winding out), "gap" (both put it about
                two windings out: a winding is missing), "overlap" (a
                substantial part of k+1 lies on k), "check" (field and
                geometry disagree: the field is not reliable there, or the
                spacing is unusual; look at the CT before calling it an
                error), or "ok".
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .calibrate import integrate_segment, normal_hits
from .lasagna import LasagnaFields
from .tifxyz import Surface


def pair_check(a: Surface, b: Surface, fields: LasagnaFields, samples: int = 2000, max_dist: float = 220.0,
               coincide_vox: float = 12.0, seed: int = 0, one_winding: float = 1.0,
               keep_samples: bool = False) -> dict:
    tree = cKDTree(a.points[a.valid])
    pb = b.points[b.valid]
    if not a.valid.any() or not len(pb):
        return {"n": 0, "coincide_fraction": 0.0, "verdict": "check", "reason": "empty surface"}
    step = max(1, int(np.ceil(len(pb) / 200_000)))
    d, _ = tree.query(pb[::step], k=1, workers=-1)
    coincide = float(np.mean(d < coincide_vox))
    pick, p, m, th = normal_hits(a, b, samples, max_dist, seed=seed)
    out = {"coincide_fraction": coincide, "coincide_sample_count": int(len(d)),
           "coincide_threshold_vox": coincide_vox,
           "fraction_basis": "directional sampled grid vertices of b near vertices of a; not area",
           "n": 0, "n_normal_hits": int(len(th))}
    if len(th):
        side = np.sign(np.median(th))
        keep = np.sign(th) == side
        p, m, th = p[keep], m[keep], th[keep]
        out["n_same_side_hits"] = int(len(th))
        if len(th):
            # Geometry must not be conditioned on the model having a valid
            # prediction: masked fields otherwise bias both diagnostics.
            out["distance_median_vox"] = float(np.median(np.abs(th)))
        w, _, ok = integrate_segment(fields, p, p + th[:, None] * m)
        w, dist = w[ok], np.abs(th[ok])
        if len(w):
            q = np.percentile(w, [25, 50, 75])
            out.update(n=int(len(w)), windings_median=float(q[1]), windings_q25=float(q[0]), windings_q75=float(q[2]),
                       field_valid_distance_median_vox=float(np.median(dist)))
            if keep_samples:
                out["samples"] = {"distance": dist.round(1).tolist(), "windings": w.round(4).tolist()}
    out["verdict"] = verdict(out, one_winding)
    return out


def verdict(r: dict, one_winding: float = 1.0, spacing: float | None = None) -> str:
    """Verdict of one pair; without ``spacing`` only the field and the coincidence are used."""
    med = r.get("windings_median")
    field = None if med is None else med / one_winding
    dist = r.get("distance_median_vox")
    geo = None if (spacing is None or dist is None) else dist / spacing
    coincide = r["coincide_fraction"]
    if coincide >= 0.5:
        return "duplicate"
    if r.get("n", 0) < 20:
        return "overlap" if coincide >= 0.15 else "check"
    if field is None:
        return "overlap" if coincide >= 0.15 else "check"
    field_low, field_high = field < 0.5, field >= 1.6
    if geo is None:
        if field_low or field_high:
            return "check"
    else:
        geo_low, geo_high = geo < 0.5, geo >= 1.5
        if field_low and geo_low:
            return "duplicate"
        if field_high and geo_high:
            return "gap"
        if field_low or field_high or geo_low or geo_high:
            return "check"
    return "overlap" if coincide >= 0.15 else "ok"


def chain_verdicts(pairs: dict, one_winding: float = 1.0, spacing: float | None = None) -> float:
    """Re-judge every pair of a chain with its typical spacing; returns the spacing used.

    ``spacing`` defaults to the median over the chain's pairs of the median
    normal distance, which is robust to a few duplicates and gaps.
    """
    if spacing is None:
        d = [r["distance_median_vox"] for r in pairs.values() if "distance_median_vox" in r]
        spacing = float(np.median(d)) if d else None
    for r in pairs.values():
        if spacing and "distance_median_vox" in r:
            r["spacing_ratio"] = r["distance_median_vox"] / spacing
        r["verdict"] = verdict(r, one_winding, spacing)
    return spacing
