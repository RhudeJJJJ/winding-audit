"""PHerc1667: the outermost traced windings lie outside the mask of the masked volume 20251217075048.

Renders two axial CT slices (level 4) of the masked volume and the vertices
of the per-winding segments w040 and w041 near each slice
(results/masked_1667_axial.npz), and reports, per slice, the fraction of
those vertices that fall on masked (zero) voxels and how far outside the
mask they lie (results/masked_1667.json).
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt

from common import open_catalog, per_winding, surface_url, volume_url
from winding_audit.ct import CTVolume
from winding_audit.tifxyz import load_tifxyz

ap = argparse.ArgumentParser()
ap.add_argument("--cache", default=".winding_audit_cache")
a = ap.parse_args()
cache, cat = open_catalog(a.cache)
sid, vid = "PHerc1667", "20251217075048"
W = per_winding(cat, sid, vid)
ct = CTVolume(volume_url(cat, sid, vid), level=4, cache=cache)
ct0 = CTVolume(volume_url(cat, sid, vid), level=2, cache=cache)
S = {k: load_tifxyz(surface_url(cat, sid, W[k][0], vid), cache) for k in (40, 41)}
step = 16.0
xs = np.arange(0, 15229, step)
arrays, stats = {}, {}
for zc in (15000.0, 31000.0):
    Y, X = np.meshgrid(xs, xs, indexing="ij")
    img = ct.sample(np.stack([X.ravel(), Y.ravel(), np.full(X.size, zc)], axis=1)).reshape(Y.shape)
    arrays[f"img_{zc:.0f}"] = img.astype(np.float32)
    # distance (voxels) from every pixel of the slice to the nearest unmasked one
    outside = distance_transform_edt(~(img > 1.0)) * step
    for k, s in S.items():
        P = s.points[s.valid]
        P = P[np.abs(P[:, 2] - zc) < 10][::5]
        arrays[f"w{k:03d}_{zc:.0f}"] = P[:, :2].astype(np.float32)
        v = ct0.sample(P)
        ix = np.clip(np.round(P[:, 0] / step).astype(int), 0, len(xs) - 1)
        iy = np.clip(np.round(P[:, 1] / step).astype(int), 0, len(xs) - 1)
        dd = outside[iy, ix]
        stats[f"w{k:03d} at z={zc:.0f}"] = {"vertices": int(len(P)), "on_masked_voxels": float(np.mean(v <= 1.0)),
                                            "distance_outside_mask_median_vox": float(np.median(dd[dd > 0])) if (dd > 0).any() else 0.0,
                                            "distance_outside_mask_max_vox": float(dd.max())}
arrays["xs"] = xs
print(json.dumps(stats, indent=1))
Path("results").mkdir(exist_ok=True)
np.savez_compressed("results/masked_1667_axial.npz", **arrays)
Path("results/masked_1667.json").write_text(json.dumps({"volume": vid, "segments": {k: W[k][1] for k in S}, "slices": stats}, indent=1))
