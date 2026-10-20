"""PHerc0139: w046 coincides with w045 over most of its area (results/duplicate_w045_w046.json, figures).

Maps, for every 4th vertex of w046, the distance to the nearest vertex of w045
and of w047, and renders the CT along w045's normals (row 600, columns
760-840) with the crossings of w046 and w047 marked.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from common import open_catalog, per_winding, surface_url, volume_url
from winding_audit.ct import CTVolume, straightened_section
from winding_audit.tifxyz import grid_normals, load_tifxyz

ap = argparse.ArgumentParser()
ap.add_argument("--cache", default=".winding_audit_cache")
a = ap.parse_args()
cache, cat = open_catalog(a.cache)
sid, vid = "PHerc0139", "20260102150214"
W = per_winding(cat, sid, vid)
S = {k: load_tifxyz(surface_url(cat, sid, W[k][0], vid), cache) for k in (45, 46, 47)}
sub = 4
A = S[46]
Av, Ap = A.valid[::sub, ::sub], A.points[::sub, ::sub]
dist = {}
for k in (45, 47):
    d = np.full(Av.shape, np.nan)
    q, _ = cKDTree(S[k].points[S[k].valid]).query(Ap[Av], k=1, workers=-1)
    d[Av] = q
    dist[k] = d
stats = {
    "segments": {k: W[k][1] for k in (45, 46, 47)},
    "w046_within_12vox_of_w045": float(np.mean(dist[45][Av] < 12)),
    "w046_30vox_or_more_from_w045": float(np.mean(dist[45][Av] >= 30)),
    "median_distance_w046_to_w045_vox": float(np.median(dist[45][Av])),
}
print(json.dumps(stats, indent=1))
Path("results").mkdir(exist_ok=True)
Path("results/duplicate_w045_w046.json").write_text(json.dumps(stats, indent=1))
np.savez_compressed("results/duplicate_w045_w046_maps.npz", d45=dist[45].astype(np.float32), d47=dist[47].astype(np.float32))

ct = CTVolume(volume_url(cat, sid, vid), level=0, cache=cache)
N = grid_normals(S[45].points, S[45].valid)
r, c = 600, 800
line, nl = S[45].points[r, c - 40:c + 41], N[r, c - 40:c + 41]
img, frac, ts = straightened_section(ct, line, nl, 180, depth_step=1.0, sub=2)
cross = {}
for k in (46, 47):
    tree = cKDTree(S[k].points[S[k].valid])
    xs, ys = [], []
    for j in range(0, len(line), 2):
        if not (np.isfinite(line[j]).all() and np.isfinite(nl[j]).all()):
            continue
        q = line[j][None, :] + ts[:, None] * nl[j][None, :]
        dd, _ = tree.query(q, k=1)
        i = int(np.argmin(dd))
        if dd[i] < 10:
            xs.append(c - 40 + j)
            ys.append(float(ts[i]))
    cross[k] = (xs, ys)
np.savez_compressed("results/duplicate_ct_section.npz", img=img, frac=frac, ts=ts, x0=c - 40,
                    x46=np.array(cross[46][0]), y46=np.array(cross[46][1]),
                    x47=np.array(cross[47][0]), y47=np.array(cross[47][1]))
