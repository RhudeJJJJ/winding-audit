"""Field winding count between adjacent turns of multi-turn traces (results/selfcalib_PHercParis4.json).

Each PHercParis4 'wNNN-MMM' surface is one traced spiral over several
windings, so walking along a vertex normal to the next crossing of the same
surface (ignoring the vertex's own neighbourhood) crosses exactly one winding.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from common import manifest, open_catalog, surface_url
from winding_audit.calibrate import integrate_segment, self_hits
from winding_audit.lasagna import LasagnaFields
from winding_audit.tifxyz import load_tifxyz

ap = argparse.ArgumentParser()
ap.add_argument("--cache", default=".winding_audit_cache")
ap.add_argument("segments", nargs="*", default=["20260623144957", "20260623151729"])
a = ap.parse_args()
cache, cat = open_catalog(a.cache)
sid, vid = "PHercParis4", "20260411134726"
fields = LasagnaFields(manifest(cat, sid, vid), cache)
out = {}
for segid in a.segments:
    s = load_tifxyz(surface_url(cat, sid, segid, vid), cache)
    pick, p, m, th = self_hits(s, n_samples=3000, max_dist=200, min_grid_sep=150, seed=3)
    w, _, ok = integrate_segment(fields, p, p + th[:, None] * m)
    w, d = w[ok], np.abs(th[ok])
    out[segid] = {"long_id": cat["samples"][sid]["segments"][segid]["long_id"], "n": int(len(w)),
                  "median": float(np.median(w)), "quartiles": [float(np.percentile(w, q)) for q in (25, 75)],
                  "distance_median_vox": float(np.median(d)), "distance": d.round(1).tolist(), "windings": w.round(4).tolist()}
    print(segid, out[segid]["long_id"], "n", out[segid]["n"], "median", round(out[segid]["median"], 3))
Path("results").mkdir(exist_ok=True)
Path("results/selfcalib_PHercParis4.json").write_text(json.dumps(out))
