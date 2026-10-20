"""Geometry-only, directional overlap and threshold sensitivity on frozen URLs.

Counts every fourth row/column vertex. No model-field selection or area claim.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from winding_audit.tifxyz import load_tifxyz
from winding_audit.zarr_io import ChunkCache

ap = argparse.ArgumentParser()
ap.add_argument('--cache', default='.winding_audit_cache')
ap.add_argument('--out', default='results/review/overlap_thresholds.json')
a = ap.parse_args()
snapshot = json.loads(Path('results/review/source_snapshot.json').read_text())
source = next(x for x in snapshot['sources'] if x['sample'] == 'PHerc0139')['per_winding']
cache = ChunkCache(a.cache)
results = {}
for ka, kb in [(41, 42), (45, 46)]:
    surfaces = {k: load_tifxyz(source[str(k)]['surface_url'], cache) for k in (ka, kb)}
    for kfrom, kto in [(kb, ka), (ka, kb)]:
        src, dst = surfaces[kfrom], surfaces[kto]
        p = src.points[::4, ::4]; valid = src.valid[::4, ::4]
        d, _ = cKDTree(dst.points[dst.valid]).query(p[valid], workers=-1)
        r = {'source': source[str(kfrom)]['surface_url'], 'target': source[str(kto)]['surface_url'],
             'basis': 'valid vertices on every fourth grid row and column; nearest full-resolution target vertex; not area',
             'samples': int(len(d)),
             'distance_percentiles_vox': {str(q): float(np.percentile(d, q)) for q in [10, 25, 50, 75, 90]},
             'fraction_below_vox': {str(t): float(np.mean(d < t)) for t in [6, 8, 10, 12, 16, 20, 30]}}
        results[f'w{kfrom:03d}->w{kto:03d}'] = r
        print(kfrom, kto, r['samples'], r['fraction_below_vox'], flush=True)
        p = Path(a.out); p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(results, indent=2, allow_nan=False))
