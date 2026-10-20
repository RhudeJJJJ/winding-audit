"""Winding-chain check of a per-winding segment set (results/chain_<sample>.json).

The per-vertex field counts and normal distances of every pair go to
results/chain_<sample>_samples.json (used for the calibration figure).

    python scripts/reproduce_chain.py PHerc0139 20260102150214
    python scripts/reproduce_chain.py PHerc1667 20251217075048
"""

import argparse
import json
import time
from pathlib import Path

from common import manifest, open_catalog, per_winding, surface_url
from winding_audit.chain import chain_verdicts, pair_check
from winding_audit.lasagna import LasagnaFields
from winding_audit.tifxyz import load_tifxyz

ap = argparse.ArgumentParser()
ap.add_argument("sample")
ap.add_argument("volume")
ap.add_argument("--cache", default=".winding_audit_cache")
ap.add_argument("--out", default=None)
ap.add_argument("--samples", type=int, default=2000)
a = ap.parse_args()
cache, cat = open_catalog(a.cache)
fields = LasagnaFields(manifest(cat, a.sample, a.volume), cache)
W = per_winding(cat, a.sample, a.volume)
res, held = {}, {}
out = Path(a.out or f"results/chain_{a.sample}.json")
out.parent.mkdir(parents=True, exist_ok=True)


def checkpoint(complete=False):
    # Copy before judging: intermediate medians use only completed pairs.
    rows = {k: {j: v for j, v in r.items() if j != "samples"} for k, r in res.items()}
    spacing = chain_verdicts(rows)
    sample_rows = {k: dict(r.get("samples", {}), verdict=rows[k]["verdict"]) for k, r in res.items()}
    out.write_text(json.dumps({"sample": a.sample, "volume": a.volume,
                              "complete": complete, "samples_per_pair": a.samples,
                              "spacing_vox": spacing, "pairs": rows}, indent=1))
    out.with_name(out.stem + "_samples.json").write_text(json.dumps(sample_rows))

for k in sorted(W):
    if k + 1 not in W:
        continue
    t = time.time()
    for j in (k, k + 1):
        if j not in held:
            held[j] = load_tifxyz(surface_url(cat, a.sample, W[j][0], a.volume), cache)
    for j in [j for j in held if j < k]:
        held.pop(j)
    r = pair_check(held[k], held[k + 1], fields, samples=a.samples, seed=k, keep_samples=True)
    r["a"], r["b"] = W[k][1], W[k + 1][1]
    res[f"w{k:03d}->w{k + 1:03d}"] = r
    checkpoint()
    print(f"w{k:03d}->w{k + 1:03d}: windings {r.get('windings_median', float('nan')):.3f} "
          f"distance {r.get('distance_median_vox', float('nan')):.0f} coincide {r['coincide_fraction']:.3f} "
          f"n {r['n']} ({time.time() - t:.0f}s)", flush=True)
spacing = chain_verdicts(res)
for k, r in res.items():
    if r["verdict"] != "ok":
        print(f"{k}: {r['verdict']} (windings {r.get('windings_median', float('nan')):.2f}, "
              f"spacing ratio {r.get('spacing_ratio', float('nan')):.2f}, coincide {r['coincide_fraction']:.2f})")
checkpoint(complete=True)
