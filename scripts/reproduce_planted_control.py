"""Planted sheet switch on real data (results/planted_control.json, figure).

A 140 x 200 vertex block of the human-traced PHerc0139 w040 surface is moved
onto the human-traced neighbouring winding w041 (walking each vertex along its
normal to the first crossing of w041), abruptly and with a 6-vertex ramp. The
truth (one winding inside the block) comes from human tracing, not from the
Lasagna field. The audit runs on a 360 x 420 crop around the block; the
unmodified crop is the negative control.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion

from common import manifest, open_catalog, surface_url
from winding_audit.audit import AuditParams, audit_surface
from winding_audit.controls import plant_block
from winding_audit.lasagna import LasagnaFields
from winding_audit.tifxyz import Surface, load_tifxyz

ap = argparse.ArgumentParser()
ap.add_argument("--cache", default=".winding_audit_cache")
a = ap.parse_args()
cache, cat = open_catalog(a.cache)
sid, vid = "PHerc0139", "20260102150214"
fields = LasagnaFields(manifest(cat, sid, vid), cache)
A = load_tifxyz(surface_url(cat, sid, "20250831000000", vid), cache)  # w040
B = load_tifxyz(surface_url(cat, sid, "20260108000000", vid), cache)  # w041
R0, R1, C0, C1 = 450, 810, 640, 1060
BR0, BR1, BC0, BC1 = 560, 700, 750, 950


def crop(s):
    return Surface(points=s.points[R0:R1, C0:C1].copy(), valid=s.valid[R0:R1, C0:C1].copy(), meta=s.meta, source=s.source)


blk = (BR0 - R0, BR1 - R0, BC0 - C0, BC1 - C0)
variants = {"original": crop(A)}
for ramp in (0, 6):
    P, _ = plant_block(A, B, BR0, BR1, BC0, BC1, ramp=ramp, max_dist=160, side=+1)
    variants[f"planted_ramp{ramp}"] = crop(P)
core = np.zeros(variants["original"].shape, bool)
core[blk[0]:blk[1], blk[2]:blk[3]] = True
seam = binary_dilation(core, iterations=12) & ~binary_erosion(core, iterations=12)
far = ~binary_dilation(core, iterations=40)


def line_sensitivity(mask, h=12):
    rows = range(blk[0] + 15, blk[1] - 15)
    cols = range(blk[2] + 15, blk[3] - 15)
    hits = [mask[r, blk[2] - h:blk[2] + h].any() for r in rows] + [mask[r, blk[3] - h:blk[3] + h].any() for r in rows]
    hits += [mask[blk[0] - h:blk[0] + h, c].any() for c in cols] + [mask[blk[1] - h:blk[1] + h, c].any() for c in cols]
    return float(np.mean(hits))


out = {"block_rows": [BR0, BR1], "block_cols": [BC0, BC1], "crop": [R0, R1, C0, C1], "variants": {}}
maps = {}
for name, S in variants.items():
    res = audit_surface(S, fields, AuditParams(), with_cos=False)
    maps[name] = res
    small = res.steps.step[1]  # +-4 edges
    thr = np.isfinite(small) & (np.abs(small) >= 0.5)
    v = {
        "threshold_only": {"vertices_far_flagged": float(thr[far & np.isfinite(small)].mean()),
                           "line_sensitivity": line_sensitivity(thr)},
        "plateau_criterion": {"vertices_far_flagged": float(res.step_mask[far].mean()),
                              "line_sensitivity": line_sensitivity(res.step_mask)},
    }
    if name != "original":
        edge_steps = []
        for r in range(blk[0] + 10, blk[1] - 10):
            for c0 in (blk[2], blk[3]):
                seg = np.abs(small[r, c0 - 4:c0 + 4])
                if np.isfinite(seg).any():
                    edge_steps.append(np.nanmax(seg))
        v["step_at_block_edge_median"] = float(np.median(edge_steps))
    out["variants"][name] = v
    print(name, json.dumps(v))
Path("results").mkdir(exist_ok=True)
Path("results/planted_control.json").write_text(json.dumps(out, indent=1))
np.savez_compressed("results/planted_control_maps.npz",
                    **{f"{k}_step4": m.steps.step[1] for k, m in maps.items()},
                    **{f"{k}_mask": m.step_mask for k, m in maps.items()})
