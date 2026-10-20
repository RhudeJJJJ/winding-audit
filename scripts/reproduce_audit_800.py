"""PHerc0800: the per-surface audit on a densely wound scroll (results/audit_PHerc0800.json).

Audits every PHerc0800 segment attributed to the 8.64 um volume and records
the share of each surface in disagreement zones together with the sheet
spacing and the grid edge length; with sheets closer together than one grid
edge, the windows of the audit are too long to separate a trace error from
ordinary tilt.
"""

import argparse
import json
from pathlib import Path

from common import manifest, open_catalog, surface_url
from winding_audit.audit import audit_surface, summarize
from winding_audit.lasagna import LasagnaFields
from winding_audit.tifxyz import load_tifxyz

ap = argparse.ArgumentParser()
ap.add_argument("--cache", default=".winding_audit_cache")
a = ap.parse_args()
cache, cat = open_catalog(a.cache)
sid, vid = "PHerc0800", "20250521135224"
fields = LasagnaFields(manifest(cat, sid, vid), cache)
out = {}
for segid, seg in sorted(cat["samples"][sid]["segments"].items()):
    if seg["original_volume_id"] != vid or not surface_url(cat, sid, segid, vid):
        continue
    sm = summarize(audit_surface(load_tifxyz(surface_url(cat, sid, segid, vid), cache), fields))
    sm.pop("params")
    out[seg["long_id"]] = sm
    print(seg["long_id"], f"disagreement {sm['disagreement_vertex_fraction']:.3f} spacing {sm['median_winding_spacing_vox']:.1f} "
          f"edge {sm['median_edge_length_vox']:.1f}", flush=True)
Path("results").mkdir(exist_ok=True)
Path("results/audit_PHerc0800.json").write_text(json.dumps(out, indent=1))
