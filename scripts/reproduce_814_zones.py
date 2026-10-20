"""PHerc0814: two independent traces disagree with the field in the same z band (results/zones_PHerc0814.json).

Audits two auto-grown PHerc0814 traces and reports the z range of their
three largest field/surface disagreement zones.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from common import manifest, open_catalog, surface_url
from winding_audit.audit import audit_surface
from winding_audit.lasagna import LasagnaFields
from winding_audit.tifxyz import load_tifxyz

ap = argparse.ArgumentParser()
ap.add_argument("--cache", default=".winding_audit_cache")
a = ap.parse_args()
cache, cat = open_catalog(a.cache)
sid, vid = "PHerc0814", "20260309142202"
fields = LasagnaFields(manifest(cat, sid, vid), cache)
out = {}
for segid in ("20260226000000", "20260226113324"):
    s = load_tifxyz(surface_url(cat, sid, segid, vid), cache)
    res = audit_surface(s, fields)
    zones = []
    for z in res.disagreements[:3]:
        r0, r1 = z["row_range"]
        c0, c1 = z["col_range"]
        zz = s.points[r0:r1 + 1, c0:c1 + 1, 2]
        zones.append({**z, "z_range": [float(np.nanmin(zz)), float(np.nanmax(zz))], "z_median": float(np.nanmedian(zz))})
    out[segid] = {"long_id": cat["samples"][sid]["segments"][segid]["long_id"], "zones": zones}
    print(segid, [(z["id"], z["vertices"], round(z["z_median"])) for z in zones])
Path("results").mkdir(exist_ok=True)
Path("results/zones_PHerc0814.json").write_text(json.dumps(out, indent=1))
