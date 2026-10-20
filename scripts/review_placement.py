"""Recheck PHerc1447 using source URLs frozen in the supplied results."""
import argparse
import json
from pathlib import Path
from common import manifest, open_catalog, volume_url
from winding_audit.ct import CTVolume
from winding_audit.placement import placement_check
from winding_audit.tifxyz import load_tifxyz
from winding_audit.lasagna import LasagnaFields

ap = argparse.ArgumentParser()
ap.add_argument('--cache', default='.winding_audit_cache')
ap.add_argument('--samples', type=int, default=1500)
ap.add_argument('--out', default='results/review/placement_PHerc1447.json')
a = ap.parse_args()
cache, cat = open_catalog(a.cache)
sid, vid = 'PHerc1447', '20250521151220'
rows = json.loads(Path('results/placement_PHerc1447.json').read_text())
fields = LasagnaFields(manifest(cat, sid, vid), cache)
out = []
for old in rows:
    s = old['surface']
    ct = CTVolume(volume_url(cat, sid, vid), cache=cache)
    r = placement_check(load_tifxyz(s, cache), ct, ct.channel.array.shape,
                        samples=a.samples, fields=fields)
    r.update(surface=s, seed=0, level=0)
    out.append(r)
    print(len(out), r['verdict'], r.get('ct_screen_verdict'),
          r.get('field_tilt_median_deg'), flush=True)
    p = Path(a.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1))
