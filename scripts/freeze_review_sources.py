"""Snapshot metadata and URLs used in the review, not every remote data chunk."""
import argparse,hashlib,json
from datetime import datetime,timezone
from pathlib import Path
from common import open_catalog,manifest,volume_url,per_winding,surface_url
from winding_audit.catalog import BUCKET_HTTPS
from winding_audit.zarr_io import read_bytes

ap=argparse.ArgumentParser();ap.add_argument('--cache',default='.winding_audit_cache');a=ap.parse_args()
cache,cat=open_catalog(a.cache)
entries=[]
for sid,vid in [('PHerc0139','20260102150214'),('PHerc1667','20251217075048'),('PHerc1447','20250521151220')]:
    mu=manifest(cat,sid,vid);vu=volume_url(cat,sid,vid);raw=read_bytes(mu,cache)
    entry={'sample':sid,'volume':vid,'lasagna_manifest_url':mu,'lasagna_manifest_sha256':hashlib.sha256(raw).hexdigest(),
           'lasagna_manifest':json.loads(raw),'ct_url':vu,'per_winding':{str(k):{'segment':v[1],'surface_url':surface_url(cat,sid,v[0],vid)} for k,v in per_winding(cat,sid,vid).items()}}
    entry['ct_metadata']={}
    for suffix in ['.zattrs','0/.zarray']:
        b=read_bytes(vu.rstrip('/')+'/'+suffix,cache)
        entry['ct_metadata'][suffix]={'sha256':hashlib.sha256(b).hexdigest(),'content':json.loads(b)} if b else None
    entries.append(entry)
rawcat=read_bytes(BUCKET_HTTPS+'/metadata.json',cache)
p=Path('results/review/source_snapshot.json');p.parent.mkdir(parents=True,exist_ok=True)
p.write_text(json.dumps({'retrieved_utc':datetime.now(timezone.utc).isoformat(),'catalog_url':BUCKET_HTTPS+'/metadata.json',
    'catalog_raw_sha256':hashlib.sha256(rawcat).hexdigest(),'sources':entries,
    'scope':'Metadata/selection snapshot only; remote TIFF and Zarr chunk objects are not immutable or fully hash-locked.'},indent=2))
print(p)
