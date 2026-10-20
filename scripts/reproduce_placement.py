"""Run placement checks using argv lists, never shell-evaluated catalog text."""
import os
from pathlib import Path
from common import manifest, open_catalog, per_winding, surface_url, volume_url
from winding_audit.__main__ import main

cache_dir = os.environ.get('CACHE', '.winding_audit_cache')
cache, cat = open_catalog(cache_dir)
Path('results').mkdir(exist_ok=True)
for sid, vid in [('PHerc1447','20250521151220'), ('PHerc0800','20250521135224'), ('PHerc1667','20251217075048')]:
    if sid == 'PHerc1667':
        W=per_winding(cat,sid,vid)
        urls=[surface_url(cat,sid,W[k][0],vid) for k in sorted(W)]
        extra=['--level','3','--samples','800']
    else:
        urls=[surface_url(cat,sid,s,vid) for s,seg in sorted(cat['samples'][sid]['segments'].items())
              if seg['original_volume_id']==vid and surface_url(cat,sid,s,vid)]
        extra=[]
    main(['placement', *urls, '--volume',volume_url(cat,sid,vid), '--lasagna',manifest(cat,sid,vid),
          '--cache',cache_dir,'--out',f'results/placement_{sid}.json', *extra])
