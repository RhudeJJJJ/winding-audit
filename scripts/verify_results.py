"""Offline arithmetic checks of bundled v0.1.0 results; no raw-data validation."""
import argparse
import json
from pathlib import Path
import numpy as np


def summarize(root):
    def load(n): return json.loads((root/n).read_text())
    chains={}
    for sid in ['PHerc0139','PHerc1667']:
        pairs=load(f'chain_{sid}.json')['pairs']; samples=load(f'chain_{sid}_samples.json')
        assert set(pairs)==set(samples)
        for k,r in pairs.items():
            s=samples[k];assert s['verdict']==r['verdict']
            if 'windings' in s:
                assert len(s['distance'])==len(s['windings'])==r['n']
                assert abs(np.median(s['windings'])-r['windings_median'])<=0.00011
        chains[sid]={'pairs':len(pairs),'flags':{k:r['verdict'] for k,r in pairs.items() if r['verdict']!='ok'}}
    selected=[r for r in load('chain_PHerc0139_samples.json').values() if r['verdict']=='ok']
    values=np.concatenate([r['windings'] for r in selected]);distance=np.concatenate([r['distance'] for r in selected])
    medians=np.array([np.median(r['windings']) for r in selected]);bins=[]
    for lo,hi in [(20,40),(40,60),(60,80),(80,100),(100,140),(140,220)]:
        use=(distance>=lo)&(distance<hi); pairbins=[]
        for r in selected:
            d=np.array(r['distance']);v=np.array(r['windings']);s=(d>=lo)&(d<hi)
            if s.any(): pairbins.append(float(np.median(v[s])))
        bins.append({'gap_vox':[lo,hi],'walks':int(use.sum()),'pairs':len(pairbins),
                     'pooled_median':float(np.median(values[use])),
                     'median_of_pair_bin_medians':float(np.median(pairbins))})
    dup=load('duplicate_w045_w046.json');p=load('placement_PHerc1447.json')
    early=[r for r in p if '/segments/20250502' in r['surface']];late=[r for r in p if r not in early]
    assert len(early)==11 and len(late)==4
    assert len(selected)==33 and len(values)==56695
    assert 0.82<dup['w046_within_12vox_of_w045']<0.83
    assert all(r['verdict']!='on_papyrus' for r in early)
    return {'scope':__doc__,'chains':chains,'overlap':dup,'selected_pairs':len(selected),'walks':len(values),
      'pooled_median':float(np.median(values)),'pooled_quartiles':np.percentile(values,[25,75]).tolist(),
      'median_of_pair_medians':float(np.median(medians)),'pair_median_range':[float(medians.min()),float(medians.max())],
      'gap_bins':bins,'placement_legacy':{'early':len(early),'later':len(late),
        'early_tilt_range':[min(r['field_tilt_median_deg'] for r in early),max(r['field_tilt_median_deg'] for r in early)],
        'later_tilt_range':[min(r['field_tilt_median_deg'] for r in late),max(r['field_tilt_median_deg'] for r in late)]},
      'caveat':'Selection uses the same field/geometry screen. Walks are correlated. Descriptive statistics, not unbiased calibration or sensitivity.'}


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--results',default='results');ap.add_argument('--out');a=ap.parse_args()
    out=summarize(Path(a.results));txt=json.dumps(out,indent=2,allow_nan=False)+'\n';print(txt)
    if a.out: Path(a.out).write_text(txt)
