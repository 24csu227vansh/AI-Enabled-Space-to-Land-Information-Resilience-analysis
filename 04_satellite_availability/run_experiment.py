import zipfile, io, json, re, csv, pathlib
import numpy as np
import tifffile
from collections import defaultdict
ROOT=pathlib.Path(__file__).resolve().parent
DATA=ROOT.parent
out=[]
for mod,bands in [('s1',['VV','VH']),('s2a',['B2','B3','B4','B8'])]:
    with zipfile.ZipFile(DATA/(mod+'.zip')) as z:
        entries=defaultdict(dict)
        for name in z.namelist():
            m=re.match(rf'{mod}/(\d{{7}})/([^/]+)/([^/]+)$',name)
            if m: entries[(m.group(1),m.group(2))][m.group(3)]=name
        for (loc,acq),files in sorted(entries.items()):
            if not all(b+'.tif' in files for b in bands): continue
            row={'modality':mod,'location_id':loc,'acquisition':acq}
            for b in bands:
                arr=tifffile.imread(io.BytesIO(z.read(files[b+'.tif']))).astype(np.float64)
                finite=np.isfinite(arr)
                row[b+'_valid_fraction']=float(finite.mean())
                row[b+'_mean']=float(arr[finite].mean()) if finite.any() else float('nan')
                row[b+'_std']=float(arr[finite].std()) if finite.any() else float('nan')
                row[b+'_shape']=str(arr.shape)
            if 'metadata.json' in files:
                md=json.loads(z.read(files['metadata.json']))
                if mod=='s2a':
                    row['cloud_pct']=md.get('CLOUDY_PIXEL_PERCENTAGE')
                    row['shadow_pct']=md.get('CLOUD_SHADOW_PERCENTAGE')
            out.append(row)
fields=list(dict.fromkeys(k for r in out for k in r))
with open(ROOT/'image_features.csv','w',newline='') as f:
    w=csv.DictWriter(f,fields);w.writeheader();w.writerows(out)
# Paired locations and acquisition counts: same location ID does not guarantee same time or spatial alignment.
s1={r['location_id'] for r in out if r['modality']=='s1'}
s2={r['location_id'] for r in out if r['modality']=='s2a'}
# Empirical observed valid-pixel fractions; never interpret these as semantic accuracy.
summary={'s1_observations':sum(r['modality']=='s1' for r in out),'s2_observations':sum(r['modality']=='s2a' for r in out),'s1_locations':len(s1),'s2_locations':len(s2),'shared_location_ids':len(s1&s2)}
for mod,bands in [('s1',['VV','VH']),('s2a',['B2','B3','B4','B8'])]:
    rr=[r for r in out if r['modality']==mod]
    summary[mod+'_mean_valid_pixel_fraction']=float(np.mean([r[b+'_valid_fraction'] for r in rr for b in bands]))
cloud=[float(r['cloud_pct']) for r in out if r['modality']=='s2a' and r.get('cloud_pct') is not None]
summary['s2_cloud_metadata_count']=len(cloud)
if cloud:
    summary['s2_cloud_pct_mean']=float(np.mean(cloud));summary['s2_cloud_pct_median']=float(np.median(cloud))
    summary['s2_cloud_pct_ge_20_count']=sum(x>=20 for x in cloud)
# Controlled independent random modality-dropout experiment, evaluated at location level.
# Availability = fraction of location IDs for which at least one retained observation exists.
# This is an availability proxy, NOT predictive performance or decision superiority.
rng=np.random.default_rng(20261118)
rows=[]
for missing in [0,.25,.5,.75]:
    for repeat in range(1000):
        a=rng.random(len(s1&s2))>=missing
        b=rng.random(len(s1&s2))>=missing
        for method,v in [('S1_only',a.mean()),('S2_only',b.mean()),('S1_or_S2',np.logical_or(a,b).mean())]:
            rows.append({'dropout_probability':missing,'repeat':repeat,'method':method,'location_availability':float(v)})
with open(ROOT/'dropout_trials.csv','w',newline='') as f:
    w=csv.DictWriter(f,rows[0]);w.writeheader();w.writerows(rows)
summary['dropout_simulation']='Independent random S1/S2 modality dropout; 1000 trials per rate; availability only'
(ROOT/'summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
