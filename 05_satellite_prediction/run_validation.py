"""Reproducible weak-label cloud-metadata prediction; NOT land-cover classification."""
from pathlib import Path
import re
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT=Path(__file__).resolve().parent
INPUT=ROOT.parent/'Satellite_Experiment'/'image_features.csv'
raw=pd.read_csv(INPUT)
s1=raw[raw.modality=='s1'].copy()
s2=raw[raw.modality=='s2a'].copy()
def timestamp(s):
 m=re.search(r'20\d{6}T\d{6}',str(s));return pd.to_datetime(m.group(0),format='%Y%m%dT%H%M%S') if m else pd.NaT
for d in (s1,s2):d['date']=d.acquisition.map(timestamp)
# Nearest-in-time acquisition for each S2 observation, restricted to same sample location.
pairs=[]
for _,r in s2.iterrows():
 pool=s1[s1.location_id==r.location_id]
 if pool.empty:continue
 delta=(pool.date-r.date).abs().dt.total_seconds()/86400
 j=delta.idxmin(); z=pool.loc[j]
 rec={'location_id':int(r.location_id),'s2_acquisition':r.acquisition,'s1_acquisition':z.acquisition,
      'gap_days':float(delta.loc[j]),'cloud_pct':float(r.cloud_pct)}
 for b in ('VV','VH'):
  for stat in ('mean','std'):rec[f's1_{b}_{stat}']=float(z[f'{b}_{stat}'])
 for b in ('B2','B3','B4','B8'):
  for stat in ('mean','std'):rec[f's2_{b}_{stat}']=float(r[f'{b}_{stat}'])
 pairs.append(rec)
df=pd.DataFrame(pairs);df.to_csv(ROOT/'paired_features.csv',index=False)
s1cols=[c for c in df if c.startswith('s1_') and c.endswith(('mean','std'))]
s2cols=[c for c in df if c.startswith('s2_') and c.endswith(('mean','std'))]
methods={'S1 only':s1cols,'S2 only':s2cols,'S1 + S2':s1cols+s2cols}
rows=[];preds=[]
for seed in [17,118,219,320,421]:
 tr,te=next(GroupShuffleSplit(n_splits=1,test_size=.25,random_state=seed).split(df,groups=df.location_id))
 ytr=df.cloud_pct.iloc[tr].to_numpy(); yte=df.cloud_pct.iloc[te].to_numpy()
 for name,cols in [('Train-mean baseline',[]),*methods.items()]:
  if cols:
   model=RandomForestRegressor(n_estimators=120,min_samples_leaf=3,max_features=0.8,random_state=seed,n_jobs=2)
   model.fit(df.iloc[tr][cols],ytr); yp=model.predict(df.iloc[te][cols])
  else:yp=np.full(len(te),ytr.mean())
  rows.append({'seed':seed,'method':name,'n_train':len(tr),'n_test':len(te),
   'mae_cloud_percentage_points':mean_absolute_error(yte,yp),
   'rmse_cloud_percentage_points':np.sqrt(mean_squared_error(yte,yp)),
   'r2':r2_score(yte,yp)})
  preds.extend({'seed':seed,'method':name,'location_id':int(df.iloc[i].location_id),
                'actual_cloud_pct':float(yte[k]),'predicted_cloud_pct':float(yp[k])} for k,i in enumerate(te))
res=pd.DataFrame(rows);res.to_csv(ROOT/'model_metrics_by_split.csv',index=False)
pd.DataFrame(preds).to_csv(ROOT/'heldout_predictions.csv',index=False)
summary=res.groupby('method').agg(mae_mean=('mae_cloud_percentage_points','mean'),mae_sd=('mae_cloud_percentage_points','std'),rmse_mean=('rmse_cloud_percentage_points','mean'),r2_mean=('r2','mean')).reset_index()
summary.to_csv(ROOT/'model_summary.csv',index=False)
print('Pairs:',len(df),'Locations:',df.location_id.nunique(),'Nearest gap median/max days:',round(df.gap_days.median(),2),round(df.gap_days.max(),2))
print(summary.to_string(index=False))
