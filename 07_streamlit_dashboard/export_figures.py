"""Export the same statistical plots used by the Streamlit dashboard without launching a browser."""
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
D=ROOT/'data'
OUT=ROOT/'figures'
OUT.mkdir(exist_ok=True)
methods=['Rule-Based','DQN','PPO']
conditions=['normal','mild','moderate','severe']

def save(fig,name):
    fig.tight_layout()
    fig.savefig(OUT/name,dpi=200,bbox_inches='tight')
    plt.close(fig)

seed=pd.read_csv(D/'rl_seed_means.csv')
fig,ax=plt.subplots(figsize=(9,4.8))
for m in methods:
    s=seed[seed.method.eq(m)].groupby('condition').performance.agg(['mean','std']).reindex(conditions)
    ax.errorbar(range(4),s['mean'],yerr=s['std'].fillna(0),marker='o',capsize=3,label=m)
ax.set_xticks(range(4),['Normal','Mild','Moderate','Severe']);ax.set_ylim(0,1);ax.set_ylabel('Normalized proxy performance');ax.set_title('Figure 2. RL performance');ax.legend();ax.grid(axis='y',alpha=.25)
save(fig,'figure2_rl_performance_python.png')

ri=pd.read_csv(D/'rl_seed_resilience.csv')
fig,ax=plt.subplots(figsize=(9,4.8));x=np.arange(3);w=.24
for j,m in enumerate(methods):
    s=ri[ri.method.eq(m)];vals=[s[f'{c}_RI'].mean() for c in ['mild','moderate','severe']];errs=[s[f'{c}_RI'].std(ddof=1) for c in ['mild','moderate','severe']]
    ax.bar(x+(j-1)*w,vals,width=w,yerr=errs,capsize=3,label=m)
ax.set_xticks(x,['Mild','Moderate','Severe']);ax.set_ylim(0,1);ax.set_ylabel('Resilience Index');ax.set_title('Figure 3. Resilience under degradation');ax.legend()
save(fig,'figure3_resilience_python.png')

drop=pd.read_csv(D/'dropout_summary.csv');fig,ax=plt.subplots(figsize=(9,4.8))
for m,label in [('S1_only','S1 only'),('S2_only','S2 only'),('S1_or_S2','S1 or S2')]:
    s=drop[drop.method.eq(m)].sort_values('dropout_probability');ax.plot(s.dropout_probability*100,s['mean']*100,marker='o',label=label)
ax.set(xlabel='Dropout probability (%)',ylabel='Information availability (%)',ylim=(0,105),title='Supplementary. Independent sensor dropout');ax.legend();ax.grid(alpha=.25)
save(fig,'supplementary_independent_dropout_python.png')

corr=pd.read_csv(D/'satellite_correlated_dropout.csv');fig,ax=plt.subplots(figsize=(9,4.8))
for frac,label in [(0.,'Independent'),(.5,'50% shared'),(1.,'Fully shared')]:
    s=corr[np.isclose(corr.shared_failure_fraction,frac)].sort_values('dropout_probability');ax.plot(s.dropout_probability*100,s.either_mean*100,marker='o',label=label)
ax.set(xlabel='Dropout probability (%)',ylabel='Either sensor available (%)',ylim=(0,105),title='Figure 4. Correlated sensor dropout');ax.legend();ax.grid(alpha=.25)
save(fig,'figure4_correlated_dropout_python.png')

pred=pd.read_csv(D/'satellite_split_metrics.csv');labels=['Train-mean baseline','S1 only','S2 only','S1 + S2'];s=pred.groupby('method').mae_cloud_percentage_points.agg(['mean','std']).reindex(labels)
fig,ax=plt.subplots(figsize=(9,4.8));ax.bar(labels,s['mean'],yerr=s['std'],capsize=4);ax.tick_params(axis='x',labelrotation=12);ax.set(ylabel='MAE (cloud-percentage points)',title='Figure 5. Cloud-metadata prediction');save(fig,'figure5_satellite_prediction_python.png')
print('Exported 5 PNG charts to',OUT)
