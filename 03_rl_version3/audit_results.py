"""Check paired policy performance using raw episodes from multi_seed_experiment.
Runs without Gymnasium or stable-baselines3; requires pandas/numpy/scipy.
"""
import argparse
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('csv',help='results/multi_seed/all_episodes.csv')
    args=ap.parse_args()
    df=pd.read_csv(args.csv)
    required={'method','condition','episode','performance','training_seed'}
    if not required.issubset(df.columns): raise ValueError(f'Missing columns: {required-set(df.columns)}')
    if not df.performance.between(0,1).all(): raise ValueError('Performance outside [0,1]')
    rows=[]
    for condition in ['normal','mild','moderate','severe']:
        d=df[df.condition==condition]
        # Average within training seed first; test is across independent training seeds.
        wide=d.groupby(['training_seed','method']).performance.mean().unstack()
        for method in ['DQN','PPO']:
            paired=wide[[method,'Rule-Based']].dropna()
            differences=paired[method]-paired['Rule-Based']
            n=len(differences)
            if n<2: continue
            stat,p=ttest_rel(paired[method],paired['Rule-Based'])
            rows.append({'condition':condition,'comparison':method+' minus Rule-Based',
                         'training_seeds':n,'mean_difference':differences.mean(),
                         'sd_difference':differences.std(ddof=1),'paired_t_p_uncorrected':p})
    result=pd.DataFrame(rows)
    out='results/multi_seed/paired_tests.csv'
    result.to_csv(out,index=False)
    print(result.to_string(index=False))
    print('Caution: exploratory p-values; small training-seed counts and multiple comparisons require care.')
    print('Training seeds share the same held-out evaluation scenarios; this is deliberate pairing.')
if __name__=='__main__':main()
