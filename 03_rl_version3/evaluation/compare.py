import numpy as np
import pandas as pd
from environment.information_env import InformationEnv
from policies.rules import rule_based

CONDITIONS=['normal','mild','moderate','severe']

def evaluate(models, episodes=30, seed=1000, feature_rows=None):
    rows=[]
    for condition in CONDITIONS:
        for ep in range(episodes):
            scenario_seed=seed+ep+10000*CONDITIONS.index(condition)
            for name,model in models.items():
                env=InformationEnv(condition=condition,feature_rows=feature_rows)
                obs,_=env.reset(seed=scenario_seed)
                scores=[]
                while True:
                    action=rule_based(obs) if model is None else int(model.predict(obs,deterministic=True)[0])
                    obs,_,terminated,truncated,info=env.step(action)
                    scores.append(info['performance'])
                    if terminated or truncated:break
                rows.append(dict(method=name,condition=condition,episode=ep+1,performance=float(np.mean(scores))))
    raw=pd.DataFrame(rows)
    summary=raw.groupby(['method','condition'],as_index=False)['performance'].agg(['mean','std'])
    normal=summary[summary.condition=='normal'][['method','mean']].rename(columns={'mean':'normal_performance'})
    report=summary.merge(normal,on='method')
    if (report['normal_performance']<=0).any(): raise ValueError('Cannot calculate RI for zero normal performance')
    report['RI']=report['mean']/report['normal_performance']
    report.loc[report.condition=='normal','RI']=1.0
    return raw,summary,report
