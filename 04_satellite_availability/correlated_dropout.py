"""Reproducible non-operational environmental sensor-availability stress test."""
import csv, pathlib
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parent
rng=np.random.default_rng(20261118)
N=100; TRIALS=2000
rows=[]
for p in [0,.25,.5,.75]:
 for shared in [0,.5,1.0]:
  # Mixture: with probability shared, both modalities share the same Bernoulli failure.
  # Otherwise failures are independent. Marginal dropout stays p for each sensor.
  for trial in range(TRIALS):
   common=rng.random(N)<p
   independent1=rng.random(N)<p
   independent2=rng.random(N)<p
   coupled=rng.random(N)<shared
   lost1=np.where(coupled,common,independent1)
   lost2=np.where(coupled,common,independent2)
   rows.append((p,shared,trial,float((~lost1).mean()),float((~lost2).mean()),float((~(lost1&lost2)).mean())))
with (ROOT/'correlated_dropout_trials.csv').open('w',newline='') as f:
 w=csv.writer(f);w.writerow(['dropout_probability','shared_failure_fraction','trial','S1_availability','S2_availability','either_availability']);w.writerows(rows)
with (ROOT/'correlated_dropout_summary.csv').open('w',newline='') as f:
 w=csv.writer(f);w.writerow(['dropout_probability','shared_failure_fraction','S1_mean','S2_mean','either_mean','either_ci95_low','either_ci95_high','expected_either'])
 for p in [0,.25,.5,.75]:
  for shared in [0,.5,1.0]:
   subset=np.array([r[3:6] for r in rows if r[0]==p and r[1]==shared]); means=subset.mean(axis=0)
   # 95% Monte Carlo CI for estimated mean, not uncertainty about real satellite failures.
   se=subset[:,2].std(ddof=1)/np.sqrt(TRIALS)
   expected=1-(shared*p+(1-shared)*p*p)
   w.writerow([p,shared,*means,means[2]-1.96*se,means[2]+1.96*se,expected])
print((ROOT/'correlated_dropout_summary.csv').read_text())
