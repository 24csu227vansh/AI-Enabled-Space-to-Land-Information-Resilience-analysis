# Next testing stage — multi-seed reproducibility

This add-on works with `SpaceToLand_RL_v3` (not Version 2). Copy `multi_seed_experiment.py` and `audit_results.py` into the **root** of the extracted v3 folder. No SEN12MS images are bundled.

Install dependencies in your existing venv:

```powershell
venv\Scripts\python.exe -m pip install -r requirements.txt
venv\Scripts\python.exe -m pip install scipy
```

First run a smaller reproducibility check:

```powershell
venv\Scripts\python.exe multi_seed_experiment.py --seeds 3 --timesteps 15000 --episodes 30
venv\Scripts\python.exe audit_results.py results/multi_seed/all_episodes.csv
```

Then a stronger experiment (longer running):

```powershell
venv\Scripts\python.exe multi_seed_experiment.py --seeds 5 --timesteps 30000 --episodes 50
venv\Scripts\python.exe audit_results.py results/multi_seed/all_episodes.csv
```

If you have prepared SEN12MS features using the v3 extraction tool, append `--features data/processed/sen12ms_features.csv` to the multi-seed command. Only **coverage** is derived from image completeness in v3; freshness, reliability, and degradation are simulated. This is not a CNN or multimodal fusion experiment.

## Important design limitation found in v3

In `environment/information_env.py`, each new observation is independently sampled. Actions improve the score of the **current** observation but do not influence the next state, which means this is primarily a contextual information-selection task rather than a long-horizon adaptive control task. The reward is based on the same synthetic quality dimensions as the performance metric. Therefore do **not** claim that the experiments prove general RL-based resilience, real-world decision superiority, or significant gains without appropriate validation.

## Next steps for the paper

1. Report training seed, test episodes, mean and SD, and paired differences.
2. Compare results with and without image-derived coverage, with matched data splits.
3. Develop an independently evaluated supervised land-cover model (e.g. Sentinel-1 only, Sentinel-2 only, both), using spatially separated train/test areas.
4. Test whether source availability changes classification accuracy/F1/IoU; don't equate image-valid-pixel fraction with prediction accuracy.
5. Describe the study as a **non-operational environmental information resilience prototype**.
