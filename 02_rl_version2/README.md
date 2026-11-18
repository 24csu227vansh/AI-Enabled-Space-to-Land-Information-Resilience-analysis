# SpaceToLand_RL v2 — environmental information-resilience benchmark

This is a **non-operational synthetic research prototype**, not a military decision-making system. It contains no targeting, movement planning, or operational recommendations.

## Windows / VS Code

Open the extracted folder in VS Code. In the terminal:

```powershell
python -m venv venv
venv\Scripts\python.exe -m pip install -r requirements.txt
venv\Scripts\python.exe main.py --timesteps 15000 --episodes 30
```

Do **not** paste these terminal commands into `main.py`.

## Changes from v1

- Fixed common performance measure: `0.4 * freshness + 0.3 * coverage + 0.3 * reliability`.
- Actions change processed information quality through a common resource-allocation mechanism, not scoring weights.
- PPO and DQN train on a mixture of synthetic degradation levels.
- Matched held-out scenario seeds across PPO, DQN, Rule-Based.
- Normal, mild, moderate, severe scenario comparisons and RI.
- CSV results and plots in `results/`.

## Important scientific caveats

This **does not** ingest SEN12MS imagery. All quality indicators and degradation scenarios are synthetic. The environment is a simplified demonstrator, not validated against real information-processing systems. It cannot support claims of operational effectiveness. The algorithms currently use one training seed each; for publishable results, run multiple independent training seeds and confidence intervals. Performance and reward are related but not identical due to a processing-cost penalty. The chosen weights and degradation levels require sensitivity analysis.

RI is calculated per degradation level as mean performance under that condition divided by the same method's mean normal-condition performance. RI is relative retention; always report absolute performance too.
