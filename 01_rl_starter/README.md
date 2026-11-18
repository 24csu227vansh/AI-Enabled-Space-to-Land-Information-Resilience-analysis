# SpaceToLand_RL — safe research prototype

This starter project is a **non-operational environmental information-resilience simulation**.
It compares PPO, DQN, and a fixed Rule-Based baseline on abstract information-quality indicators.
It does not perform military planning, targeting, navigation, or operational decision-making.

## 1. Setup (Windows / VS Code)

```powershell
cd SpaceToLand_RL
python -m venv venv
venv\Scripts\python.exe -m pip install --upgrade pip
venv\Scripts\python.exe -m pip install -r requirements.txt
```

If activation works in your terminal, you can also run:
```powershell
.\venv\Scripts\Activate.ps1
```

## 2. Run the starter experiment

```powershell
python main.py --timesteps 15000 --episodes 30
```

The first run uses reproducible synthetic observations so that the software can be tested
without downloading the full SEN12MS dataset. It saves CSV results and plots in `results/`.

## 3. Connect SEN12MS later

SEN12MS is not a ready-made Gymnasium environment. Its imagery must first be read and
preprocessed into numerical feature records. This starter accepts a CSV file with these
columns:

- `freshness`
- `coverage`
- `reliability`

Each value should be normalized to [0, 1]. Example:

```powershell
python main.py --features data/processed/features.csv --timesteps 15000 --episodes 30
```

The `utils/data_loader.py` module validates this format. Do not label synthetic scenario
variables as real SEN12MS fields. SEN12MS contains Sentinel-1, Sentinel-2 and MODIS
land-cover data; communication status and delay variables in this project are simulated.

## 4. What the prototype measures

Per-step information score:
`P = 0.4 * freshness + 0.3 * coverage + 0.3 * reliability`

Resilience Index:
`RI = mean(degraded performance) / mean(normal performance)`

Both normal and degraded scores are saved alongside RI. RI is undefined if normal
performance is zero, so the evaluator reports an error in that case.

## 5. Project layout

```text
SpaceToLand_RL/
├── data/
│   └── processed/
├── environment/
│   └── space_land_env.py
├── models/
├── training/
│   ├── train_ppo.py
│   ├── train_dqn.py
│   └── rule_based.py
├── evaluation/
│   └── evaluate.py
├── utils/
│   └── data_loader.py
├── results/
├── main.py
└── requirements.txt
```

## 6. Research integrity

- The default run is a software prototype using synthetic feature records.
- Do not report its output as a result obtained from SEN12MS.
- For the paper, document the exact dataset subset, geographic split, preprocessing,
  random seeds, training settings and simulation degradation settings.
- Run multiple seeds and report mean and standard deviation.
- The reward weights are initial design choices and should be tested with sensitivity analysis.
