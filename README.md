# Space-to-Land Environmental Information Resilience — Experiments

Reproducible research prototype supporting the seminar paper. This repository studies **non-operational environmental information quality and resilience**, not targeting, tactical decisions, or physical actions.

## Repository contents

| Folder | Content |
|---|---|
| `01_rl_starter/` | Initial synthetic RL prototype (historical version) |
| `02_rl_version2/` | Revised synthetic RL baseline |
| `03_rl_version3/` | PPO/DQN/Rule-Based multi-seed experiment scripts |
| `04_satellite_availability/` | SSL4EO-S12 S1/S2 feature extraction, independent and correlated dropout simulations, outputs |
| `05_satellite_prediction/` | Random-forest cloud-metadata prediction, S1 vs S2 vs S1+S2 |
| `06_statistical_validation/` | RL seed-level and satellite split-level analysis, summary CSVs |
| `07_streamlit_dashboard/` | Streamlit visualizations and figure export |
| `08_low_storage_validation/` | Additional independent replay analysis |
| `09_validation_upgrade/` | Supporting RL audit and multi-seed scripts |
| `10_sen12ms_integration/` | Earlier optional SEN12MS data integration code; **not** the SSL4EO-S12 experiment dataset |

## Start the dashboard (Windows / VS Code)

```powershell
cd 07_streamlit_dashboard
py -m pip install -r requirements.txt
py -m streamlit run app.py
```

## Reproduce satellite experiments

The original `s1.zip` and `s2a.zip` archives are **not included** because they are hundreds of MB each. Obtain the SSL4EO-S12 100-patch sample from the official project, and place the archives in the repository root.

```powershell
py -m pip install numpy pandas tifffile matplotlib scikit-learn
cd 04_satellite_availability
py run_experiment.py
py correlated_dropout.py
cd ..\05_satellite_prediction
py run_validation.py
```

**Note:** The scripts were authored in separate experiment directories. Some use paths relative to their original directory names. If a relative-path error occurs, use the included precomputed CSVs or adjust input paths to the `04_satellite_availability/` folder. See each folder's README.

## Reproduce RL experiments

```powershell
cd 03_rl_version3
py -m pip install -r requirements.txt
py main.py --timesteps 15000 --episodes 30
```

See `03_rl_version3/README.md` for optional real-image feature integration. RL results in the paper are from **synthetic proxy-score experiments**, not end-to-end training on satellite imagery. The Rule-Based baseline slightly outperformed PPO/DQN in the reported experiment.

## Interpretation

- Resilience Index = mean degraded proxy performance / mean normal proxy performance; higher is better.
- S1/S2 dropout experiments simulate information **availability**, not image-classification performance.
- Satellite prediction uses cloud-percentage **metadata**, not independent land-cover labels.
- Multiple episodes from one seed are not independent RL training replications.
- The code and CSVs reflect separate stages of a developing research prototype; no claim of fully unified end-to-end training is made.

## Publication / privacy checklist

Do not commit personal API tokens, private Google Drive links, unpublished datasets, or copyrighted paper PDFs. Check the upstream dataset license before redistribution. The original large image archives and personal literature spreadsheet are deliberately excluded.
