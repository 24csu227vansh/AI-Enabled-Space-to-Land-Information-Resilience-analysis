# SpaceToLand_RL v3 — SEN12MS image-feature integration

Non-operational environmental information-resilience research prototype.
No targeting, route optimization, military planning, or physical-world action execution.

## Windows setup in VS Code

```powershell
python -m venv venv
venv\Scripts\python.exe -m pip install -r requirements.txt
```

## A. Smoke test using only synthetic indicators

```powershell
venv\Scripts\python.exe main.py --timesteps 15000 --episodes 30
```

## B. Add actual SEN12MS TIFF patches

Download SEN12MS from the original dataset source and unzip the appropriate
season/ROI image files into `data/raw/sen12ms/`.
Dataset: https://mediatum.ub.tum.de/1474000
Citation: Schmitt et al. (2019), DOI: 10.5194/isprs-annals-IV-2-W7-153-2019.

Start with a **small extracted subset**. SEN12MS native filenames use `_s1_`, `_s2_`
and `_lc_` tokens for SAR, optical and land-cover patches. The feature extractor
pairs SAR and optical TIFFs by matching names and modality directories.

```powershell
venv\Scripts\python.exe -m utils.sen12ms_features --dataset data/raw/sen12ms --limit 100 --output data/processed/sen12ms_features.csv
venv\Scripts\python.exe main.py --features data/processed/sen12ms_features.csv --timesteps 15000 --episodes 30
```

Outputs appear in `results/`. No imagery is included in this ZIP; the original
dataset must be downloaded separately.

### Evidence boundaries

- **Observed from imagery:** Sentinel-1/2 band summary statistics and finite/valid-pixel fractions.
- **Used in RL prototype:** mean S1/S2 valid fraction influences the *coverage* component.
- **Still simulated:** freshness, reliability, degradation levels, processing efficiency, action effects.
- A valid fraction is not a satellite sensor reliability measurement. Nonzero valid
  fractions do **not** demonstrate genuine awareness/decision performance.
- This is partial dataset integration, **not CNN/ViT training** or true multimodal
  fusion. The next experiment should build supervised image-derived land-cover
  features, geographic train/test splits, single-modality ablations and repeated seeds.
- Scores are proxy simulation scores, not operational effectiveness metrics.
- RI = mean degraded proxy performance / mean normal proxy performance.
- Never interpret Version 2 numbers as validated real-data outcomes.
