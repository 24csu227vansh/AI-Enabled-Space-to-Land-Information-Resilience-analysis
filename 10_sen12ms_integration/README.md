# SEN12MS integration — Stage 1

This is an independent feature-extraction add-on for the existing SpaceToLand_RL project. It reads **real** matched Sentinel-1 (2-band) and Sentinel-2 (13-band) GeoTIFF images and writes descriptive statistics to CSV. It does not fabricate data or claim a CNN has been trained.

## Dataset

Original dataset: https://mediatum.ub.tum.de/1474000
Official toolbox and split guidance: https://github.com/schmitt-muc/SEN12MS

Download and extract a manageable matching Sentinel-1/Sentinel-2 season or subset from the official dataset. The extractor recursively searches for files with `_s1_` and `_s2_` in their names. Keep the original directory structure. The full dataset is very large; don't download everything just to test this step.

## Windows commands

In the folder containing this script:

```powershell
python -m pip install -r requirements.txt
python extract_sen12ms.py --data "data/raw" --limit 100
```

The output is `data/processed/sen12ms_features.csv`.

## Important limitations

- These are real image statistics, but **not** CNN embeddings or validated situational-awareness scores.
- `s1_valid_fraction` and `s2_valid_fraction` indicate valid pixel fractions, not satellite availability, freshness, communication reliability, or classification accuracy.
- The existing RL prototype expects a CSV with `freshness,coverage,reliability`. Do **not** rename or transform the columns into those concepts without an explicit, validated mapping. This stage intentionally does not plug directly into RL.
- For research evaluation, use the official holdout split; avoid spatial leakage between train and test.
- A subsequent stage will implement a proper supervised image model, modality-ablation tests, and a defined link to the abstract RL information state.
