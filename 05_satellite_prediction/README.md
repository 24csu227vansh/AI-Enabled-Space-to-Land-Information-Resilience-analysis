# Preliminary S1/S2 feature-based prediction validation

Input: `Satellite_Experiment/image_features.csv`, generated from the user's SSL4EO-S12 S1/S2 archives. This validation uses **real image-derived summary statistics**, not raw-pixel neural network training.

## Task
Predict the S2 metadata field `cloud_pct` from S1 VV/VH pixel mean and standard deviation, S2 B2/B3/B4/B8 pixel mean and standard deviation, or their concatenation. The target is **scene/acquisition metadata** rather than independently validated local pixel cloud-cover ground truth. It is **not land-cover classification**.

## Matching
For each of 400 S2 acquisitions, match the nearest-date S1 acquisition with the same numbered sample location ID. Median date difference 4.19 days; maximum 28.31 days. Same sample ID does not establish exact georegistration. Different acquisition dates can reduce meaningful fusion.

## Evaluation
100 location IDs, 400 S2 acquisition records. Five seeded 75%/25% group splits by location (17,118,219,320,421), 300 train and 100 test records per split. The same location never appears in both sets for a split. RandomForestRegressor(n_estimators=120,min_samples_leaf=3,max_features=0.8) compared with training-mean predictor. Each split tests generalization to unseen sample IDs, but repeated splits are not independent datasets.

## Mean results
| Model | MAE (cloud percentage points) | RMSE | R2 |
|---|---:|---:|---:|
| S1 only | 2.633 | 3.113 | -0.080 |
| S2 only | 2.481 | 2.963 | 0.023 |
| S1+S2 | 2.502 | 2.998 | -0.000 |
| Train-mean baseline | 2.596 | 3.003 | -0.004 |

**Interpretation:** S2-only is marginally better; fusion does not show predictive improvement. Very low R2 indicates little predictive explanatory power. No claims of robust AI, land-cover classification, statistical significance, or operational performance are justified.

## Reproduce
From the project root with `numpy`, `pandas`, and `scikit-learn` installed: `python Satellite_Fusion_Validation/run_validation.py`. The code expects the precomputed image_features.csv one directory above. Outputs are paired_features.csv, model_metrics_by_split.csv, model_summary.csv, heldout_predictions.csv. For stand-alone reproduction, copy image_features.csv into a sibling `Satellite_Experiment` folder.

## Next experiment
Obtain reliable patch-level land-cover labels and verify temporal/spatial alignment; then evaluate S1-only, S2-only and fused classifiers with location-disjoint splits and degradation tests. Do not calculate a task-performance Resilience Index from metadata availability alone.
