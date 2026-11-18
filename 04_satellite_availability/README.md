# Satellite image availability experiment (preliminary)

Input: user-provided SSL4EO-S12 sample archives `s1.zip` and `s2a.zip`. Place both in the parent directory and run `python run_experiment.py`. Dependencies: numpy, tifffile. For plotting: pandas and matplotlib.

## Verified
- 100 common location IDs, 400 acquisition folders per modality.
- Extracted VV/VH S1 and B2/B3/B4/B8 S2 pixel summary statistics, plus S2 cloud metadata.
- All inspected pixels in selected bands were finite. This does **not** establish cloud-free images or correct georegistration.
- Cloud percentages are metadata and are not independently validated cloud masks.

## Experiment
Simulate independent complete S1 and S2 modality dropout at probabilities 0%, 25%, 50%, 75%; 1000 Monte Carlo repeats for each. Measure whether each location has an available S1 observation, S2 observation, or at least one modality. This is an **availability-only** redundancy illustration, not trained multimodal fusion, classification accuracy, PPO/DQN evaluation, or resilience index for actual task performance. Independent failures are an assumption; correlated failures can reduce redundancy gains.

## Limitations
Location ID match is not proof of spatial registration or same-date acquisition. No land-cover labels. Four acquisitions per location are not independent locations. No trained model, predictive metrics, or evidence of superiority.

Files: `image_features.csv`, `dropout_trials.csv`, `dropout_summary.csv`, `summary.json`, `dropout_plot.png`, and this reproducible script.

## Correlated-dropout sensitivity extension
Run `python correlated_dropout.py` (numpy required). The extension models per-location dropout probabilities of 0%, 25%, 50%, 75% and shared-failure mixture fractions 0%, 50%, 100%, using 2,000 Monte Carlo trials for each setting. Both modalities retain the same marginal dropout probability. The shared-failure fraction is a **simulation assumption**, not an empirical estimate of satellite outage correlation. The CSV includes theoretical expectations and Monte Carlo 95% confidence intervals for the simulated mean (not real-world uncertainty). For p=0.5, expected availability of either modality is 75% with independent failures, 62.5% with 50% shared failures, and 50% with fully shared failures. No trained fusion or RL policy is evaluated.
