"""Extract transparent image summary features from genuine paired SEN12MS GeoTIFFs.
No synthetic fields are passed off as real observations.
"""
import argparse
import csv
from pathlib import Path
import numpy as np


def paired_files(root):
    for s1 in sorted(root.rglob('*_s1_*.tif')):
        s2_name = s1.name.replace('_s1_', '_s2_')
        s2_dir = s1.parent.name.replace('_s1', '_s2')
        candidates = [s1.with_name(s2_name), s1.parent.parent / s2_dir / s2_name]
        match = next((p for p in candidates if p.exists()), None)
        if match:
            yield s1, match


def stats(path, expected_bands):
    import rasterio
    with rasterio.open(path) as ds:
        if ds.count != expected_bands:
            raise ValueError(f'{path}: expected {expected_bands} bands, got {ds.count}')
        arr = ds.read(masked=True).astype(np.float32)
        features = {}
        fractions = []
        for band in range(ds.count):
            vals = arr[band].compressed()
            vals = vals[np.isfinite(vals)]
            fractions.append(len(vals) / arr[band].size)
            features[f'b{band+1:02d}_mean'] = float(np.mean(vals)) if len(vals) else np.nan
            features[f'b{band+1:02d}_std'] = float(np.std(vals)) if len(vals) else np.nan
        return features, float(np.mean(fractions))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True, help='Root folder of extracted SEN12MS GeoTIFF files')
    ap.add_argument('--output', default='data/processed/sen12ms_features.csv')
    ap.add_argument('--limit', type=int, default=100)
    args = ap.parse_args()
    if args.limit < 1:
        ap.error('--limit must be at least 1')
    root = Path(args.data)
    if not root.exists():
        ap.error(f'Dataset folder not found: {root}')
    rows = []
    for s1, s2 in paired_files(root):
        try:
            s1_feats, s1_valid = stats(s1, 2)
            s2_feats, s2_valid = stats(s2, 13)
        except Exception as exc:
            print(f'Skipping {s1.name}: {exc}')
            continue
        row = {'scene_id': s1.stem.replace('_s1_', '_'), 's1_file': str(s1), 's2_file': str(s2),
               's1_valid_fraction': s1_valid, 's2_valid_fraction': s2_valid}
        row.update({f's1_{k}': v for k,v in s1_feats.items()})
        row.update({f's2_{k}': v for k,v in s2_feats.items()})
        rows.append(row)
        if len(rows) >= args.limit:
            break
    if not rows:
        raise SystemExit('No matching S1/S2 GeoTIFF pairs found. Check extracted folder layout and filenames.')
    dest = Path(args.output)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f'Extracted {len(rows)} REAL paired-image feature records to {dest}')
    print('These are descriptive features, not a trained CNN, fusion model, or RL results.')

if __name__ == '__main__':
    main()
