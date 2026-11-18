"""Read original SEN12MS _s1_ and _s2_ TIFF pairs; no automatic downloads.

Image-derived quantities are descriptive features, not ground-truth information
freshness, satellite communication reliability, or model confidence.
"""
import argparse
import csv
from pathlib import Path
import numpy as np


def find_pairs(dataset_root, limit=100):
    root = Path(dataset_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {root}")
    # Native SEN12MS commonly uses filenames with `_s1_` / `_s2_` tokens.
    s1_files = sorted(p for p in root.rglob('*.tif') if '_s1_' in p.name.lower())
    pairs = []
    for s1 in s1_files:
        s2_name = s1.name.replace('_s1_', '_s2_')
        # Sibling modality folders frequently use the same relative ROI name
        candidates = [s1.with_name(s2_name)]
        candidates += [Path(str(s1).replace('_s1_', '_s2_'))]
        s2 = next((p for p in candidates if p.exists()), None)
        if s2 is not None:
            pairs.append((s1, s2))
        if len(pairs) >= limit:
            break
    return pairs


def stats(path):
    import rasterio
    with rasterio.open(path) as ds:
        data = ds.read(masked=True).astype('float32')
        values = np.asarray(data.filled(np.nan), dtype=np.float32)
        valid = np.isfinite(values)
        n_bands = values.shape[0]
        if not np.any(valid):
            raise ValueError(f'No valid pixels: {path}')
        # finite sample statistics; no assumptions about spectral scales
        band_means=[];band_stds=[]
        for b in range(n_bands):
            a=values[b][valid[b]]
            band_means.append(float(a.mean()) if len(a) else 0.0)
            band_stds.append(float(a.std()) if len(a) else 0.0)
        return dict(bands=n_bands,valid_fraction=float(valid.mean()),
                    mean=float(np.mean(band_means)),std=float(np.mean(band_stds)))


def extract(dataset_root, output, limit=100):
    pairs=find_pairs(dataset_root, limit)
    if not pairs:
        raise ValueError('No matching SEN12MS _s1_ / _s2_ TIFF pairs. Check the extracted folder layout.')
    output=Path(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    fields=['patch','s1_path','s2_path','s1_bands','s2_bands','s1_valid_fraction',
            's2_valid_fraction','s1_mean','s2_mean','s1_std','s2_std']
    rows=[]
    for s1,s2 in pairs:
        a,b=stats(s1),stats(s2)
        rows.append(dict(patch=s1.stem.replace('_s1_','_'),s1_path=str(s1),s2_path=str(s2),
                         s1_bands=a['bands'],s2_bands=b['bands'],
                         s1_valid_fraction=a['valid_fraction'],s2_valid_fraction=b['valid_fraction'],
                         s1_mean=a['mean'],s2_mean=b['mean'],s1_std=a['std'],s2_std=b['std']))
    with output.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    return len(rows)


def load_features(csv_file):
    with Path(csv_file).open(newline='',encoding='utf-8-sig') as f:
        rows=list(csv.DictReader(f))
    if not rows: raise ValueError('Feature CSV is empty')
    required={'s1_valid_fraction','s2_valid_fraction'}
    if not required.issubset(rows[0]): raise ValueError('Feature CSV missing required columns')
    result=[]
    for row in rows:
        vals=[float(row[k]) for k in ('s1_valid_fraction','s2_valid_fraction')]
        if not all(np.isfinite(v) and 0<=v<=1 for v in vals):
            raise ValueError('Invalid fraction; expected finite [0, 1] values')
        result.append(tuple(vals))
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser(description='Extract descriptive features from SEN12MS TIFF pairs')
    parser.add_argument('--dataset',required=True,help='Path to extracted SEN12MS directory')
    parser.add_argument('--output',default='data/processed/sen12ms_features.csv')
    parser.add_argument('--limit',type=int,default=100)
    args=parser.parse_args()
    if args.limit<1: parser.error('--limit must be positive')
    count=extract(args.dataset,args.output,args.limit)
    print(f'Saved {count} genuine S1/S2 patch-pair feature records to {args.output}')
    print('NOTE: only observed-image statistics are real. Freshness/reliability/degradation are simulated.')
