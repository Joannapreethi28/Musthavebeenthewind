# Shared model and input data

The files required for the existing demo are committed on branch `joanna`:

| File | Purpose |
|---|---|
| `checkpoints/lstm_seed0.pt` | Display model: state dict, frozen training normalization, configuration and seed. |
| `checkpoints/lstm_seed1.pt`, `checkpoints/lstm_seed2.pt` | The other two frozen runs used in the published results. |
| `data/processed/dataset.npz` | Normalized hourly features, observed channels, timestamps, availability indices, split indices and training mean/std. |
| `data/processed/hourly.csv` | Human-readable hourly inputs before normalization, including causal short-gap fills. Use the NPZ split indices to select valid model samples. |

From a clean checkout, fetch and switch to `joanna`, then pull with `git pull --ff-only origin joanna`. If continuing work on another branch, merge `origin/joanna` into that branch. Run the following from the repository root (the local `source/` folder), using an environment with the dependencies in `results/setup/requirements.freeze.txt`.

`dataset.npz` stores the hourly feature matrix and accepted sample indices, rather than duplicated windows. `train.tensors(data, 'test')` materializes the correct `(samples, 72, 6)` windows in the same order as `results/preds_test.csv`. Its features are already normalized; do not normalize them again. For a smaller demo interval, match CSV timestamps and apply the same row selection to the windows. `time_utc` labels the target hour's start; `issue_time_utc` is when the forecast is issued.

This example checks every shared checkpoint hash and the dataset hash, reloads seed zero, and compares fresh inference with the published predictions. It needs no raw archives and writes no results:

```python
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path.cwd() / 'src'))
from models import LSTMForecaster
from train import predict, seed_all, tensors

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

frozen = json.loads(Path('results/frozen_experiment.json').read_text())
assert sha256('data/processed/dataset.npz') == frozen['provenance']['dataset_sha256']
for run in frozen['runs']:
    assert sha256(f"checkpoints/lstm_seed{run['seed']}.pt") == run['checkpoint_sha256']

seed_all(0)
saved = torch.load('checkpoints/lstm_seed0.pt', map_location='cpu', weights_only=True)
assert saved['seed'] == 0
assert saved['config'] == frozen['config']
assert saved['contract'] == frozen['contract']
model = LSTMForecaster()
model.load_state_dict(saved['state_dict'])

with np.load('data/processed/dataset.npz', allow_pickle=False) as data:
    assert saved['normalization'] == {'mean': data['mean'].tolist(), 'std': data['std'].tolist()}
    windows, _ = tensors(data, 'test')
    indices = data['test']
    forecast = data['raw'][indices, 0] + predict(model, windows)
    issue_times = pd.to_datetime(data['times'][indices].astype('datetime64[h]') + np.timedelta64(1, 'h'), utc=True)

published = pd.read_csv('results/preds_test.csv')
assert (issue_times == pd.to_datetime(published['issue_time_utc'], utc=True)).all()
# Float tolerance allows minor numerical differences across supported CPU builds.
np.testing.assert_allclose(forecast, published['lstm'].to_numpy(), rtol=0, atol=1e-5)
print(f'PASS: {len(forecast):,} forecasts; input shape {tuple(windows.shape)}')
print(f'Maximum absolute difference: {np.max(np.abs(forecast - published["lstm"].to_numpy())):.3g}')
```

`predict` sets evaluation mode and disables gradients. The model predicts a change in log10 flux; adding the observed current log10 flux gives the forecast stored in the CSV. The observed targets, baseline forecasts and display-seed forecasts are already in `results/preds_test.csv`; the training-only diurnal corrections are in `results/metrics.json` under `baseline.diurnal_means`.

Run `python scripts/verify_project.py` for the existing dataset/results/report audit. Do not run `train.py` or rebuild the shared dataset for this demo. Full `evaluate.py` also performs historical diagnostics on raw archives, so it requires the documented raw-data download first. Use the README's separate-checkout workflow for a new experiment.
