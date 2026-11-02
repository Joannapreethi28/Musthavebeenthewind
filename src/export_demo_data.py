"""Export the 2017 test series the demo pages explore.

Why a series and not windows: adjacent samples overlap by 71 of their 72 hours,
so shipping 8,262 materialized windows would cost ~14 MB for data the browser
can slice itself from ~9k rows. Storing the series once also lets the demo edit
one hour and have every affected forecast update, which is the whole point.

Both the normalized features (what the model eats) and the physical channels
(what a scientist reasons about) are exported, because the perturbation UI must
present km/s and nT, not z-scores.
"""
import base64
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, RESULTS, read_json, save_json  # noqa: E402
from export_weights import file_sha256, require  # noqa: E402

WINDOW, HORIZON = 72, 12


def encode(array: np.ndarray, dtype: str) -> str:
    return base64.b64encode(np.ascontiguousarray(array, dtype=dtype).tobytes()).decode()


def run() -> dict:
    frozen = read_json(RESULTS / 'frozen_experiment.json')
    dataset_path = ROOT / 'data/processed/dataset.npz'
    require(dataset_path.exists(), f'Missing {dataset_path}')
    require(file_sha256(dataset_path) == frozen['provenance']['dataset_sha256'],
            'dataset.npz is not the evaluated dataset')

    with np.load(dataset_path, allow_pickle=False) as data:
        indices = data['test']
        # The demo lets a scientist drag inputs anywhere; it must be able to say
        # when a value leaves the range the model was actually fitted on, so the
        # training spread travels with the data rather than being guessed in JS.
        train_rows = data['train']
        mean, std = data['mean'], data['std']
        train_physical = data['features'][train_rows][:, :4] * std + mean
        train_stats = [{'min': float(column.min()), 'p01': float(np.percentile(column, 1)),
                        'p50': float(np.percentile(column, 50)), 'p99': float(np.percentile(column, 99)),
                        'max': float(column.max())} for column in train_physical.T]
        start = int(indices.min()) - (WINDOW - 1)
        stop = int(indices.max()) + HORIZON + 1
        require(start >= 0 and stop <= len(data['times']), 'Test window range escapes the grid')
        features = data['features'][start:stop]
        raw = data['raw'][start:stop]
        lt = data['lt'][start:stop]
        times = data['times'][start:stop]
        relative = indices - start

    published = pd.read_csv(RESULTS / 'preds_test.csv')
    require(len(published) == len(indices), 'Prediction count does not match test indices')

    series = {
        'contract': frozen['contract'],
        'startRow': start,
        'rows': int(stop - start),
        'window': WINDOW,
        'horizonHours': HORIZON,
        'startTimeUtc': str(np.datetime64(int(times[0]), 'h')),
        'channelOrder': ['log_flux', 'v_sw', 'bz_gsm', 'dst_nt'],
        'normalization': {'mean': mean.tolist(), 'std': std.tolist()},
        'trainStats': train_stats,
        'features': encode(features, 'float32'),
        'raw': encode(raw, 'float32'),
        'lt': encode(lt, 'float32'),
        'samples': {
            'count': int(len(indices)),
            'row': encode(relative, 'int32'),
            'obs': encode(published['obs'].to_numpy(), 'float32'),
            'persistence': encode(published['persistence'].to_numpy(), 'float32'),
            'persistenceDiurnal': encode(published['persistence_diurnal'].to_numpy(), 'float32'),
            'lstm': encode(published['lstm'].to_numpy(), 'float32'),
        },
        'provenance': {'dataset_sha256': frozen['provenance']['dataset_sha256'],
                       'frozen_experiment_sha256': read_json(RESULTS / 'metrics.json')['frozen_experiment_sha256']},
    }
    save_json(ROOT / 'report/demo/series.json', series)
    size = (ROOT / 'report/demo/series.json').stat().st_size
    print(f'PASS exported {series["rows"]} rows, {series["samples"]["count"]} samples, {size / 1024:.0f} KB', flush=True)
    print(f'first issue row {start + WINDOW - 1}, series starts {series["startTimeUtc"]}Z', flush=True)
    return series


if __name__ == '__main__':
    run()
