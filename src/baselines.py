"""Fit the daily-cycle correction on train samples and score common timestamps."""
import argparse
import numpy as np
from scipy.io import netcdf_file
from common import ROOT, RESULTS, CONTRACT, save_json
from data_validation import require


def scores(observed: np.ndarray, predicted: np.ndarray) -> dict:
    observed, predicted = np.asarray(observed, float), np.asarray(predicted, float)
    require(observed.shape == predicted.shape and observed.size > 1, 'Scoring shapes differ or are empty')
    require(np.isfinite(observed).all() and np.isfinite(predicted).all(), 'Nonfinite scored values')
    sse = float(np.sum((observed - predicted) ** 2))
    denominator = float(np.sum((observed - observed.mean()) ** 2))
    require(denominator > 0, 'PE undefined for constant observations')
    return {'pe': 1 - sse / denominator, 'rmse': float(np.sqrt(sse / observed.size)), 'n': int(observed.size)}


def fit_diurnal(raw: np.ndarray, lt: np.ndarray, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    bins = np.floor(lt[indices]).astype(int)
    delta = raw[indices + 12, 0] - raw[indices, 0]
    counts = np.bincount(bins, minlength=24)
    require((counts > 0).all(), 'Empty train local-time bin')
    return np.bincount(bins, weights=delta, minlength=24) / counts, counts


def historical_probe(split: str = 'val') -> dict:
    """Independently reproduce the original E/W recipe without importing the probe."""
    combined, timestamps = [], []
    final_year = 2016 if split == 'val' else 2017
    for path in sorted((ROOT / 'data/raw/goes15').glob('*.nc')):
        year = int(path.name.split('_')[-2][:4])
        if year > final_year:
            continue
        with netcdf_file(path, 'r', mmap=False) as dataset:
            timestamps.append(dataset.variables['time_tag'][:].astype(float) // 3600000)
            sensors = []
            for sensor in ('E2E', 'E2W'):
                f = dataset.variables[sensor + '_COR_FLUX'][:].astype(float)
                q = dataset.variables[sensor + '_QUAL_FLAG'][:]
                sensors.append(np.where((f > 0) & (q == 0), f, np.nan))
            stack = np.stack(sensors)
            count = np.isfinite(stack).sum(axis=0)
            combined.append(np.divide(np.nansum(stack, axis=0), count, out=np.full(len(count), np.nan), where=count > 0))
    samples = np.concatenate(combined)
    absolute_hours = np.concatenate(timestamps).astype('int64')
    hour_index = absolute_hours - np.datetime64('2012-01-01T00', 'h').astype('int64')
    count = np.bincount(hour_index, weights=np.isfinite(samples))
    sums = np.bincount(hour_index, weights=np.nan_to_num(samples))
    lf = np.log10(np.divide(sums, count, out=np.full(len(count), np.nan), where=count >= 6))
    time = np.datetime64('2012-01-01T00', 'h') + np.arange(len(lf)).astype('timedelta64[h]')
    year = time.astype('datetime64[Y]').astype(int) + 1970
    bins = np.floor((np.arange(len(lf)) % 24 + .5 - 135.5 / 15) % 24).astype(int)
    x, y = lf[:-12], lf[12:]
    finite = np.isfinite(x) & np.isfinite(y)
    train = (year[:-12] <= 2015) & finite
    means = np.array([np.mean((y - x)[train & (bins[:-12] == k)]) for k in range(24)])
    selected = (year[:-12] == final_year) & finite
    result = {'persistence': scores(y[selected], x[selected]),
              'persistence_diurnal': scores(y[selected], x[selected] + means[bins[:-12][selected]])}
    expected = {'val': [(0.354, .903), (.706, .609)], 'test': [(.293, .856), (.683, .573)]}
    for measured, (pe, rmse) in zip(result.values(), expected[split]):
        require(abs(measured['pe'] - pe) <= .01 and abs(measured['rmse'] - rmse) <= .01, 'Historical baseline differs; investigate before proceeding')
    return result


def run() -> dict:
    with np.load(ROOT / 'data/processed/dataset.npz') as data:
        raw, lt, idx = data['raw'], data['lt'], data['val']
        means, counts = fit_diurnal(raw, lt, data['train'])
        current, observed = raw[idx, 0], raw[idx + 12, 0]
        result = {'contract': CONTRACT, 'diurnal_means': means.tolist(), 'diurnal_counts': counts.tolist(),
                  'val': {'persistence': scores(observed, current),
                          'persistence_diurnal': scores(observed, current + means[np.floor(lt[idx]).astype(int)])},
                  'historical_val_probe': historical_probe()}
    save_json(RESULTS / 'baselines.json', result)
    print('PASS validation baselines and historical probe:', result['val'], flush=True)
    return result


if __name__ == '__main__':
    run()
