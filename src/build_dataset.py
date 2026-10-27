"""Build observed targets and causal histories; never invent a forecast label."""
import numpy as np
import pandas as pd
from scipy.io import netcdf_file
from common import ROOT, RESULTS, CONTRACT, digest, read_json, save_json
from data_validation import require

CHANNELS = ['log_flux', 'v_sw', 'bz_gsm', 'dst_nt']
PERIODS = {'train': ('2012', '2016'), 'val': ('2016', '2017'), 'test': ('2017', '2018')}


def hourly_flux(flux: np.ndarray, quality: np.ndarray, hour_index=None, n_hours=None) -> tuple[np.ndarray, int]:
    """Average positive, quality-zero linear samples before flooring and log10."""
    valid = np.isfinite(flux) & (flux > 0) & (quality == 0)
    if hour_index is None:
        hour_index = np.arange(len(flux)) // 12
        n_hours = len(flux) // 12
    counts = np.bincount(hour_index, weights=valid, minlength=n_hours)
    sums = np.bincount(hour_index, weights=np.where(valid, flux, 0), minlength=n_hours)
    means = np.divide(sums, counts, out=np.full(len(counts), np.nan), where=counts >= 6)
    floored = int(np.sum(means < 1))
    return np.log10(np.maximum(means, 1)), floored


def fill_short_gaps(values: np.ndarray, periods: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Record right-end availability for complete bounded runs of at most 3 hours."""
    output = values.copy()
    available = np.broadcast_to(np.arange(len(values))[:, None], values.shape).copy()
    for channel in range(values.shape[1]):
        missing = ~np.isfinite(values[:, channel])
        edges = np.flatnonzero(np.diff(np.r_[False, missing, False]))
        for start, end in edges.reshape(-1, 2):
            if start == 0 or end == len(values) or end - start > 3:
                continue
            if periods[start - 1] != periods[end]:
                continue
            output[start:end, channel] = np.linspace(values[start - 1, channel], values[end, channel], end - start + 2)[1:-1]
            available[start:end, channel] = end
    return output, available


def sample_indices(raw: np.ndarray, filled: np.ndarray, available: np.ndarray,
                   times: np.ndarray) -> tuple[dict, dict]:
    """Issue at hour end; require all labels and interpolation support to be legal."""
    n = len(raw)
    issue_times = times + np.timedelta64(1, 'h')
    target_ends = issue_times + np.timedelta64(12, 'h')
    history_finite = np.zeros(n, dtype=bool)
    history_known = np.zeros(n, dtype=bool)
    history_finite[71:] = np.lib.stride_tricks.sliding_window_view(np.isfinite(filled).all(axis=1), 72).all(axis=1)
    history_known[71:] = np.lib.stride_tricks.sliding_window_view(available.max(axis=1), 72).max(axis=1) <= np.arange(71, n)
    anchor = np.isfinite(raw[:, 0])
    target = np.r_[np.isfinite(raw[12:, 0]), np.zeros(12, dtype=bool)]
    indices, counts = {}, {}
    for name, (start, end) in PERIODS.items():
        remaining = (issue_times >= np.datetime64(start)) & (issue_times < np.datetime64(end))
        counts[name] = {'candidate_issue_hours': int(remaining.sum())}
        rules = {'target_outside_split': target_ends < np.datetime64(end),
                 'insufficient_history': np.arange(n) >= 71,
                 'unobserved_anchor': anchor, 'unobserved_target': target,
                 'missing_history': history_finite, 'future_interpolation': history_known}
        for reason, accepted in rules.items():
            counts[name][reason] = int((remaining & ~accepted).sum())
            remaining &= accepted
        indices[name] = np.flatnonzero(remaining)
        counts[name]['accepted'] = int(remaining.sum())
        require(len(indices[name]) > 0, f'No samples in {name}')
    return indices, counts


def run() -> dict:
    manifest = read_json(RESULTS / 'data_manifest.json')
    require(manifest['status'] == 'complete', 'Download gate has not passed')
    for name, entry in manifest['files'].items():
        require(digest(ROOT / name) == entry['sha256'], f'Raw file changed: {name}')
    times = np.arange(np.datetime64('2012-01-01T00', 'h'), np.datetime64('2018-01-01T00', 'h'))
    require(len(times) == 52608, 'Unexpected hourly grid')
    flux = np.full(len(times), np.nan)
    floored = 0
    for path in sorted((ROOT / 'data/raw/goes15').glob('*.nc')):
        with netcdf_file(path, 'r', mmap=False) as data:
            hour_index = (data.variables['time_tag'][:].astype(float) // 3600000).astype('int64') - times[0].astype('int64')
            hours, floor_count = hourly_flux(data.variables['E2E_COR_FLUX'][:].astype(float),
                                             data.variables['E2E_QUAL_FLAG'][:].copy(), hour_index, len(times))
            flux[np.unique(hour_index)] = hours[np.unique(hour_index)]
            floored += floor_count
    omni = np.concatenate([np.loadtxt(path) for path in sorted((ROOT / 'data/raw/omni2').glob('*.dat'))])
    raw = np.column_stack([flux, omni[:, 24], omni[:, 16], omni[:, 40]])
    for column, fill in ((1, 9999), (2, 999.9), (3, 99999)):
        raw[raw[:, column] == fill, column] = np.nan
    require(len(raw) == len(times), 'GOES/OMNI coverage mismatch')
    issue = times + np.timedelta64(1, 'h')
    periods = np.where(issue < np.datetime64('2016'), 0, np.where(issue < np.datetime64('2017'), 1, 2))
    filled, available = fill_short_gaps(raw, periods)
    indices, exclusions = sample_indices(raw, filled, available, times)
    train_hours = (issue >= np.datetime64('2012')) & (issue < np.datetime64('2016'))
    mean, std = np.nanmean(filled[train_hours], axis=0), np.nanstd(filled[train_hours], axis=0)
    require(np.isfinite(mean).all() and (std > 0).all(), 'Invalid train normalization')
    lt = ((np.arange(len(times)) % 24) + .5 - 135.5 / 15) % 24
    features = np.column_stack([(filled - mean) / std, np.sin(2 * np.pi * lt / 24), np.cos(2 * np.pi * lt / 24)]).astype('float32')
    folder = ROOT / 'data/processed'
    folder.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(filled, columns=CHANNELS)
    frame.insert(0, 'time_utc', pd.to_datetime(times, utc=True))
    frame['lt_hours'] = lt
    frame.to_csv(folder / 'hourly.csv', index=False)
    np.savez_compressed(folder / 'dataset.npz', features=features, raw=raw, available=available,
                        times=times.astype('int64'), lt=lt, mean=mean, std=std, **indices)
    lt_bin = np.floor(lt).astype(int)
    diurnal = [float(np.nanmean(raw[train_hours & (lt_bin == k), 0])) for k in range(24)]
    coverage = {str(y): float(np.isfinite(raw[times.astype('datetime64[Y]') == np.datetime64(str(y)), 0]).mean()) for y in range(2012, 2018)}
    audit = {'contract': CONTRACT, 'rows': len(times), 'feature_order': CHANNELS + ['sin_lt', 'cos_lt'],
             'observed_flux_coverage': coverage, 'floored_hours': floored,
             'diurnal_mean_logflux': diurnal, 'diurnal_amplitude': max(diurnal) - min(diurnal),
             'diurnal_peak_bin': int(np.argmax(diurnal)), 'exclusions': exclusions,
             'normalization': {'mean': mean.tolist(), 'std': std.tolist(), 'fit': 'unique train-available hours'},
             'dst_2015_min': float(np.nanmin(raw[times.astype('datetime64[Y]') == np.datetime64('2015'), 3])),
             'original_probe_thresholds': 'Historical E/W thresholds are diagnostic only; E-only, observed labels, causal complete windows define v1.',
             'data_manifest_sha256': digest(RESULTS / 'data_manifest.json')}
    require(audit['dst_2015_min'] == -234, 'Dst sanity check failed')
    save_json(RESULTS / 'data_quality.json', audit)
    print('PASS dataset:', len(times), 'hourly rows;', {k: len(v) for k, v in indices.items()}, flush=True)
    print('Observed E-only coverage:', coverage, flush=True)
    print('Diurnal amplitude / peak:', audit['diurnal_amplitude'], audit['diurnal_peak_bin'], flush=True)
    return audit


if __name__ == '__main__':
    run()
