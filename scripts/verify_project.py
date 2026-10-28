"""Independently validate delivered results without requiring local raw data or weights."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from common import ROOT, RESULTS, digest, read_json, save_json
from baselines import scores
from data_validation import require
from render_report import result_table


def run() -> None:
    metrics = read_json(RESULTS / 'metrics.json')
    require(metrics['config']['seeds'] == [0, 1, 2], 'Expected the three predetermined seeds')
    require(metrics['overfit']['mse'] < .01, 'Overfit acceptance gate failed')
    require(metrics['frozen_experiment_sha256'] == digest(RESULTS / 'frozen_experiment.json'), 'Frozen experiment changed after evaluation')
    checks = []
    for split, start, end in [('val', '2016', '2017'), ('test', '2017', '2018')]:
        frame = pd.read_csv(RESULTS / f'preds_{split}.csv')
        times = {name: pd.to_datetime(frame[name], utc=True) for name in ['time_utc', 'issue_time_utc', 'target_end_utc']}
        require(times['time_utc'].is_unique and times['time_utc'].is_monotonic_increasing, 'Duplicated or unordered predictions')
        require(((times['target_end_utc'] - times['issue_time_utc']) == pd.Timedelta(hours=12)).all(), 'Wrong forecast horizon')
        require(((times['target_end_utc'] - times['time_utc']) == pd.Timedelta(hours=1)).all(), 'Wrong hour labels')
        require((times['issue_time_utc'] >= pd.Timestamp(start, tz='UTC')).all(), 'Issue precedes split')
        require((times['target_end_utc'] < pd.Timestamp(end, tz='UTC')).all(), 'Target crosses split boundary')
        require(len(frame) == metrics['split_sizes'][split], 'Wrong sample count')
        for name in ['persistence', 'persistence_diurnal']:
            measured = scores(frame['obs'].to_numpy(), frame[name].to_numpy())
            recorded = metrics['models'][name][split]
            require(all(abs(measured[key] - recorded[key]) < 1e-12 for key in ['pe', 'rmse']), 'Baseline export disagrees')
        values = []
        seed_scores = metrics['models']['lstm'][split]['seeds']
        recorded_by_seed = {entry['seed']: entry for entry in seed_scores}
        require(len(seed_scores) == len(recorded_by_seed) == 3 and set(recorded_by_seed) == {0, 1, 2}, 'Missing or duplicate seed scores')
        for seed in metrics['config']['seeds']:
            seed_frame = pd.read_csv(RESULTS / f'preds_{split}_lstm_seed{seed}.csv')
            require(frame[['time_utc', 'issue_time_utc', 'target_end_utc', 'obs']].equals(seed_frame[['time_utc', 'issue_time_utc', 'target_end_utc', 'obs']]), 'Models score different samples')
            measured = scores(seed_frame['obs'].to_numpy(), seed_frame['lstm'].to_numpy())
            recorded = recorded_by_seed[seed]
            require(all(abs(measured[key] - recorded[key]) < 1e-12 for key in ['pe', 'rmse']), 'LSTM export disagrees')
            if seed == 0:
                np.testing.assert_array_equal(frame['lstm'], seed_frame['lstm'])
            values.append(measured)
        for metric in ['pe', 'rmse']:
            require(abs(np.mean([v[metric] for v in values]) - metrics['models']['lstm'][split]['aggregate'][metric]['mean']) < 1e-12, 'Wrong mean')
            require(abs(np.std([v[metric] for v in values], ddof=1) - metrics['models']['lstm'][split]['aggregate'][metric]['std']) < 1e-12, 'Wrong sample SD')
        checks.append(f'{split}: common timestamps, observed targets, forecast horizon, per-seed and baseline scores, mean/sample SD')
    require(result_table(metrics, html=False) in (ROOT / 'README.md').read_text(encoding='utf-8'), 'README results stale')
    manifest = read_json(RESULTS / 'report_manifest.json')
    require(manifest['metrics_sha256'] == digest(RESULTS / 'metrics.json'), 'Report metrics changed')
    require(manifest['index_sha256'] == digest(ROOT / 'report/index.html'), 'Report HTML changed')
    require(manifest['style_sha256'] == digest(ROOT / 'report/style.css'), 'Report styles changed')
    require(manifest['readme_sha256'] == digest(ROOT / 'README.md'), 'README changed after generation')
    require(manifest['data_sha256'] == digest(ROOT / 'report/data.json'), 'Report data changed')
    for filename, expected in manifest['figure_sha256'].items():
        require(digest(RESULTS / 'figs' / filename) == expected, 'Figure changed')
    browser = read_json(RESULTS / 'report_checks.json')
    require(browser['status'] == 'passed' and browser['index_sha256'] == digest(ROOT / 'report/index.html'), 'Browser checks missing or stale')
    require(browser['style_sha256'] == digest(ROOT / 'report/style.css'), 'Browser styling checks stale')
    for filename, entry in browser['captures'].items():
        require(digest(ROOT / 'report/screens' / filename) == entry['sha256'], 'Screenshot changed')
    checks.append('README/report values, figures, screenshots, offline browser and mobile layout')
    if (ROOT / 'data/processed/dataset.npz').exists():
        require(digest(ROOT / 'data/processed/dataset.npz') == metrics['provenance']['dataset_sha256'], 'Local dataset differs from evaluated data')
        with np.load(ROOT / 'data/processed/dataset.npz') as data:
            raw, features, available = data['raw'], data['features'], data['available']
            for split in ['train', 'val', 'test']:
                idx = data[split]
                require(np.isfinite(raw[idx, 0]).all() and np.isfinite(raw[idx + 12, 0]).all(), 'Invented anchor or target')
                for i in idx:
                    require(np.isfinite(features[i - 71:i + 1]).all(), 'Missing history')
                    require(available[i - 71:i + 1].max() <= i, 'Future support in input')
            train_ends = data['times'].astype('datetime64[h]') + np.timedelta64(1, 'h')
            fit_mask = (train_ends >= np.datetime64('2012')) & (train_ends < np.datetime64('2016'))
            from build_dataset import fill_short_gaps
            periods = np.where(train_ends < np.datetime64('2016'), 0, np.where(train_ends < np.datetime64('2017'), 1, 2))
            modified = raw.copy()
            modified[~fit_mask] += 50000
            filled, _ = fill_short_gaps(modified, periods)
            np.testing.assert_allclose(np.nanmean(filled[fit_mask], axis=0), data['mean'], rtol=0, atol=1e-10)
            np.testing.assert_allclose(np.nanstd(filled[fit_mask], axis=0), data['std'], rtol=0, atol=1e-10)
        checks.append('Every local sample has observed labels and causal complete history; train statistics invariant to future perturbation')
    save_json(RESULTS / 'acceptance.json', {'status': 'passed', 'checks': checks,
              'metrics_sha256': digest(RESULTS / 'metrics.json'), 'test_samples': metrics['split_sizes']['test'],
              'overfit_mse': metrics['overfit']['mse']})
    for check in checks:
        print('PASS', check, flush=True)


if __name__ == '__main__':
    run()
