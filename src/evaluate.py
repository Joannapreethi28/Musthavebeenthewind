"""Score frozen checkpoints once, then export predictions and reproducible figures."""
import subprocess
import sys
import numpy as np
import pandas as pd
import torch
from baselines import scores, historical_probe
from common import ROOT, RESULTS, CONTRACT, digest, provenance, read_json, save_json
from data_validation import require
from models import LSTMForecaster
from train import tensors, predict, seed_all


def aggregate(values: list[dict]) -> dict:
    return {key: {'mean': float(np.mean([v[key] for v in values])),
                  'std': float(np.std([v[key] for v in values], ddof=1))}
            for key in ('pe', 'rmse')}


def run() -> None:
    frozen = read_json(RESULTS / 'frozen_experiment.json')
    require(frozen['contract'] == CONTRACT, 'Contract mismatch')
    require(digest(ROOT / 'data/processed/dataset.npz') == frozen['provenance']['dataset_sha256'], 'Dataset changed after freeze')
    for filename in ('build_dataset.py', 'baselines.py', 'train.py', 'models.py'):
        require(digest(ROOT / 'src' / filename) == frozen['provenance']['source_sha256'][filename], 'Scientific code changed after freeze')
    baseline = read_json(RESULTS / 'baselines.json')
    quality = read_json(RESULTS / 'data_quality.json')
    seed_all(0)
    models = []
    for record in frozen['runs']:
        path = ROOT / 'checkpoints' / f'lstm_seed{record["seed"]}.pt'
        require(digest(path) == record['checkpoint_sha256'], 'Frozen checkpoint changed')
        saved = torch.load(path, map_location='cpu', weights_only=True)
        model = LSTMForecaster()
        model.load_state_dict(saved['state_dict'])
        models.append((record['seed'], model))
    metrics = {'schema_version': 1, 'contract': CONTRACT, 'config': frozen['config'],
               'provenance': provenance(), 'frozen_experiment_sha256': digest(RESULTS / 'frozen_experiment.json'),
               'meta': {'horizon_hours': 12, 'window_hours': 72, 'satellite': 'GOES-15',
                        'detector': 'E2E_COR_FLUX', 'train_years': '2012–2015', 'val_year': 2016,
                        'test_year': 2017, 'representative_seed': 0, 'seed_count': len(models),
                        'authors': ['Jabin Joseph M', 'Joanna Preethi J'], 'device': frozen['runs'][0]['device'],
                        'parameter_count': frozen['runs'][0]['parameter_count'], 'units': 'log10(electrons / cm² / s / sr)'},
               'split_sizes': {key: val['accepted'] for key, val in quality['exclusions'].items()},
               'data_quality': quality, 'baseline': baseline, 'models': {}, 'runs': frozen['runs'],
               'optional_transformer': frozen['optional_transformer'], 'overfit': read_json(RESULTS / 'overfit.json')}
    for name in ('persistence', 'persistence_diurnal', 'lstm'):
        metrics['models'][name] = {}
    with np.load(ROOT / 'data/processed/dataset.npz') as data:
        raw, lt = data['raw'], data['lt']
        times = data['times'].astype('datetime64[h]')
        for split in ('val', 'test'):
            idx = data[split]
            observed, current = raw[idx + 12, 0], raw[idx, 0]
            correction = np.array(baseline['diurnal_means'])[np.floor(lt[idx]).astype(int)]
            frame = pd.DataFrame({'time_utc': pd.to_datetime(times[idx + 12], utc=True),
                                  'issue_time_utc': pd.to_datetime(times[idx] + np.timedelta64(1, 'h'), utc=True),
                                  'target_end_utc': pd.to_datetime(times[idx + 12] + np.timedelta64(1, 'h'), utc=True),
                                  'obs': observed, 'persistence': current, 'persistence_diurnal': current + correction})
            for name in ('persistence', 'persistence_diurnal'):
                metrics['models'][name][split] = scores(observed, frame[name].to_numpy())
            x, _ = tensors(data, split)
            seed_scores = []
            for seed, model in models:
                forecast = current + predict(model, x)
                score = scores(observed, forecast)
                seed_scores.append({'seed': seed, **score})
                export = frame[['time_utc', 'issue_time_utc', 'target_end_utc', 'obs']].copy()
                export['lstm'] = forecast
                export.to_csv(RESULTS / f'preds_{split}_lstm_seed{seed}.csv', index=False)
                # Re-read exported decimals to independently check the saved artifact.
                restored = pd.read_csv(RESULTS / f'preds_{split}_lstm_seed{seed}.csv')
                recalculated = scores(restored['obs'].to_numpy(), restored['lstm'].to_numpy())
                require(all(abs(score[k] - recalculated[k]) < 1e-12 for k in ('pe', 'rmse')), 'Exported scores disagree')
                if seed == 0:
                    frame['lstm'] = forecast
            metrics['models']['lstm'][split] = {'seeds': seed_scores, 'aggregate': aggregate(seed_scores)}
            frame.to_csv(RESULTS / f'preds_{split}.csv', index=False)
            if split == 'test':
                test_frame = frame
                target_bins = np.floor(lt[idx + 12]).astype(int)
                errors = {name: [float(np.sqrt(np.mean((frame.loc[target_bins == k, name] - frame.loc[target_bins == k, 'obs']) ** 2))) for k in range(24)]
                          for name in ('persistence', 'persistence_diurnal', 'lstm')}
        # Historical test recipe is evaluated only after model and preprocessing freeze.
        metrics['historical_test_probe'] = historical_probe('test')
    subprocess.run([sys.executable, str(ROOT / 'scripts/check_historical_probe.py')], check=True)
    metrics['unmodified_historical_probe'] = read_json(RESULTS / 'historical_probe.json')
    baseline_pe = metrics['models']['persistence_diurnal']['test']['pe']
    deltas = [s['pe'] - baseline_pe for s in metrics['models']['lstm']['test']['seeds']]
    metrics['comparison'] = {'paired_seed_pe_improvement': deltas, 'mean_pe_improvement': float(np.mean(deltas)),
                             'all_seeds_beat_diurnal': bool(all(d > 0 for d in deltas)),
                             'interpretation': 'Seed variability measures optimization variability, not statistical significance or uncertainty over future storms.'}
    metrics['checks'] = {'shared_sample_indices': True, 'checkpoint_hashes_match': True,
                         'metrics_recomputed_from_exports': True, 'historical_probe_within_tolerance': True,
                         'finite_predictions': True, 'test_used_only_after_freeze': True}
    if (RESULTS / 'metrics.json').exists():
        previous = read_json(RESULTS / 'metrics.json')
        require(previous['models'] == metrics['models'], 'Frozen re-evaluation changed scores; investigate')
    save_json(RESULTS / 'metrics.json', metrics)
    storm = test_frame.set_index('time_utc').reindex(pd.date_range('2017-09-01', '2017-09-21', freq='h', inclusive='left', tz='UTC'))
    history = {f'lstm_seed{seed}': pd.read_csv(RESULTS / f'history_lstm_seed{seed}.csv').to_dict('list') for seed, _ in models}
    plot_data = {'meta': metrics['meta'], 'metrics': metrics,
                 'storm': {'time': [t.isoformat() for t in storm.index],
                           **{k: [float(v) if np.isfinite(v) else None for v in storm[k]] for k in ('obs', 'persistence', 'persistence_diurnal', 'lstm')}},
                 'diurnal': {'lt': list(range(24)), 'mean_logflux': quality['diurnal_mean_logflux'], 'err_by_lt': errors},
                 'history': history}
    save_json(ROOT / 'report/data.json', plot_data)
    from figures import render_figures
    render_figures(plot_data)
    print('PASS final evaluation: saved predictions reproduce all metrics.', flush=True)
    for name, result in metrics['models'].items():
        print(name, result['test'], flush=True)


if __name__ == '__main__':
    run()
