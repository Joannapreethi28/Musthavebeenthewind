"""Three predetermined seeds; select checkpoints using validation, never test."""
import copy
import os
import random
import time
import numpy as np
import pandas as pd
import torch
from torch import nn
from common import ROOT, RESULTS, CONTRACT, digest, provenance, read_json, save_json
from data_validation import require
from models import LSTMForecaster

CONFIG = {'model': 'lstm', 'hidden': 64, 'dropout': .2, 'layers': 2, 'window': 72, 'horizon': 12,
          'batch_size': 256, 'max_epochs': 60, 'patience': 8, 'lr': .001, 'weight_decay': .0001,
          'gradient_clip': 1., 'shuffle': False, 'seeds': [0, 1, 2], 'representative_seed': 0}


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(min(4, os.cpu_count() or 1))


def tensors(data, split: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Materialize only the requested split; 72 historical hours per example."""
    idx = data[split]
    windows = np.lib.stride_tricks.sliding_window_view(data['features'], 72, axis=0).transpose(0, 2, 1)
    x = torch.from_numpy(windows[idx - 71].copy())
    y = torch.from_numpy((data['raw'][idx + 12, 0] - data['raw'][idx, 0]).astype('float32'))
    require(tuple(x.shape[1:]) == (72, 6) and bool(torch.isfinite(x).all()), 'Bad model inputs')
    return x, y


def predict(model: nn.Module, x: torch.Tensor, device: str = 'cpu') -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return torch.cat([model(batch.to(device)).cpu() for batch in x.split(256)]).numpy().astype(float)


def overfit(x: torch.Tensor, y: torch.Tensor, device: str) -> dict:
    seed_all(0)
    model = LSTMForecaster().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    # Fixed, widely spaced training samples exercise diverse observed changes.
    positions = torch.linspace(0, len(x) - 1, 64).long()
    small_x, small_y = x[positions].to(device), y[positions].to(device)
    for step in range(1, 2001):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.mse_loss(model(small_x), small_y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        if step % 25 == 0:
            model.eval()
            with torch.no_grad():
                mse = float(nn.functional.mse_loss(model(small_x), small_y))
            if step % 100 == 0:
                print(f'Overfit step {step}: MSE {mse:.6f}', flush=True)
            if mse < .01:
                print(f'PASS 64-sample overfit: MSE {mse:.6f} after {step} updates', flush=True)
                return {'samples': 64, 'steps': step, 'mse': mse, 'threshold': .01}
    raise ValueError('Overfit gate failed; no full training started')


def run() -> None:
    require((RESULTS / 'baselines.json').exists(), 'Baseline gate must pass first')
    require(not (RESULTS / 'frozen_experiment.json').exists(), 'Experiment frozen: do not tune after final evaluation')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    seed_all(0)
    with np.load(ROOT / 'data/processed/dataset.npz') as data:
        x, y = tensors(data, 'train')
        vx, vy = tensors(data, 'val')
        normalization = {'mean': data['mean'].tolist(), 'std': data['std'].tolist()}
    prov = provenance()
    run_dir = RESULTS / 'runs'
    run_dir.mkdir(exist_ok=True)
    checkpoints = ROOT / 'checkpoints'
    checkpoints.mkdir(exist_ok=True)
    diagnostic = overfit(x, y, device)
    save_json(RESULTS / 'overfit.json', diagnostic)
    records = []
    for seed in CONFIG['seeds']:
        record_path = run_dir / f'lstm_seed{seed}.json'
        checkpoint_path = checkpoints / f'lstm_seed{seed}.pt'
        if record_path.exists():
            record = read_json(record_path)
            require(record['config'] == CONFIG and record['provenance']['dataset_sha256'] == prov['dataset_sha256'], 'Existing run belongs to another experiment')
            for module in ('train.py', 'models.py', 'build_dataset.py', 'baselines.py'):
                require(record['provenance']['source_sha256'][module] == prov['source_sha256'][module], 'Training source changed; inspect existing runs')
            require(digest(checkpoint_path) == record['checkpoint_sha256'], 'Checkpoint changed')
            records.append(record)
            print(f'PASS reuse completed seed {seed}', flush=True)
            continue
        seed_all(seed)
        model = LSTMForecaster().to(device)
        parameters = sum(p.numel() for p in model.parameters())
        require(parameters == 51777, 'Unexpected LSTM size')
        initial = float(np.mean((predict(model, vx, device) - vy.numpy()) ** 2))
        optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'])
        best, stale, history, best_epoch = float('inf'), 0, [], 0
        started = time.perf_counter()
        for epoch in range(1, CONFIG['max_epochs'] + 1):
            tick = time.perf_counter()
            model.train()
            total = 0.
            for begin in range(0, len(x), CONFIG['batch_size']):
                xb, yb = x[begin:begin + 256].to(device), y[begin:begin + 256].to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = nn.functional.mse_loss(model(xb), yb)
                require(bool(torch.isfinite(loss)), 'Nonfinite training loss')
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                total += float(loss.detach()) * len(xb)
            validation = float(np.mean((predict(model, vx, device) - vy.numpy()) ** 2))
            require(np.isfinite(validation), 'Nonfinite validation loss')
            history.append({'epoch': epoch, 'train_mse': total / len(x), 'val_mse': validation,
                            'seconds': time.perf_counter() - tick})
            if validation < best:
                best, best_epoch, stale = validation, epoch, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                torch.save({'state_dict': best_state, 'config': CONFIG, 'seed': seed,
                            'normalization': normalization, 'contract': CONTRACT}, checkpoint_path)
            else:
                stale += 1
            pd.DataFrame(history).to_csv(RESULTS / f'history_lstm_seed{seed}.csv', index=False)
            print(f'Seed {seed} epoch {epoch:02d}: train {total / len(x):.5f}; val {validation:.5f}; {history[-1]["seconds"]:.1f}s; best {best_epoch}', flush=True)
            if stale >= CONFIG['patience']:
                break
        require(best < initial, 'Validation did not improve from initialization')
        model.load_state_dict(best_state)
        expected = predict(model, vx, device)
        restored = LSTMForecaster().to(device)
        restored.load_state_dict(torch.load(checkpoint_path, weights_only=True, map_location=device)['state_dict'])
        actual = predict(restored, vx, device)
        require(np.array_equal(expected, actual), 'Checkpoint reload differs')
        record = {'contract': CONTRACT, 'seed': seed, 'config': CONFIG, 'normalization': normalization,
                  'device': device, 'torch_version': str(torch.__version__), 'numpy_version': np.__version__,
                  'provenance': prov, 'parameter_count': parameters, 'initial_val_mse': initial,
                  'best_val_mse': best, 'best_epoch': best_epoch, 'epochs': len(history),
                  'elapsed_seconds': time.perf_counter() - started, 'checkpoint_sha256': digest(checkpoint_path),
                  'checkpoint_reload_exact': True, 'deterministic_algorithms': True}
        save_json(record_path, record)
        records.append(record)
        print(f'PASS trained and reloaded seed {seed}', flush=True)
    save_json(RESULTS / 'frozen_experiment.json', {'contract': CONTRACT, 'config': CONFIG, 'runs': records,
              'representative_seed': 0, 'optional_transformer': 'Omitted before test evaluation to prioritize the required model and completion time.',
              'selection': 'Default configuration fixed in advance; no tuning grid.', 'provenance': provenance()})
    print('PASS frozen three-seed experiment; final test evaluation is now permitted.', flush=True)


if __name__ == '__main__':
    run()
