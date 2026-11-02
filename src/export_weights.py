"""Export the frozen seed-0 checkpoint so the demo pages can run it in a browser.

Why: the demo must execute the same model that produced the published 2017
scores, with no backend and no build step (DESIGN_BRIEF section 5). Weights are
written as base64 float32 in PyTorch's own layout, which report/demo/lstm.js
decodes directly; tests/test_lstm_js.mjs proves that decoding matches torch.

This reads a checkpoint and never trains, so it cannot disturb the frozen
experiment. It refuses to export a checkpoint whose hash is not the one recorded
in results/metrics.json -- a demo running unidentified weights would be worse
than no demo.
"""
import base64
import hashlib
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CONTRACT, ROOT, RESULTS, read_json, save_json  # noqa: E402

SEED = 0


def require(condition: bool, message: str) -> None:
    """Local copy: data_validation pulls in scipy, which exporting weights does not need."""
    if not condition:
        raise ValueError(message)


def file_sha256(path: Path) -> str:
    """Chunked read rather than hashlib.file_digest, which needs Python 3.11+."""
    sha = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            sha.update(block)
    return sha.hexdigest()


def encode(tensor: torch.Tensor) -> str:
    return base64.b64encode(tensor.detach().contiguous().float().numpy().tobytes()).decode()


def run() -> dict:
    checkpoint_path = ROOT / 'checkpoints' / f'lstm_seed{SEED}.pt'
    require(checkpoint_path.exists(), f'Missing {checkpoint_path}; copy it from the training machine')

    metrics = read_json(RESULTS / 'metrics.json')
    recorded = next(r for r in metrics['runs'] if r['seed'] == SEED)
    actual = file_sha256(checkpoint_path)
    require(actual == recorded['checkpoint_sha256'],
            f'Checkpoint hash {actual} is not the evaluated seed-{SEED} checkpoint')

    checkpoint = torch.load(checkpoint_path, weights_only=True, map_location='cpu')
    require(checkpoint['contract'] == CONTRACT, 'Checkpoint was trained under a different contract')
    state = checkpoint['state_dict']
    hidden = int(checkpoint['config']['hidden'])

    normalization = checkpoint['normalization']
    mean, std = list(normalization['mean']), list(normalization['std'])
    require(len(mean) == 4 and len(std) == 4, 'Expected four continuous channels')

    model = {
        'contract': CONTRACT,
        'seed': SEED,
        'hidden': hidden,
        'inputSize': 6,
        'steps': int(checkpoint['config']['window']),
        'horizonHours': int(checkpoint['config']['horizon']),
        'featureOrder': ['log_flux', 'v_sw', 'bz_gsm', 'dst_nt', 'sin_lt', 'cos_lt'],
        'normalization': {'mean': mean, 'std': std},
        'diurnalMeanDelta': metrics['baseline']['diurnal_means'],
        'layers': [{'weightIh': encode(state[f'lstm.weight_ih_l{i}']),
                    'weightHh': encode(state[f'lstm.weight_hh_l{i}']),
                    'biasIh': encode(state[f'lstm.bias_ih_l{i}']),
                    'biasHh': encode(state[f'lstm.bias_hh_l{i}'])} for i in (0, 1)],
        'head': {'weight': encode(state['head.weight']), 'bias': encode(state['head.bias'])},
        'provenance': {'checkpoint_sha256': actual,
                       'frozen_experiment_sha256': metrics['frozen_experiment_sha256'],
                       'torch_version': torch.__version__},
    }
    save_json(ROOT / 'report/demo/model.json', model)
    parameters = sum(state[k].numel() for k in state)
    print(f'PASS exported seed {SEED}: {parameters} parameters, hidden {hidden}', flush=True)
    print('checkpoint sha256 matches metrics.json:', actual, flush=True)
    return model


if __name__ == '__main__':
    run()
