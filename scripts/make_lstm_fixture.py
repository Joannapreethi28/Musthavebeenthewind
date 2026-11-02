"""Generate a torch-truth fixture so the browser LSTM can be proved correct offline.

Why: report/demo/lstm.js re-implements nn.LSTM by hand. A wrong gate order or a
transposed weight matrix still yields smooth, believable forecasts, so the only
honest check is to run torch itself on random weights and require the JS to
reproduce it. Random weights beat the real checkpoint here because they exercise
every gate instead of whatever regime training happened to settle in.
"""
import base64
import json
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))
from models import LSTMForecaster  # noqa: E402

HIDDEN, INPUT_SIZE, STEPS, CASES = 64, 6, 72, 8


def encode(tensor: torch.Tensor) -> str:
    """Base64 of contiguous float32 bytes, the layout report/demo/lstm.js decodes."""
    return base64.b64encode(tensor.detach().contiguous().float().numpy().tobytes()).decode()


def build() -> dict:
    torch.manual_seed(20261102)
    model = LSTMForecaster(hidden=HIDDEN, dropout=.2)
    # Random init leaves biases near zero, which would hide a bias-row mix-up.
    for parameter in model.parameters():
        torch.nn.init.uniform_(parameter, -.6, .6)
    model.eval()

    state = model.state_dict()
    layers = [{'weightIh': encode(state[f'lstm.weight_ih_l{i}']),
               'weightHh': encode(state[f'lstm.weight_hh_l{i}']),
               'biasIh': encode(state[f'lstm.bias_ih_l{i}']),
               'biasHh': encode(state[f'lstm.bias_hh_l{i}'])} for i in (0, 1)]

    windows = torch.randn(CASES, STEPS, INPUT_SIZE)
    with torch.no_grad():
        deltas = model(windows)
    return {'hidden': HIDDEN, 'inputSize': INPUT_SIZE, 'steps': STEPS,
            'torch_version': torch.__version__,
            'layers': layers,
            'head': {'weight': encode(state['head.weight']), 'bias': encode(state['head.bias'])},
            'cases': [{'window': encode(windows[i]), 'delta': float(deltas[i])} for i in range(CASES)]}


if __name__ == '__main__':
    target = pathlib.Path(__file__).resolve().parents[1] / 'tests/fixtures/lstm_reference.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    fixture = build()
    target.write_text(json.dumps(fixture), encoding='utf-8')
    print('wrote', target)
    print('torch', fixture['torch_version'], '| deltas:', [round(c['delta'], 6) for c in fixture['cases']])
