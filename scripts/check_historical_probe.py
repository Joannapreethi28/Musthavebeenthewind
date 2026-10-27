"""Run the unchanged historical probe only after the production model is frozen."""
from pathlib import Path
import re
import subprocess
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from common import ROOT, RESULTS, save_json
from data_validation import require


def run() -> dict:
    require((RESULTS / 'frozen_experiment.json').exists(), 'Historical test scoring requires frozen experiment')
    output = subprocess.check_output([sys.executable, str(ROOT / 'reference/probe_baselines.py'), str(ROOT / 'data/raw/goes15')], text=True)
    pattern = r'(val2016|test2017): N=(\d+)  persist PE=([-\d.]+) RMSE=([\d.]+)\s+\| persist\+diurnal PE=([-\d.]+) RMSE=([\d.]+)'
    result = {'recipe': 'Unmodified reference/probe_baselines.py; historical E/W pairs, including original boundary behavior.', 'stdout': output, 'splits': {}}
    for name, n, pe, rmse, dp, dr in re.findall(pattern, output):
        result['splits'][name] = {'n': int(n), 'persistence': {'pe': float(pe), 'rmse': float(rmse)}, 'persistence_diurnal': {'pe': float(dp), 'rmse': float(dr)}}
    require(set(result['splits']) == {'val2016', 'test2017'}, 'Could not parse reference probe')
    for split, expected in [('val2016', [.354, .903, .706, .609]), ('test2017', [.293, .856, .683, .573])]:
        actual = result['splits'][split]
        values = [actual['persistence']['pe'], actual['persistence']['rmse'], actual['persistence_diurnal']['pe'], actual['persistence_diurnal']['rmse']]
        require(all(abs(a - b) <= .01 for a, b in zip(values, expected)), 'Historical probe mismatch')
    save_json(RESULTS / 'historical_probe.json', result)
    print('PASS unchanged historical probe for validation and test.', flush=True)
    return result


if __name__ == '__main__':
    run()
