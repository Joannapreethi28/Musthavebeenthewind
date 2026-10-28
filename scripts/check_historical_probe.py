"""Run the unchanged historical probe only after the production model is frozen."""
from pathlib import Path
import re
import subprocess
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from common import ROOT, RESULTS, CONTRACT, digest, read_json, save_json
from data_validation import require


def run() -> dict:
    require((RESULTS / 'frozen_experiment.json').exists(), 'Historical test scoring requires frozen experiment')
    frozen = read_json(RESULTS / 'frozen_experiment.json')
    require(frozen['contract'] == CONTRACT and frozen['config']['seeds'] == [0, 1, 2], 'Invalid frozen experiment')
    require(sorted(record['seed'] for record in frozen['runs']) == [0, 1, 2], 'Experiment is not fully frozen')
    for record in frozen['runs']:
        require(digest(ROOT / 'checkpoints' / f'lstm_seed{record["seed"]}.pt') == record['checkpoint_sha256'], 'Frozen checkpoint changed')
    require(digest(RESULTS / 'data_manifest.json') == frozen['provenance']['data_manifest_sha256'], 'Raw data manifest changed after freeze')
    manifest = read_json(RESULTS / 'data_manifest.json')
    expected_goes = {ROOT / name for name in manifest['files'] if name.startswith('data/raw/goes15/')}
    require(set((ROOT / 'data/raw/goes15').glob('g15_epead_e13ew_5m_*.nc')) == expected_goes, 'Historical probe raw file set changed')
    for path in expected_goes:
        require(digest(path) == manifest['files'][path.relative_to(ROOT).as_posix()]['sha256'], 'Historical probe raw file changed')
    output = subprocess.check_output([sys.executable, str(ROOT / 'reference/probe_baselines.py'), str(ROOT / 'data/raw/goes15')], text=True)
    pattern = r'(val2016|test2017): N=(\d+)  persist PE=([-\d.]+) RMSE=([\d.]+)\s+\| persist\+diurnal PE=([-\d.]+) RMSE=([\d.]+)'
    result = {'recipe': 'Unmodified reference/probe_baselines.py; historical E/W pairs, including original boundary behavior.',
              'source_sha256': digest(ROOT / 'reference/probe_baselines.py'),
              'frozen_experiment_sha256': digest(RESULTS / 'frozen_experiment.json'),
              'stdout': output, 'splits': {}}
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
