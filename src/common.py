"""Shared artifact helpers keep the experiment auditable and JSON strictly finite."""
from pathlib import Path
import hashlib
import json
import subprocess

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results'
CONTRACT = 'e-only-hour-end-v1'


def save_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path: Path) -> str:
    """Chunked read instead of hashlib.file_digest so the stated Python 3.10 floor still hashes."""
    running = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            running.update(block)
    return running.hexdigest()


def provenance() -> dict:
    """Hash source as well as HEAD because an experiment may precede its commit."""
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    return {'git_commit': revision, 'source_sha256': {
        p.name: digest(p) for p in sorted((ROOT / 'src').glob('*.py'))},
        'data_manifest_sha256': digest(RESULTS / 'data_manifest.json'),
        'dataset_sha256': digest(ROOT / 'data/processed/dataset.npz')}
