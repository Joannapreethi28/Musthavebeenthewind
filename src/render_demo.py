"""Inline the demo pages into standalone HTML.

Why inline rather than ship modules and fetch JSON: browsers refuse ES-module
imports and fetch() across file:// for CORS reasons, and DESIGN_BRIEF section 5
requires the pages to open from disk with no build step and no runtime network.
Keeping lstm.js and runtime.js as real modules on disk is what lets Node test
them; this step is the only place the module syntax is flattened.
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, save_json  # noqa: E402
from export_weights import require  # noqa: E402

DEMO = ROOT / 'report/demo'
PAGES = {'explorer.template.html': 'index.html', 'sensitivity.template.html': 'sensitivity.html'}
EXTERNAL = re.compile(r'(?:src|href)\s*=\s*["\'](?:https?:)?//', re.IGNORECASE)


def flatten(source: str) -> str:
    """Drop ES module syntax so the code runs from an inline <script> tag."""
    source = re.sub(r'^\s*import\s[^;]+;\s*$', '', source, flags=re.MULTILINE)
    return re.sub(r'^export\s+(?=(?:const|function|class|let)\b)', '', source, flags=re.MULTILINE)


def render(template: str, parts: dict) -> str:
    output = template
    for marker, value in parts.items():
        token = f'/*__{marker}__*/'
        require(token in output, f'Template is missing {token}')
        output = output.replace(token, value)
    return output


def run() -> dict:
    model = (DEMO / 'model.json').read_text(encoding='utf-8')
    series = (DEMO / 'series.json').read_text(encoding='utf-8')
    parts = {
        'DEMO_CSS': (DEMO / 'demo.css').read_text(encoding='utf-8'),
        'LSTM_JS': flatten((DEMO / 'lstm.js').read_text(encoding='utf-8')),
        'RUNTIME_JS': flatten((DEMO / 'runtime.js').read_text(encoding='utf-8')),
        'MODEL_JSON': model.strip(),
        'SERIES_JSON': series.strip(),
    }
    written = {}
    for template_name, page_name in PAGES.items():
        template_path = DEMO / template_name
        if not template_path.exists():
            continue
        page = render(template_path.read_text(encoding='utf-8'), parts)
        require(not EXTERNAL.search(page), f'{page_name} references an external URL')
        require('/*__' not in page, f'{page_name} still has an unfilled marker')
        (DEMO / page_name).write_text(page, encoding='utf-8')
        written[page_name] = round(len(page.encode('utf-8')) / 1024)
        print(f'PASS wrote report/demo/{page_name} ({written[page_name]} KB)', flush=True)

    require(written, 'No demo templates found')
    save_json(ROOT / 'report/demo/manifest.json',
              {'pages': written, 'offline': True,
               'checkpoint_sha256': __import__('json').loads(model)['provenance']['checkpoint_sha256']})
    return written


if __name__ == '__main__':
    run()
