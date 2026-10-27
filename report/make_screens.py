"""Check the offline report in a real browser before exporting its five plates."""
from pathlib import Path
import json
import sys
import struct
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from common import ROOT, RESULTS, digest, read_json, save_json
from render_report import lookup
from data_validation import require
from playwright.sync_api import sync_playwright


def png_size(path: Path) -> tuple[int, int]:
    with path.open('rb') as stream:
        header = stream.read(24)
    require(header[:8] == b'\x89PNG\r\n\x1a\n', 'Expected PNG')
    return struct.unpack('>II', header[16:24])


def run() -> None:
    metrics = read_json(RESULTS / 'metrics.json')
    screens = ROOT / 'report/screens'
    screens.mkdir(exist_ok=True)
    requests, errors, captures = [], [], {}
    with sync_playwright() as playwright:
        options = {'headless': True}
        edge = Path('C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe')
        chrome = Path('C:/Program Files/Google/Chrome/Application/chrome.exe')
        if edge.exists():
            options['executable_path'] = str(edge)
        elif chrome.exists():
            options['executable_path'] = str(chrome)
        browser = playwright.chromium.launch(**options)
        context = browser.new_context(viewport={'width': 1440, 'height': 900}, device_scale_factor=2)
        context.route('http://**/*', lambda route: route.abort())
        context.route('https://**/*', lambda route: route.abort())
        page = context.new_page()
        page.on('request', lambda request: requests.append(request.url) if request.url.startswith(('http:', 'https:')) else None)
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.goto((ROOT / 'report/index.html').as_uri(), wait_until='load')
        checked = 0
        for item in page.locator('[data-metric]').evaluate_all('(els) => els.map(e => ({path:e.dataset.metric, fmt:e.dataset.format, text:e.textContent}))'):
            require(item['text'] == format(lookup(metrics, item['path']), item['fmt']), f'Wrong displayed value: {item["path"]}')
            checked += 1
        embedded = json.loads(page.locator('#report-data').text_content())
        require(embedded['metrics'] == metrics, 'Embedded metrics differ from saved results')
        for index, name in enumerate(['01_storm', '02_scoreboard', '03_diurnal', '04_method'], 1):
            element = page.locator(f'#s{index}')
            rectangle = element.bounding_box()
            output = screens / f'{name}.png'
            element.screenshot(path=str(output), animations='disabled')
            width, height = png_size(output)
            require(abs(width - rectangle['width'] * 2) <= 2 and abs(height - rectangle['height'] * 2) <= 2, 'Wrong section pixel dimensions')
            captures[output.name] = {'width': width, 'height': height, 'sha256': digest(output)}
        require(page.evaluate('document.documentElement.scrollWidth <= window.innerWidth'), 'Desktop overflow')
        page.set_viewport_size({'width': 390, 'height': 844})
        require(page.evaluate('document.documentElement.scrollWidth <= window.innerWidth'), 'Mobile overflow')
        require(page.locator('h1').is_visible() and page.locator('#s4').is_visible(), 'Mobile content missing')
        mobile = screens / 'mobile_check.png'
        page.screenshot(path=str(mobile), full_page=True, animations='disabled')
        social_context = browser.new_context(viewport={'width': 1440, 'height': 900}, device_scale_factor=1)
        social_context.route('http://**/*', lambda route: route.abort())
        social_context.route('https://**/*', lambda route: route.abort())
        social = social_context.new_page()
        social.goto((ROOT / 'report/index.html').as_uri(), wait_until='load')
        social.locator('#social').screenshot(path=str(screens / 'social.png'), animations='disabled')
        require(png_size(screens / 'social.png') == (1200, 630), 'Share card must be exactly 1200 by 630')
        captures['social.png'] = {'width': 1200, 'height': 630, 'sha256': digest(screens / 'social.png')}
        require(not requests and not errors, f'Offline/browser check failed: {requests}, {errors}')
        browser.close()
    save_json(RESULTS / 'report_checks.json', {'status': 'passed', 'values_checked': checked,
              'index_sha256': digest(ROOT / 'report/index.html'), 'metrics_sha256': digest(RESULTS / 'metrics.json'),
              'desktop_viewport': [1440, 900], 'section_device_scale_factor': 2,
              'mobile_width': 390, 'mobile_no_overflow': True, 'runtime_network_requests': requests,
              'page_errors': errors, 'captures': captures})
    print(f'PASS offline browser: {checked} metric values; desktop/mobile; four sections and 1200x630 share card.', flush=True)


if __name__ == '__main__':
    run()
