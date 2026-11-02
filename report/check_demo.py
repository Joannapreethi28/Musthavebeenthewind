"""Load each rendered demo page in a real browser and fail loudly on any defect.

Why: the pages run a neural network in the browser from a file:// URL. A silent
console error or a blocked module would leave a page that looks fine and
forecasts nothing, so rendering is never trusted without this check.
"""
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / 'report/demo'
EDGE = Path(r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe')
CHROME = Path(r'C:\Program Files\Google\Chrome\Application\chrome.exe')
PAGES = ['index.html', 'sensitivity.html']


def check(page_name: str, browser, width: int, height: int) -> list[str]:
    problems: list[str] = []
    context = browser.new_context(viewport={'width': width, 'height': height}, device_scale_factor=2)
    page = context.new_page()
    page.on('console', lambda m: problems.append(f'console.{m.type}: {m.text}') if m.type in ('error', 'warning') else None)
    page.on('pageerror', lambda e: problems.append(f'pageerror: {e}'))
    page.on('request', lambda r: problems.append(f'network: {r.url}') if not r.url.startswith('file:') else None)
    page.goto((DEMO / page_name).as_uri(), wait_until='load')

    # The sensitivity sweep runs thousands of forward passes in animation-frame
    # slices, so wait for the number to appear rather than guessing a delay.
    try:
        page.wait_for_function(
            "() => { const n = document.querySelector('#r-model, #s-readout');"
            " return n && !['', '\\u2014'].includes(n.textContent.trim()); }",
            timeout=30000)
    except Exception:
        problems.append('forecast readout never filled in within 30 s')
    page.wait_for_timeout(250)
    if page.locator('#run').count() and page.locator('#run').is_disabled():
        problems.append('compute button left disabled after the run finished')
    if page.locator('svg').count() == 0:
        problems.append('no chart rendered')
    overflow = page.evaluate('document.documentElement.scrollWidth - document.documentElement.clientWidth')
    if overflow > 1:
        problems.append(f'horizontal overflow of {overflow}px at {width}px')
    page.screenshot(path=str(DEMO / f'screens/{Path(page_name).stem}_{width}.png'), full_page=True)
    context.close()
    return problems


def run() -> int:
    (DEMO / 'screens').mkdir(exist_ok=True)
    failures = 0
    with sync_playwright() as playwright:
        options = {}
        if EDGE.exists():
            options['executable_path'] = str(EDGE)
        elif CHROME.exists():
            options['executable_path'] = str(CHROME)
        browser = playwright.chromium.launch(**options)
        for page_name in PAGES:
            if not (DEMO / page_name).exists():
                continue
            for width, height in ((1440, 900), (390, 844)):
                problems = check(page_name, browser, width, height)
                status = 'PASS' if not problems else 'FAIL'
                print(f'{status} {page_name} @ {width}x{height}')
                for problem in problems:
                    print('   ', problem)
                failures += len(problems)
        browser.close()
    print('\n' + ('ALL DEMO CHECKS PASSED' if not failures else f'{failures} problem(s)'))
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(run())
