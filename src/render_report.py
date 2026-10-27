"""Generate every reported result from metrics, including the README and viva handoff."""
from html import escape
import json
from common import ROOT, RESULTS, digest, read_json, save_json

LABELS = {'persistence': 'Persistence', 'persistence_diurnal': 'Persistence + diurnal', 'lstm': 'LSTM'}


def lookup(data: dict, path: str):
    for key in path.split('.'):
        data = data[int(key)] if isinstance(data, list) else data[key]
    return data


def result_table(metrics: dict, html: bool = True) -> str:
    rows = []
    def number(path, fmt='.3f'):
        value = format(lookup(metrics, path), fmt)
        return f'<span data-metric="{path}" data-format="{fmt}">{value}</span>' if html else value
    for model, label in LABELS.items():
        cells = []
        for split in ['val', 'test']:
            for key in ['pe', 'rmse']:
                base = f'models.{model}.{split}'
                if model == 'lstm':
                    cells.append(number(f'{base}.aggregate.{key}.mean') + ' ± ' + number(f'{base}.aggregate.{key}.std'))
                else:
                    cells.append(number(f'{base}.{key}'))
        rows.append(f'<tr class="{ "model" if model == "lstm" else "baseline" }"><td>{label}</td>' + ''.join(f'<td>{v}</td>' for v in cells) + '</tr>' if html else '| ' + ' | '.join([label, *cells]) + ' |')
    if html:
        return '<div class="table-wrap"><table aria-label="Validation and test results"><thead><tr><th>Forecast</th><th>Val PE ↑</th><th>Val RMSE ↓</th><th>Test PE ↑</th><th>Test RMSE ↓</th></tr></thead><tbody>' + ''.join(rows) + '</tbody></table></div>'
    return '| Forecast | Val PE | Val RMSE | Test PE | Test RMSE |\n|---|---:|---:|---:|---:|\n' + '\n'.join(rows)


def run() -> None:
    m = read_json(RESULTS / 'metrics.json')
    data = read_json(ROOT / 'report/data.json')
    def n(path, fmt='.3f'):
        return f'<span class="num" data-metric="{path}" data-format="{fmt}">{format(lookup(m, path), fmt)}</span>'
    def svg(name):
        content = (RESULTS / f'figs/{name}.svg').read_text(encoding='utf-8')
        return content[content.index('<svg'):]
    improved = m['comparison']['mean_pe_improvement'] > 0
    conclusion = 'The LSTM improves on the daily-cycle baseline.' if improved else 'The daily-cycle baseline remains the stronger forecast.'
    qualification = 'All three seeds improve on the diurnal baseline.' if m['comparison']['all_seeds_beat_diurnal'] else 'Seed-level differences are reported in the saved metrics; the mean alone does not describe every run.'
    limitations = ('This is retrospective forecasting with archived inputs; processing latency is not modelled. '
                   'One satellite and one test year limit generalization. Adjacent windows overlap, so samples are not independent. '
                   'The operational electron product may retain proton contamination and calibration error. '
                   'Missing inputs exclude some hours, including storm intervals. The flux floor changes low-level observations. '
                   'Seed spread measures training variability, not confidence over future storms. This experiment does not establish operational warning skill.')
    storm = svg('01_storm')
    html = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="description" content="A reproducible twelve-hour GOES-15 electron-flux forecasting experiment, with an LSTM and two measured baselines.">
<title>GEO / Twelve hours ahead</title><link rel="stylesheet" href="style.css"></head><body>
<header class="masthead"><a class="wordmark" href="#s1">GEO / FIELD NOTES</a><nav aria-label="Report sections"><a href="#s2">Results</a><a href="#s3">Daily cycle</a><a href="#s4">Method</a></nav></header>
<main><section id="s1" aria-labelledby="hero-title"><div class="intro-grid"><div>
<p class="eyebrow">NNDL · LICET · Research note 01</p><h1 id="hero-title">Twelve hours ahead<br>at geostationary orbit.</h1>
<p class="deck">Forecasting high-energy electron flux from its recent history and the solar wind. A small LSTM, tested against the daily cycle it must improve on.</p>
<p class="authors">Jabin Joseph M &amp; Joanna Preethi J<br><span>B.Tech Artificial Intelligence &amp; Data Science · LICET</span></p></div>
<aside class="result-note"><p class="label">LSTM · {n('meta.test_year','d')} test PE</p>
<p class="big-score">{n('models.lstm.test.aggregate.pe.mean')} <span class="uncertainty">± {n('models.lstm.test.aggregate.pe.std')}</span></p>
<p class="note">Mean ± sample SD across {n('meta.seed_count','d')} seeds.<br>Diurnal baseline: {n('models.persistence_diurnal.test.pe')} PE.</p></aside></div>
<div class="figure-row"><figure role="img" aria-label="Observed and predicted electron flux during September 2017">{storm}<figcaption>01 / The September storm window. Every forecast uses only history available at its issue time. Lines show seed {n('meta.representative_seed','d')}; gaps preserve excluded hours.</figcaption></figure>
<aside class="caption"><span class="label">The finding</span>{conclusion}<p class="note">Mean PE difference: {n('comparison.mean_pe_improvement','+.3f')}. Scores cover the full eligible test year, not only this storm.</p></aside></div></section>
<section id="s2" aria-labelledby="score-title"><div class="chapter-head"><span class="plate">02 / Skill</span><div><h2 id="score-title">The daily cycle sets the bar.</h2><p class="deck">Persistence carries today's flux forward. The stronger baseline adds a local-time correction learned only from training data.</p></div></div>
{result_table(m)}<figure role="img" aria-label="Prediction efficiency for validation and test">{svg('02_scoreboard')}<figcaption>Bars show one sample standard deviation across seeds. Dashed markers use the measured diurnal baseline for each split. RMSE is in log₁₀ flux units. PE = 1 − squared error / observed variance sum.</figcaption></figure>
<p class="note">Validation: {n('split_sizes.val',',d')} common samples. Test: {n('split_sizes.test',',d')} common samples. {qualification} No statistical significance claim is made.</p></section>
<section id="s3" aria-labelledby="cycle-title"><div class="chapter-head"><span class="plate">03 / Rhythm</span><div><h2 id="cycle-title">A forecast crosses the daily cycle.</h2><p class="deck">The training flux peaks near local-time bin {n('data_quality.diurnal_peak_bin','d')}. Its peak-to-trough range is {n('data_quality.diurnal_amplitude')} log₁₀ units.</p></div></div>
<figure role="img" aria-label="Training daily flux cycle and test error by local solar time">{svg('03_diurnal')}<figcaption>Local solar time uses the satellite's fixed reference longitude. The left curve uses observed training hours; the right uses the common test samples, grouped by target local time. This is local solar time, not magnetic local time.</figcaption></figure></section>
<section id="s4" aria-labelledby="method-title"><div class="chapter-head"><span class="plate">04 / Method</span><div><h2 id="method-title">Learn the change. Preserve the clock.</h2><p class="deck">A forecast is the observed current log-flux plus a predicted change. Time boundaries and missing observations are part of the model contract.</p></div></div>
<div class="method-strip"><div><span class="label">01 / Observe</span><strong>GOES + OMNI</strong><p>E-detector flux, wind speed, Bz, Dst and local-time sine/cosine.</p></div><div><span class="label">02 / Window</span><strong>{n('meta.window_hours','d')} hours × 6</strong><p>Training-only normalization; causal short-gap interpolation.</p></div><div><span class="label">03 / Learn</span><strong>{n('meta.parameter_count',',d')} weights</strong><p>Two LSTM layers; best validation checkpoint per seed.</p></div><div><span class="label">04 / Forecast</span><strong>+{n('meta.horizon_hours','d')} hours</strong><p>Current observed log-flux + predicted Δ. Observed targets only.</p></div></div>
<figure role="img" aria-label="Chronological splits and training curves for three seeds">{svg('04_method')}<figcaption>Training loss includes dropout; validation runs in evaluation mode. Dotted lines mark restored checkpoints. The test year is evaluated after all checkpoints and settings are frozen.</figcaption></figure>
<div class="limitations"><h3>Limitations</h3><div><p>{limitations}</p><p>The optional Transformer and tuning grid were omitted before test evaluation to keep this course experiment bounded.</p></div></div></section></main>
<footer><p>Sources: <a href="https://www.ncei.noaa.gov/products/goes-1-15/space-weather-instruments">NOAA GOES</a> · <a href="https://omniweb.gsfc.nasa.gov/">NASA OMNI</a><br>Data contract: {escape(m['contract'])}</p><p><a href="../README.md">Reproduction notes</a> · <a href="../results/metrics.json">Measured metrics</a><br>Offline report · no runtime network requests</p></footer>
<p class="screen-only-title">Share card / exported separately</p><div class="social-wrap"><article id="social" class="share-card" aria-label="Share card"><div><p class="eyebrow">GEO / FIELD NOTES · {n('meta.test_year','d')} TEST</p><h2>Twelve hours ahead at geostationary orbit.</h2><p class="note">{conclusion}</p></div><div class="share-result"><span class="label">LSTM test PE</span><div class="big-score">{n('models.lstm.test.aggregate.pe.mean')}</div><span class="note">Diurnal: {n('models.persistence_diurnal.test.pe')}<br>Mean of {n('meta.seed_count','d')} seeds</span></div><div class="share-chart">{storm}</div><p class="share-authors">Jabin Joseph M &amp; Joanna Preethi J · LICET · Neural Networks &amp; Deep Learning</p></article></div>
<script type="application/json" id="report-data">{json.dumps(data, allow_nan=False).replace('<', chr(92)+'u003c')}</script></body></html>'''
    (ROOT / 'report/index.html').write_text(html, encoding='utf-8')
    write_readme(m, conclusion, limitations)
    write_viva(m, conclusion)
    save_json(RESULTS / 'report_manifest.json', {'metrics_sha256': digest(RESULTS / 'metrics.json'),
              'data_sha256': digest(ROOT / 'report/data.json'),
              'figure_sha256': {p.name: digest(p) for p in sorted((RESULTS / 'figs').glob('*'))},
              'index_sha256': digest(ROOT / 'report/index.html'), 'readme_sha256': digest(ROOT / 'README.md')})
    print('PASS generated offline report, README and viva results from metrics.json.', flush=True)


def write_readme(m: dict, conclusion: str, limitations: str) -> None:
    table = result_table(m, html=False)
    q = m['data_quality']
    content = f'''# Twelve-hour GEO electron-flux forecasting

{conclusion} The mean test PE difference from persistence + diurnal is {m['comparison']['mean_pe_improvement']:+.3f}. This is a retrospective, hourly GOES-15 experiment by Jabin Joseph M and Joanna Preethi J, B.Tech AI & Data Science, LICET.

[Open the offline report](report/index.html) · [Share card](report/screens/social.png) · [Measured results](results/metrics.json) · [Viva handoff](docs/RESULTS_AND_VIVA.md)

![September storm comparison](results/figs/01_storm.png)

## Measured results

{table}

PE is on log10 forecast levels (higher is better); RMSE is in log10 flux units (lower is better). LSTM values are mean ± sample standard deviation across {m['meta']['seed_count']} seeds, not an ensemble score. Validation year {m['meta']['val_year']}: {m['split_sizes']['val']:,} samples; held-out test year {m['meta']['test_year']}: {m['split_sizes']['test']:,}. Each model uses the same timestamps. Figures and named LSTM CSV columns show seed {m['meta']['representative_seed']}. Seed variability is not a confidence interval over storms.

## Reproduce

Run commands from the `source/` checkout in PowerShell. Python and Git must be installed. The current machine uses CPU PyTorch; exact verified dependencies are in `results/setup/requirements.freeze.txt`.

```powershell
python scripts/setup_environment.py
.\\.venv\\Scripts\\python.exe -m unittest discover -s tests -v
.\\.venv\\Scripts\\python.exe src/download_data.py
.\\.venv\\Scripts\\python.exe src/download_data.py --verify-only
.\\.venv\\Scripts\\python.exe src/build_dataset.py
.\\.venv\\Scripts\\python.exe src/baselines.py
.\\.venv\\Scripts\\python.exe src/train.py
.\\.venv\\Scripts\\python.exe src/evaluate.py
.\\.venv\\Scripts\\python.exe src/render_report.py
.\\.venv\\Scripts\\python.exe report/make_screens.py
.\\.venv\\Scripts\\python.exe scripts/verify_project.py
```

These are the fresh-experiment commands. Training refuses to run after `frozen_experiment.json` exists, protecting the held-out experiment. To reproduce training from scratch, use a separate clean checkout and a new local results directory; retain the published experiment. A completed local checkout can rerun evaluation, rendering and screenshots using its frozen checkpoints. Downloading skips validated files. Raw data, processed arrays, `.venv` and checkpoints are Git-ignored; reproduce them locally using these commands. The saved predictions allow metric verification without training or local checkpoints. Setup uses the recorded CPU torch release; for exact remaining dependency versions install the freeze file after torch. On Linux/macOS use `.venv/bin/python` in place of the Windows interpreter path.

Screenshots use installed Microsoft Edge or Chrome when available; otherwise install Playwright Chromium with `.\\.venv\\Scripts\\python.exe -m playwright install chromium`. The HTML opens directly from disk and makes no network requests. The report has no build system or backend.

## Method and data

Use {m['meta']['window_hours']} hourly rows × six features to predict delta = log10F(t+{m['meta']['horizon_hours']}h) − log10F(t). Forecast level = observed current log-flux + predicted delta. The {m['meta']['parameter_count']:,}-parameter LSTM uses two hidden layers, training-only normalization, AdamW and early stopping on validation MSE. Training order is chronological. Default configuration and seeds are fixed in `results/frozen_experiment.json`; the optional Transformer and tuning grid were omitted before test evaluation.

The {q['rows']:,}-hour grid uses NOAA GOES-15 operational EPEAD E-detector corrected flux and NASA OMNI2 wind speed, Bz GSM and Dst. Average at least six quality-valid five-minute flux samples per hour in linear space, floor valid hourly means at one, then log10. Timestamp-based aggregation preserves missing archive records. Local solar time supplies sine/cosine features. See [protocol amendments](docs/PROTOCOL_AMENDMENTS.md), which supersede preliminary E/W averages and fixed score expectations in the original brief.

Rows are hour-start labels; forecasts are issued at hour end. The target is the hourly mean ending twelve hours after issue. Training is 2012–2015, validation 2016 and test 2017. Targets crossing either split boundary are excluded. Only complete, bounded gaps of at most three hours can be filled, and both endpoints must be available at issue time. Anchors and targets are never interpolated. Earlier observed history is allowed at the start of a later split.

## Verification and provenance

- Accepted training samples: {m['split_sizes']['train']:,}; per-rule exclusions: `results/data_quality.json`.
- Annual observed E-only flux coverage: {min(q['observed_flux_coverage'].values()):.2%}–{max(q['observed_flux_coverage'].values()):.2%}; floored hourly observations: {q['floored_hours']:,}.
- Training daily amplitude: {q['diurnal_amplitude']:.5f} log10; peak local-time bin: {q['diurnal_peak_bin']}; 2015 minimum Dst: {q['dst_2015_min']:.0f} nT.
- The 64-sample diagnostic passed with MSE {m['overfit']['mse']:.6f}. Every saved checkpoint reproduced its validation predictions exactly when reloaded.
- All final PE/RMSE values were independently recomputed from saved per-seed predictions. Historical E/W baseline checks are recorded separately from corrected experiment results.
- Source URLs, retrieval dates, byte sizes, missing record counts and SHA-256 hashes: `results/data_manifest.json`. Source/config/data/checkpoint provenance: `results/runs/` and `results/frozen_experiment.json`.

All result numbers in this README are generated by `src/render_report.py` from `results/metrics.json`.

## Limitations

{limitations}

## Sources

- [NOAA GOES operational archive](https://www.ncei.noaa.gov/data/goes-space-environment-monitor/access/avg/) and [instrument documentation](https://www.ncei.noaa.gov/products/goes-1-15/space-weather-instruments).
- [NASA OMNI2 format](https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/omni2.text) and [time preparation](https://omniweb.gsfc.nasa.gov/html/omni2_doc_addition.html).
- [Rodriguez et al. (2025)](https://repository.library.noaa.gov/view/noaa/67315/noaa_67315_DS1.pdf), GOES-15 W-sensor degradation; motivation for E-only use. This project's operational product differs from the paper's science-quality input.
- [Sun et al. (2024)](https://doi.org/10.1051/swsc/2024021), context for LSTM/Transformer forecasting. Different satellites, sampling and periods prevent direct score ranking.

Student ownership includes understanding, reviewing and explaining the implementation and results; the author names do not assert an unverified division of implementation work.
'''
    (ROOT / 'README.md').write_text(content, encoding='utf-8')


def write_viva(m: dict, conclusion: str) -> None:
    text = f'''# Results and viva handoff

{conclusion}

{result_table(m, html=False)}

1. What do we predict? The change in log10 electron flux twelve hours ahead. Add it to the observed current value to recover the forecast level.
2. What does one sample contain? The last 72 hourly rows, each with log-flux, speed, Bz GSM, Dst, and local-time sine/cosine. Input shape is batch × 72 × 6. Two 64-unit LSTM layers feed a scalar linear head: {m['meta']['parameter_count']:,} learned parameters.
3. Why E only? Published instrument assessments identify degradation of GOES-15 W. This choice was fixed before training, not selected for a better score.
4. Which time is predicted? A row labelled 10:00 describes 10:00–11:00. Issue at 11:00; predict the mean over 22:00–23:00, ending twelve hours later. Archived processing delay is not modelled.
5. Why not shuffle or normalize on everything? Both risk giving training information from later periods. Statistics use unique training-available hours; both issue and target times must stay in their assigned split.
6. Why preserve gaps? A missing day is not an invitation to shift the next day's timestamps. Only bounded gaps up to three hours are interpolated, and their right endpoints must already be available. Current anchors and labels remain observed.
7. What is the real baseline? Persistence plus the training-only mean twelve-hour change for the current local-time bin. It scores test PE {m['models']['persistence_diurnal']['test']['pe']:.3f}. Plain persistence is weaker because the target falls on the opposite side of the daily cycle.
8. How did the LSTM do? Test PE {m['models']['lstm']['test']['aggregate']['pe']['mean']:.3f} ± {m['models']['lstm']['test']['aggregate']['pe']['std']:.3f}; the mean difference from the diurnal baseline is {m['comparison']['mean_pe_improvement']:+.3f}. Explain this measured result without claiming statistical significance.
9. Why three seeds? They expose optimization variability. Their sample SD does not estimate uncertainty over future storms. The plotted model is predetermined seed zero, not the best test seed or an ensemble.
10. How do we know the implementation works? The overfit MSE was {m['overfit']['mse']:.6f}, checkpoint reload was exact, leakage/timing tests passed, and saved predictions reproduce the scores. Validation selects checkpoints; test is read for scoring only after freeze.
11. What are the limits? One satellite/year, overlapping samples, contaminated operational flux, floor convention, missing-input exclusions, and unmodelled input latency. No operational alert claim and no direct ranking against a different paper's dataset.
12. Why no Transformer? It was optional. The required experiment and complete reproducibility/report were prioritized before opening test results.

Rehearsal: each student should explain one raw observation's path through hourly aggregation, split assignment, interpolation eligibility, normalization, the LSTM tensor, delta reconstruction and scoring. Do not assign personal contributions that cannot be substantiated.
'''
    (ROOT / 'docs/RESULTS_AND_VIVA.md').write_text(text, encoding='utf-8')


if __name__ == '__main__':
    run()
