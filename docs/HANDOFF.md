# Completed project — 28 October 2026

The required experiment and presentation are complete. No training or browser process from this task is left running.

- Final results and reproduction commands: [README](../README.md).
- Offline report: [report/index.html](../report/index.html); four section screenshots and a share card: [report/screens](../report/screens).
- Scientific results: [metrics.json](../results/metrics.json); exported validation/test predictions and all three seed histories are in results/.
- Acceptance audit: [acceptance.json](../results/acceptance.json). Browser checks: [report_checks.json](../results/report_checks.json).
- Student explanation and results: [RESULTS_AND_VIVA.md](RESULTS_AND_VIVA.md).
- Authoritative scientific corrections: [PROTOCOL_AMENDMENTS.md](PROTOCOL_AMENDMENTS.md). The original brief and probe remain as historical references.

All three LSTM seeds finished with early stopping and exact checkpoint reload checks. Configuration, source/data provenance and checkpoint hashes were frozen before test scoring. Every final score was recomputed from exported predictions. The report, README and viva results were generated from metrics; desktop/mobile rendering and offline requests were checked in a real browser. Screenshot clipping rounds CSS coordinates correctly; the share card is captured at an integer origin at its exact required size.

The optional Transformer and tuning grid were omitted before test evaluation to prioritize the required model and the user's speed request. The report discloses single-satellite/year coverage, overlapping windows, operational-product limitations, missing-input exclusions and unmodelled input latency. Seed spread is not a significance test.

## Local assets and future use

Work from the source/ checkout. The .venv, raw/processed data and all three checkpoints remain local and Git-ignored. Source, download provenance, configs, metrics, predictions, figures, report and screenshots are tracked.

To verify the delivered artifacts:

```powershell
.\.venv\Scripts\python.exe scripts/verify_project.py
```

This also works in a clean repository copy using an interpreter with the documented dependencies; local raw data and checkpoints are optional for the exported-results audit. The original local checkout additionally verifies every sample's observed labels, causal history, dataset hash and training-statistics invariance.

Do not rerun train.py on the frozen experiment or tune it using the published test result. Follow the README's separate-checkout instructions for a new reproduction. For presentation changes, rerun render_report.py, make_screens.py and verify_project.py, then commit/push the changed artifacts. Keep raw data, environments and checkpoints excluded.

The earlier pause occurred after seed 0; resumption reused that verified checkpoint and trained seeds 1 and 2. The primary repository is Joannapreethi28/Musthavebeenthewind, branch joanna. GitHub and headless browser commands may require elevated tool permissions in the managed workspace.
