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

Work from the source/ checkout, branch joanna. As requested on 2 November 2026, all three frozen checkpoints, data/processed/dataset.npz and data/processed/hourly.csv are now tracked alongside the source, configs, results and report. The .venv, raw archives, caches and additional experimental checkpoints remain local and Git-ignored. See [INFERENCE_HANDOFF.md](INFERENCE_HANDOFF.md) for loading the shared weights and input windows.

To verify the delivered artifacts:

```powershell
.\.venv\Scripts\python.exe scripts/verify_project.py
```

This also works in a clean repository copy using an interpreter with the documented dependencies. The included dataset enables the checks of every sample's observed labels, causal history, dataset hash and training-statistics invariance without downloading raw archives. The inference handoff additionally checks the frozen dataset/checkpoint hashes and reproduces seed-zero test predictions.

Do not rerun train.py on the frozen experiment or tune it using the published test result. Follow the README's separate-checkout instructions for a new reproduction. For presentation changes, rerun render_report.py, make_screens.py and verify_project.py, then commit/push the changed artifacts. Keep the shared frozen files intact; raw archives, environments, caches and new experimental checkpoints stay excluded. Full evaluate.py execution includes historical diagnostics requiring the raw archives; inference uses only the included processed data and weights.

The earlier pause occurred after seed 0; resumption reused that verified checkpoint and trained seeds 1 and 2. The primary repository is Joannapreethi28/Musthavebeenthewind, branch joanna. GitHub and headless browser commands may require elevated tool permissions in the managed workspace.
