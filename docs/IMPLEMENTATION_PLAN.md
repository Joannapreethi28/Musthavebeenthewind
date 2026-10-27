**Implementation plan: 12-hour GEO electron-flux forecasting**

Prepared 25 October 2026 against commit `597083f`. This is the planning record. Execution was authorized for the whole project on 27 October; [PROTOCOL_AMENDMENTS.md](PROTOCOL_AMENDMENTS.md) records the implemented scientific corrections. Work is now paused at the user's request after environment/data/baseline gates and completed LSTM seed 0. See [HANDOFF.md](HANDOFF.md) for exact progress, remaining work and resume commands. Final evaluation/report execution is pending. Earlier statements about work not yet begun below describe the planning session, not current status.

**1. Intended outcome.** Deliver a reproducible course project that uses 72 hours of GOES-15 electron flux and OMNI observations to predict the next 12-hour change in log10 flux. Deliver the required LSTM, two fair baselines, measured results, four scientific figures, an offline report, five screenshots, and a README that both students can explain. Add the optional Transformer only after the core experiment works and its measured runtime fits the remaining budget.

Completion means correct data handling, a reproducible experiment, honest evaluation, and finished presentation. Beating the daily-cycle baseline is a scientific finding to test, not a result to promise. Keep the existing scope: one satellite, one horizon, six features, 2012–2017, and the existing small Python stack.

**2. Research findings that affect the build.**

| Finding | Proposed decision and evidence |
|---|---|
| The current detector averaging rule needs correction. | Use `E2E_COR_FLUX` with its quality flag for the final target; do not fill its gaps with W-sensor values. Rodriguez et al. explicitly advise avoiding GOES-15 W-channel E1/E2 data because of degradation. This changes the current E/W average and requires new baselines. [Rodriguez et al., 2025, Appendix A](https://repository.library.noaa.gov/view/noaa/67315/noaa_67315_DS1.pdf). |
| The concern applies to the project's era. | Baker et al.'s comparison covers 2012–2018 and excludes the GOES-15 W sensor because of seasonal and local-time-dependent degradation. The E-only choice is based on instrument evidence, not whichever choice gives better test scores. [Baker et al., 2019](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2019JA027331). |
| The existing 5-minute product is operational data. | Retain it for this bounded course project, explicitly identify it as such, and disclose residual contamination and calibration limitations. NOAA recommends science-quality products when available; changing product family would require a separate data-contract revision, so do not silently mix products. [NOAA product documentation](https://www.ncei.noaa.gov/products/goes-1-15/space-weather-instruments). |
| OMNI hour labels can be resolved. | NASA describes averaging shifted observations from 00:00–01:00 into the first hourly record. Treat the stored timestamp as hour start and its observation as available at hour end. OMNI is already shifted toward Earth; do not add a second L1 propagation shift. [NASA OMNI preparation](https://omniweb.gsfc.nasa.gov/html/omni2_doc_addition.html). |
| Short-gap interpolation needs an explicit implementation. | Only fill complete, internally bounded missing runs of at most three hours. `interpolate(limit=3)` alone partially fills longer runs; it does not implement that rule. Also prevent use of any bounding observation later than the forecast issue time. [pandas interpolation documentation](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.interpolate.html). |
| Literature scores are context, not our benchmark. | Sun et al. use GOES-10, five-minute data, training 2002–2003, validation 2004, and test 2005. Their best 12-hour LSTM PE is 0.554 and Transformer PE is 0.747; the paper evaluates log10 flux with the same PE formula. The LSTM number marked unconfirmed in our notes is confirmed in section 3.2. Different inputs, sampling, and years prevent direct ranking against our experiment. [Sun et al., 2024, sections 2.4 and 3.2–3.3](https://gfzpublic.gfz.de/rest/items/item_5027958_2/component/file_5027959/content). |
| Raw-flux and log-flux scores must stay separate. | Balikhin et al. Table 1 gives REFM/SNB3GEO PE of −1.31/0.63 on raw flux and 0.70/0.77 on log10 flux. The current notes place values from different columns together. Our results will always state log10 units and use our own baselines. [Balikhin et al., 2016](https://repository.library.noaa.gov/view/noaa/16062/noaa_16062_DS1.pdf). |
| The CUDA installation example needs a version choice. | Do not blindly combine an unpinned current torch release with the old CUDA 12.8 example. PyTorch's 2.12 release notes change that wheel matrix. Select a supported Windows build for the actual GPU/driver, run a forward/backward smoke check, then record the installed versions. CPU remains a supported route. [PyTorch release guidance](https://pytorch.org/blog/pytorch-2-12-release-blog/). |

These are proposed amendments to `PLAN.md`, `DATA_SPEC.md`, and the reference notes. Preserve the original probe and document why the production experiment differs. Add the instrument papers to the permitted bibliography as part of that revision.

**3. Freeze the forecasting and data contract before training.**

- Source files remain 72 monthly GOES-15 `epead_e13ew_5m` NetCDF files and six annual `omni2_YYYY.dat` files. No extra features or datasets.
- Retain only positive E-sensor flux with quality flag zero. Average at least six valid five-minute samples per hour in linear flux space, floor the resulting valid hourly mean at 1.0, then take log10. Flooring must never turn a missing hour into a valid one. Count and report floored observations.
- Read OMNI Bz GSM, speed, and Dst from 1-based columns 17, 25, and 41. Convert their documented fill values to missing values before arithmetic. Check all 55 columns and day-of-year parsing. [NASA format specification](https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/omni2.text).
- Maintain a unique, monotonically increasing UTC index with exactly 52,608 hour-start rows from 2012-01-01 through 2017-12-31. Preserve the specified `hourly.csv` columns; retain observation masks and interpolation availability metadata in a small companion file under ignored `data/processed/`.
- Keep local solar time at fixed longitude 135.5°W, evaluated at each hour's midpoint. Call this local solar time, not magnetic local time. Construct sine/cosine from it; do not z-score these two channels.
- Interpolate only bounded runs of one to three missing hours in continuous history channels, separately across the training/validation/test source periods. Retain longer and edge gaps. A sample may use an interpolated history value only if both supporting observations were already available at its issue time.
- Require an actually observed current-flux anchor and an actually observed future-flux target for every training and scoring sample. Interpolated history is permitted; invented labels are not. Record how many samples each rule removes.
- Fit four-channel means and standard deviations on finite, unique hourly values whose hour-end availability time belongs to the training period, without weighting repeatedly overlapping windows. Reject zero/invalid standard deviations. Persist these statistics with every trained checkpoint.

**Timestamp example.** A row labelled 10:00 describes 10:00–11:00. A forecast using that row is issued at 11:00. Its 12-hour-ahead target is the hourly mean ending at 23:00, i.e. the row labelled 22:00. Thus the model still targets the row 12 indices ahead. Record the issue time and the target interval explicitly. This is a forecast of an hourly mean ending 12 hours later, not an instantaneous observation at the start of the target hour. Archived OMNI/Dst processing latency is not modelled; describe the experiment as retrospective forecasting.

**Splits.** Use actual issue times: training 2012–2015, validation 2016, and test 2017. Require target end times to remain inside the same half-open calendar period; discard boundary-crossing targets at both the training and validation boundaries, and beyond the end of test. Earlier-period observations may supply the beginning of a later-period 72-hour window because they are already known. Do not move rows randomly between splits or shuffle the training loader. Calculate final sample counts from these rules rather than forcing the probe's approximate counts.

**4. Reconcile the preliminary baseline and protect the test set.**

The reference script averages two detectors, does not floor flux at 1.0, does not require complete six-feature 72-hour windows, and allows training issue times whose targets enter 2016. Its historical 0.683 test PE is therefore not a fixed acceptance threshold for the corrected experiment.

Implement two baselines over exactly the samples accepted for the neural models:

1. Persistence: predicted log-flux equals the observed current log-flux.
2. Persistence plus diurnal correction: add the training-only mean 12-hour delta for the current local-time bin. Store the counts and means for all 24 bins; reject empty bins.

Reproduce the reference's 2016 results under its original recipe as a provenance check, then measure the corrected validation baselines. Preserve the distinction in the audit output. Investigate unexplained differences from the reference's ±0.01 tolerance; do not change cleaning to recover a desired score. Document expected changes caused by the explicit amendments.

The training entry point must not calculate 2017 metrics or select configurations using the September 2017 plot. Downloading and structural quality checks can cover all years, but model decisions use only training and validation. Freeze preprocessing, models, checkpoints, display seed, and configuration before the final evaluator reads test targets. Historical 2017 probe reproduction can run then as a separately labelled provenance check.

**5. Build in this order and require the checks to pass.**

| Stage | Work and deliverable | Acceptance gate |
|---|---|---|
| 0. Repository and environment | Work in `source/`; fetch and inspect teammates' changes; establish a feature branch and verify commit/push identity. Create a project `.venv`; install the small dependency set and record exact versions/device. | Clean starting checkout; correct remote; imports succeed; an actual model forward/backward pass works on the selected device. |
| 1. Download | `src/download_data.py`: find the correct monthly file, bounded retries, timeouts, temporary downloads, integrity checks, skip validated existing files. Write source URLs, retrieval dates, file sizes, and SHA-256 hashes to a manifest. | 72 valid NetCDF files + 6 valid OMNI files; no missing months, duplicates, or HTML error pages; rerun does not redownload valid files. |
| 2. Dataset | `src/build_dataset.py`: quality filtering, hourly aggregation, OMNI alignment, masks, safe gap handling, features, and sample indices. Write data-quality counts and the revised contract version. | 52,608 unique hourly rows; exact 72×6 inputs and 12-hour target offset; time/leakage checks pass; original 2015 Dst sanity value −234 nT reproduced or discrepancy resolved. |
| 3. Baselines | `src/baselines.py`: training-only diurnal fit and validation evaluation; reproduce historical validation probe separately. | All models share the same validation timestamps; all 24 bins populated; no validation rows fit the lookup; measured PE/RMSE and sample counts saved. |
| 4. LSTM | `src/models.py` and `src/train.py`: debug overfit, selected configuration, three seeds, early stopping, checkpoint restore. | 64-sample diagnostic MSE <0.01; finite gradients/losses; validation improves from initialization; best checkpoint reload reproduces predictions. |
| 5. Optional Transformer | Reuse the same dataset, loss, selection protocol, seeds, and evaluation code with `--model transformer`. Decide inclusion from runtime and validation before test evaluation. | Same model checks as LSTM; complete three-seed results or clearly mark the model omitted. |
| 6. Frozen evaluation | `src/evaluate.py`: run all frozen models and baselines on the same final 2017 sample indices; export per-seed predictions, aggregate metrics, and figures. | Metrics recomputed from saved predictions agree numerically; no missing/nonfinite scored predictions; target timestamps align; no retraining based on test results. |
| 7. Report | Generate the offline HTML, figures, results table, narrative, and Playwright screenshots from saved artifacts. | Four figure sections + share card; no runtime network requests; figures and displayed values match exports; desktop/mobile and screenshot checks pass. |
| 8. Handoff | Generate README results, update the viva notes and limitations, verify documented commands, commit and push intended artifacts. | Fresh documented workflow works; artifacts trace to data/config/code; no raw data, environments, secrets, or local checkpoints in Git; remote commit verified. |

The E-only target may change the old ≥0.98 flux coverage, 0.74 diurnal amplitude, and peak-bin diagnostics. Measure and report them. A failure against the old probe expectations triggers investigation; any revised threshold must have a recorded data-based explanation before training, never a silent relaxation. Show observed coverage before interpolation as well as final usable-window coverage. Do not assume 8,600 valid test windows simply because that many flux pairs existed in the probe.

**6. Model and run protocol.**

The default LSTM has input `(batch,72,6)`, two layers of 64 hidden units, dropout 0.2 between layers, and a scalar linear head on the final hidden state. The expected parameter count is 51,777; print the actual count. Each window starts with fresh hidden state. Output is the raw log10 delta; forecast level is observed current log-flux plus that delta.

Retain MSE on delta, AdamW at learning rate 0.001 and weight decay 0.0001, batch size 256, maximum 60 epochs, gradient norm clipping at 1.0, and early stopping with patience eight on validation MSE. Use example-weighted epoch averages, evaluation mode during validation, and restore the best weights. Do not clip predictions to make metrics look better.

Seed Python, NumPy, PyTorch, and CUDA. Use a Windows-safe loader initially with zero worker processes and no shuffle. Save configuration, normalization, epoch history, best epoch, device, package versions, data-manifest hash, and code revision beside every run. Set deterministic options where supported and document any necessary exception; matching seeds do not guarantee identical results across hardware or PyTorch releases. [PyTorch reproducibility guidance](https://docs.pytorch.org/docs/2.14/notes/randomness.html).

Run the default configuration with seeds 0, 1, and 2. If validation results justify the allowed tiny grid and compute permits it, evaluate the remaining `(hidden, dropout)` combinations in `{32,64} × {0.1,0.2}`, with the same three seeds for each evaluated configuration. Select the lowest mean best-validation MSE. Reuse completed runs: the ceiling is 12 full fits per model, not an open-ended sweep. Stop after that budget even if the deep model loses. A checkpoint at initialization and the tiny overfit diagnostic do not count as candidate configurations.

The optional Transformer uses a 6→64 projection, learned 72×64 positional embeddings, two encoder layers with four heads, feedforward width 128, dropout 0.2, pre-normalization, mean pooling, and a scalar head. All 72 tokens are historical, so attention within this observed window need not be causally masked. Initialize encoder layers separately as PyTorch recommends. [TransformerEncoder documentation](https://docs.pytorch.org/docs/2.14/generated/torch.nn.TransformerEncoder.html).

**7. Evaluate and display one coherent experiment.**

Compute PE and RMSE on forecast log10 levels, on the complete common sample mask. Reject zero-variance observation sets for PE rather than producing a misleading finite number. Report every seed and the arithmetic mean ± sample standard deviation (`ddof=1`) across three seeds. This measures optimization variability, not uncertainty over future storms; do not convert it into a significance claim or use the old 0.02 heuristic as a statistical test.

Choose seed 0 in advance for the displayed forecast lines and the named `lstm`/`transformer` columns in `preds_test.csv`. Store each seed's predictions separately. Label the figure's seed and show three-seed aggregate scores in the table; do not confuse scores of averaged predictions with an average of seed scores. Use paired per-seed differences from the deterministic diurnal baseline to explain any measured improvement.

Outputs:

- `results/metrics.json`: schema/contract version, data/config/code provenance, split counts and exclusion counts, baseline coefficients, per-seed validation/test scores, mean/std, selected configurations, and the representative seed.
- `results/preds_test.csv`: target hour-start `time_utc`, `issue_time_utc`, `target_end_utc`, observed log-flux, persistence, persistence+diurnal, LSTM seed 0, and optional Transformer seed 0.
- `results/history_<model>_seed<k>.csv`, per-seed prediction files, small run configurations, and local ignored checkpoints with normalization sufficient to reproduce inference.
- Four figures: September 1–20, 2017 forecast comparison; full validation/test scoreboard; training mean flux and final test error by local time; method/split timeline with training and validation loss curves. Use target times for forecast overlays, show gaps, and display flux axes as powers of ten with physical units.

The diurnal baseline marker is read from the measured metrics, not hard-coded to 0.683. Discuss residual contamination, single-satellite/single-test-year coverage, overlapping windows, unmodelled processing latency, the floor convention, and missing-input exclusions. Do not claim that one storm demonstration establishes operational warning skill.

**8. Report and student handoff.** Follow the existing observatory/logbook design: warm paper, dark ink, vermilion forecasts, direct line labels, readable captions, and explicit limitations. Use plain local HTML/CSS and embedded data/SVG; keep fonts local or use suitable fallbacks. Before report work, check the design skills requested by `AGENTS.md`; keep the design brief as the actual visual specification.

`evaluate.py` exports `report/data.json`; rendering embeds its contents into `report/index.html` so opening the page via `file://` works. README numbers and all numerical report copy come from `metrics.json`. Formatting rules may round values, but must be consistent and checked automatically.

Use Playwright to capture the four sections at a 1440×900 viewport with device scale factor two. Generate `social.png` at exactly 1200×630 pixels with a separate scale-one context; this resolves the brief's ambiguity between CSS size and physical pixels. Verify the four section images' actual dimensions, mobile layout at 390 px, visible labels, contrast, and no horizontal overflow. Capture elements directly and inspect the resulting images. [Playwright screenshots](https://playwright.dev/python/docs/screenshots).

The README will contain setup, download/build/train/evaluate/report commands, the measured result table, reproduction details, source attribution, and limitations. Update viva notes to match the actual detector, time convention, split protection, model tensors, baseline, and outcomes. Describe each person's contribution accurately; proposed roles in the original brief are not evidence of work completed.

**9. Focused checks.** Use a small standard-library `unittest` suite plus phase acceptance scripts. Test scientific failure modes rather than trivial getters or visual snapshots:

- Flux quality flags, fill values, valid-count thresholds, correct order of averaging/floor/log, and NaNs staying NaNs.
- Leap day, day-of-year conversion, duplicate timestamps, and 55-column OMNI parsing.
- Complete gaps of length three versus four; unbounded edges; an input near issue time whose right interpolation endpoint is in the future.
- A future-perturbation check: changing later observations must not alter an earlier input window or forecast.
- Train statistics and diurnal coefficients unchanged when validation/test values change.
- Target timestamps stay in their assigned split; earlier historical context remains allowed.
- Current anchors and targets are observed; model/baseline row indices match exactly.
- Known-answer PE/RMSE examples, zero-variance handling, checkpoint reload, and metric recalculation from exported predictions.
- Every visible results value matches its generated source; the report loads without network access.

**10. Time, fallbacks, and repository delivery.** Budget roughly 8–12 hours of implementation/verification effort for the core project, subject to dependency and data access. Treat a single working day as a target, not a measured runtime. Benchmark a full epoch before committing to the training schedule; CPU training or the allowed grid may run longer or overnight. The Transformer adds roughly 1–3 hours of implementation/review plus its measured training time. Reserve time for the report and reproducibility instead of spending it on extra experiments.

The active Python is 3.13.15 and the six project packages are not installed there. A registered Python 3.12 interpreter could not be launched in this session. GPU inventory was not readable, so the RTX 5060 mentioned in the notes is unconfirmed. Use the working interpreter if supported wheels resolve; verify the actual device rather than assuming the teammate's hardware. Network and Git permissions have previously limited shell operations. Git fetch now works; dependency/data download and push authentication still need actual checks. Do not require a plugin to use Git.

For an older driver that supports CUDA 12.8 but not the current build's CUDA requirements, an explicit compatible torch release is preferable to an unplanned driver change. PyTorch's archive lists Windows CUDA 12.8 wheels for torch 2.11.0. Verify interpreter compatibility and the GPU smoke check before freezing that fallback. [Official version matrix](https://pytorch.org/get-started/previous-versions/).

If downloads fail, preserve complete files, retry with bounds, and follow NOAA/NASA's official archive links for the identical product. If essential months remain unavailable, report the blocked stage; do not substitute synthetic data or quietly shorten the experiment. If GPU setup fails, use CPU with the same model contract. If the LSTM loses to the baseline, complete the negative-result report. If a bug invalidates an already-scored test run, document the correction and rerun the frozen experiment; do not present a subsequently tuned model as an untouched holdout result.

Use small commits for the agreed protocol, data/baselines, models, and evaluation/report. Fetch before each push and reconcile teammate changes without overwriting them. Verify the connected GitHub identity and commit attribution rather than inventing an email. Push the intended source, documents, metrics, predictions, figures, and screenshots to a feature branch using the user's existing authorization; verify its remote commit. Keep raw/processed data, `.venv`, and checkpoints excluded as the repository specifies. No paid compute or public site hosting is required by this plan.

**11. Research status and remaining empirical checks.** The paper sections and technical documentation above were inspected. The expected GOES file is listed in both the [January 2012 archive](https://www.ncei.noaa.gov/data/goes-space-environment-monitor/access/avg/2012/01/goes15/netcdf/) and the [September 2017 archive](https://www.ncei.noaa.gov/data/goes-space-environment-monitor/access/avg/2017/09/goes15/netcdf/); all 78 files still need downloading and validation. No new model score, raw-data coverage figure, or runtime has been measured during this planning task.

Other bibliography checks: [Reeves et al., 2011](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2010JA015735) supports a more complex solar-wind/flux relationship than a deterministic speed-to-flux rule. [Forsyth et al., 2020](https://discovery.ucl.ac.uk/id/eprint/10102371/) evaluates threshold forecasts with hit and false-alarm rates, a different task from our regression scores. The publisher's indexed record identifies Paulikas and Blake's 1979 chapter DOI as [10.1029/GM021p0180](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/GM021p0180); its full page returned 403 here, so only bibliographic metadata was verified. The fabricated arXiv reference identified in the repository remains excluded.

**Suggestions beyond this build.** A future science-quality-product replication and additional independently held-out years would strengthen scientific generalization. They are not part of this implementation plan.
