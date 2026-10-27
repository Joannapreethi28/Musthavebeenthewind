# PLAN.md — 12-hour GEO electron-flux forecaster (LSTM / Transformer)

> Production implementation: [protocol amendments](docs/PROTOCOL_AMENDMENTS.md) supersede conflicting preliminary rules and thresholds below. Measured diagnostics are in `results/data_quality.json` and `results/baselines.json`.

> Contract for the build. Numbers marked **[probe]** were measured on real data while writing this plan (see `reference/probe_baselines.py`). The real pipeline must reproduce them within ±0.01 — if not, there is a bug.

## 1. Problem (what & why)
Satellites at geostationary orbit (GEO, 35,786 km) are damaged by >2 MeV "killer" electrons (deep-dielectric charging). Their flux swings over ~4 orders of magnitude and responds to the solar wind with a 1–3 day delay.
**Task:** from the last **72 hours** of observations, predict `log10` of the >2 MeV electron flux **12 hours ahead**.
**Why 12 h:** far enough that "tomorrow looks like now" fails (plain persistence PE ≈ 0.29 **[probe]**), short enough to learn from hourly data.

## 2. Scope — basic on purpose
**IN:** one horizon (12 h) · one satellite (GOES-15) · hourly data · 6 input features · 2 baselines · **LSTM (required)** · Transformer encoder (optional, same pipeline, one flag) · metrics PE + RMSE · 4 figures · a static results page + screenshots · README.
**OUT (do not add):** other horizons, GRASP/ISRO data, dashboard/backend, extra features, custom losses, sweeps, ablations, probabilistic outputs.

## 3. Contract
| Item | Decision |
|---|---|
| Target | `delta = log10F(t+12) − log10F(t)`; forecast = `log10F(t) + delta_hat` |
| Flux | GOES-15 EPEAD >2 MeV corrected electron flux, hourly (details: `docs/DATA_SPEC.md`) |
| Solar wind | NASA OMNI2 hourly: speed V, Bz (GSM), Dst |
| Features (6/step) | `log_flux, v_sw, bz_gsm, dst, sin(2π·LT/24), cos(2π·LT/24)` — LT = local solar time at the satellite = `(UTC_hour + 0.5 − 135.5/15) mod 24` |
| Window | 72 hourly steps ending at the forecast issue time `t` → tensor `(72, 6)` |
| Split (by issue time `t`) | Train 2012-01-01..2015-12-31 · Val 2016 · Test 2017. Drop train samples whose target (t+12 h) crosses into 2016 |
| Normalisation | z-score of the 4 continuous channels using train-split mean/std; sin/cos untouched |
| Gaps | linearly interpolate gaps ≤ 3 h; drop any sample with a remaining NaN in window or target |
| Metrics | **PE** = `1 − Σ(obs−pred)² / Σ(obs−mean(obs))²` on `log10` flux; **RMSE** in `log10` units. Both on the forecast level (not on delta) |

Sample counts for sanity [probe, before NaN drops]: train 35,064 h · val 8,784 h · test 8,760 h; usable val ≈ 8.7k, test ≈ 8.6k samples.

## 4. Baselines — the bar to beat
| Model | Val 2016 PE | Val RMSE | **Test 2017 PE** | Test RMSE |
|---|---|---|---|---|
| Persistence (`F(t+12)=F(t)`) | 0.354 | 0.903 | 0.293 | 0.856 |
| **Persistence + diurnal** | 0.706 | 0.609 | **0.683** | 0.573 |

*Persistence + diurnal* = `log10F(t) + mean_delta[LT bin of t]`, where `mean_delta` (24 values, one per local-time hour) is the average 12-h change in the **train** years. It works because flux has a strong local-time cycle (peak LT≈11, trough LT≈23, amplitude ≈0.74 in log10 **[probe]**) and a 12-h forecast lands on the opposite side of the cycle.
**Consequence:** beating plain persistence is easy and proves nothing. A deep model is only interesting if it beats **0.683 on the 2017 test set**.

## 5. Models
**LSTM (required):** input `(B,72,6)` → `LSTM(6→64, 2 layers, dropout 0.2, batch_first)` → last hidden state → `Linear(64→1)` → `delta_hat`.
**Transformer encoder (optional):** `Linear(6→64)` → + learned positional embedding `(72,64)` → `TransformerEncoder(2 layers, 4 heads, ff 128, dropout 0.2, batch_first, norm_first)` → mean-pool over time → `Linear(64→1)`.
Both ≈ 50–70k parameters on purpose (few independent storms ⇒ big models memorise).

## 6. Training protocol (same for both models)
MSE on `delta` (raw log10 units) · AdamW lr 1e-3, weight-decay 1e-4 · batch 256 · max 60 epochs · early stopping patience 8 on val MSE, restore best weights · grad-clip 1.0.
Final results: **3 seeds (0,1,2)**, report mean ± std. Allowed tiny grid (val only): hidden ∈ {32, 64}, dropout ∈ {0.1, 0.2}. Nothing else.
Debug aid: before a real run, overfit 64 samples to near-zero loss — proves the model/loss code works.

## 7. Outputs
`results/metrics.json` (config, split sizes, val+test PE/RMSE for every model; mean/std over seeds) · `results/preds_test.csv` (`time_utc, obs, persistence, persistence_diurnal, lstm[, transformer]` in log10) · `results/figs/*.png` · `results/history_<model>_seed<k>.csv` (loss per epoch).

## 8. Repo layout
```
AGENTS.md CLAUDE.md PLAN.md README.md requirements.txt
docs/        DATA_SPEC.md PHYSICS_AND_REFERENCES.md VIVA_NOTES.md DESIGN_BRIEF.md
reference/   probe_baselines.py        (throwaway reference; do not import)
src/         download_data.py build_dataset.py baselines.py models.py train.py evaluate.py
data/raw/ data/processed/hourly.csv    (git-ignored)
results/     metrics.json preds_test.csv figs/
report/      index.html style.css (+ data json) make_screens.py screens/*.png
```

## 9. Phases and acceptance checks (≈ one working day)
| # | Phase | Time | Accept when |
|---|---|---|---|
| 0 | Env: install deps, GPU check | 30 min | `torch.cuda.is_available()` prints True (CPU is acceptable) |
| 1 | `download_data.py` (GOES-15 2012–2017 + OMNI2 2012–2017) | 30 min | `data/raw` has 72 GOES `.nc` + 6 OMNI `.dat`; re-running skips existing files |
| 2 | `build_dataset.py` → `data/processed/hourly.csv` (columns in DATA_SPEC §7) | 60 min | 52,608 rows; `log_flux` valid fraction ≥ 0.98 every year; diurnal amplitude 0.74 ± 0.05, peak LT 10–12; OMNI Dst min of 2015 = −234 |
| 3 | `baselines.py` | 20 min | Table in §4 reproduced within ±0.01 |
| 4 | `models.py`, `train.py`: LSTM, 64-sample overfit test, then 3 seeds | 75 min | overfit test loss < 0.01; val loss falls; test PE/RMSE written to `metrics.json` |
| 5 | *(optional)* Transformer, same script, `--model transformer` | 45 min | same as phase 4 |
| 6 | `evaluate.py`: figures + `preds_test.csv` | 45 min | figures listed in DESIGN_BRIEF §3 exist |
| 7 | `report/` page + `make_screens.py` | 90 min | 4 screenshots + 1 share card in `report/screens/`, numbers match `metrics.json` |
| 8 | README (generated numbers), viva rehearsal | 45 min | README has setup, results table, limitations |

## 10. Honest expectations
- The LSTM may only edge past 0.683, or fail to. Both are reportable. Differences < ~0.02 PE are inside seed noise — say so.
- One test year with one large storm (Sept 2017) is a thin test set; flux level drifts upward 2012→2017 (mean log10 flux 2.50, 2.36, 1.64, 2.46, 2.76, 2.97 **[probe]**), a mild distribution shift — another reason to predict `delta`.
- Prior work (Sun et al. 2024, JSWSC) reports 12 h PE ≈ 0.747 for a Transformer on **their** data/period; not directly comparable. Do not claim to match or beat it.

## 11. Roles (proposed — confirm) & open decisions
- Jabin: phases 1–4 (data, baselines, LSTM). Joanna: phase 5 (Transformer) + phases 6–7 (figures, report). Both: phase 8; each must be able to explain the other's part.
- Open: whether to include the Transformer (needs ≈ 45 min; skip if behind) · whether to add a 2018 hold-out (GOES-15 data exists for 2018–2019 but longitude/instrument drift is unchecked — default **no**).
