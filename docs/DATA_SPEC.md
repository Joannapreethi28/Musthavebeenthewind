# DATA_SPEC.md — verified data facts

> Production implementation: [PROTOCOL_AMENDMENTS.md](PROTOCOL_AMENDMENTS.md) supersedes the E/W average, timestamp uncertainty and preliminary acceptance thresholds below. Use E-only flux, explicit hour-end availability and timestamp-based aggregation of sparse records.

Everything marked **[probe]** was measured on the real files while writing this spec (Oct 2026). Everything else is from the file metadata / the OMNI format document.

## 1. Flux: GOES-15 EPEAD, 5-minute files
- Base URL: `https://www.ncei.noaa.gov/data/goes-space-environment-monitor/access/avg/{YYYY}/{MM}/goes15/netcdf/`
- Each month directory contains **exactly one** file matching `g15_epead_e13ew_5m_\d{8}_\d{8}\.nc` (e.g. `g15_epead_e13ew_5m_20150101_20150131.nc`). Parse the directory listing for it — the end date differs per month. **Do not** use the `1m`, `_science_v1.0.0`, `a16ew`, `p17ew`, `cpflux` or `hepad` files. All 84 months of 2011–2017 exist **[probe]**; download 2012–2017 only (72 files; 84 files = 75 MB **[probe]**).
- Format: NetCDF-3 classic (header bytes `CDF\x01`). `scipy.io.netcdf_file(path, "r", mmap=False)` reads it — **no netCDF4/xarray needed**. Values are big-endian: cast with `.astype("float64")`.
- Variables used: `time_tag` (milliseconds since 1970-01-01 UTC, 5-min cadence), `E2W_COR_FLUX` (detector A) and `E2E_COR_FLUX` (detector B) = >2 MeV electron flux with proton-contamination correction, unit e/(cm²·s·sr); `E2W_QUAL_FLAG`, `E2E_QUAL_FLAG`.
- Missing flux = `-99999`; a non-zero quality flag means a data-quality problem (the flag's own fill is `99999`, also non-zero). **Valid 5-min sample ⇔ flux > 0 and flag == 0.** ≈ 99–99.8 % of samples valid per year **[probe]**.
- Two redundant detectors: the A detector reads about 0.06–0.08 log10 (15–20 %) higher than B in every year **[probe]**. Rule: **average the two in linear space** where both are valid, otherwise use the one that is valid.

## 2. Hourly flux rule
`flux_hour = mean of valid combined 5-min values in [h:00, h+1:00)`; require ≥ 6 of 12 valid else NaN. Then `flux_hour = max(flux_hour, 1.0)` and `log_flux = log10(flux_hour)`. (0.88 % of hours are ≤ 1 e/cm²/s/sr before flooring **[probe]** — near the instrument noise floor after proton correction.) Hourly valid fraction per year: 2012 0.989 · 2013 0.999 · 2014 0.995 · 2015 0.992 · 2016 0.996 · 2017 0.993 **[probe]**.
Mean log10 flux per year: 2012 2.50 · 2013 2.36 · 2014 1.64 · 2015 2.46 · 2016 2.76 · 2017 2.97 (upward drift, see PLAN §10) **[probe]**.

## 3. Traps (all hit while probing)
1. **Why 2012, not 2011:** GOES-15 sat at ≈ 89 °W until Oct 2011, drifted, and reached ≈ 135 °W in Dec 2011 **[probe]**. Local time differs by ~3 h before that → excluded.
2. **Do not use the file's `west_longitude`:** it is fill (`-99999`) for all of 2016, 87 days of 2012, 200 days of 2015 **[probe]**. Use a fixed **135.5 °W** (station-keeping range 134.0–135.5 °W in 2012–2017 **[probe]**; ≤ 0.3 h of LT error, irrelevant for hourly bins).
3. **Proton contamination:** during large solar energetic-particle events (e.g. Sept 2017) the `COR` product may still be contaminated. Do not hand-clean; mention it as a limitation.
4. Detector naming: "W/E" in the variable names does not mean the physical pointing (the detectors swap with spacecraft yaw flips). Treat as detector A/B.

## 4. Solar wind & Dst: NASA OMNI2 hourly
- URL: `https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/omni2_{YYYY}.dat` (one file per year). Format document: `.../low_res_omni/omni2.text`.
- Plain text, whitespace-delimited, **55 columns**; `np.loadtxt` works **[probe]**.
- Columns used (**1-based**, from the format document): `1` year · `2` day-of-year (Jan 1 = 1) · `3` hour (0–23) · `17` **Bz GSM** nT (fill `999.9`) · `25` **flow speed** km/s (fill `9999.`) · `41` **Dst** nT (fill `99999`). Others that exist but are unused: `24` proton density (fill 999.9), `39` Kp×10, `40` sunspot number.
- Convert fills to NaN **before** any arithmetic.
- Sanity check **[probe]**: 2015 file → Dst minimum = −234 nT at day 76, hour 22 (17 Mar 2015, "St Patrick's Day storm"); Bz range −26.2…20.1 nT; V range 259…800 km/s.
- Time convention: treat "day d, hour h" as the hour starting at h:00 UTC so it aligns with the GOES hourly floor. (Whether OMNI labels the hour start or centre was not verified; a ±1 h shift is immaterial for a 12 h forecast.)

## 5. Local time (LT) feature
`LT = (UTC_hour + 0.5 − 135.5/15) mod 24` (hours, 0–24) with `UTC_hour` = hour-of-day of the hourly timestamp. Features: `sin(2π·LT/24)`, `cos(2π·LT/24)`.
Observed (train 2012–15, **[probe]**): mean log10 flux vs LT peaks at LT ≈ 11, troughs at LT ≈ 23, peak-to-trough ≈ 0.74 log10. (Empirical finding; the physical mechanism was not tested here.)

## 6. Index & gaps
Hourly index `2012-01-01 00:00 … 2017-12-31 23:00` UTC = **52,608 rows**. Linearly interpolate gaps ≤ 3 h in the four continuous channels; leave longer gaps as NaN.

## 7. `data/processed/hourly.csv` (output of `build_dataset.py`)
Columns, in order: `time_utc` (ISO-8601, hour start) · `log_flux` · `v_sw` · `bz_gsm` · `dst_nt` · `lt_hours`. NaN written as empty. No normalisation here (done in `train.py` with train-split stats).

## 8. Environment
numpy, pandas, scipy, matplotlib, torch (CUDA 12.8+ build for an RTX 5060 — Blackwell GPUs need it; verify with a small matmul on `cuda`; CPU is fine at this size), playwright (screenshots). Windows + Python 3.10+.
