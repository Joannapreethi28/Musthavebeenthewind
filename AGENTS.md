# AGENTS.md — brief for any coding agent (Codex / Claude Code)

**Read in this order:** `PLAN.md` → `docs/DATA_SPEC.md` → `docs/PHYSICS_AND_REFERENCES.md` → `docs/DESIGN_BRIEF.md` (only for the report/UI step).
`docs/VIVA_NOTES.md` is for the humans; skim it so your code comments match what they will have to explain.

## What this is
NNDL (Neural Networks & Deep Learning) course mini project by two students (Jabin Joseph M and Joanna Preethi J, B.Tech AI & Data Science, LICET).
Goal: a **basic, correct, well-understood** LSTM (optionally Transformer) that forecasts the >2 MeV electron flux at geostationary orbit **12 hours ahead**, plus a clean results page with screenshots they can show in a portfolio.
It is a ONE-DAY build. Simple and correct beats clever. The students must be able to explain every line in a viva.

## Non-negotiables
1. **Chronological split, never shuffled.** Train 2012-01-01..2015-12-31 · Val 2016 · Test 2017. Test is touched only for the final evaluation.
2. **Normalisation statistics come from the train split only.**
3. **Predict the change** `delta = log10F(t+12h) - log10F(t)`, output `log10F(t) + delta_hat`.
4. **Beat the right bar.** The bar is `persistence + diurnal` (PLAN.md §4), not plain persistence.
5. **No number is hand-typed.** README and report numbers are generated from `results/metrics.json`.
6. **Report negative results honestly.** If the deep model does not beat the bar, say so; that is a valid result.
7. **Verify, don't assume.** Run each phase's acceptance check (PLAN.md §9) and print the result before moving on. If a check fails, stop and fix it.
8. Set and log seeds (`torch`, `numpy`, `random`). Save every run's config next to its results.

## Scope guard — do NOT add
Other horizons, GRASP/ISRO data, backend/dashboard, extra features (E_y, Kp, Np, AE), custom/asymmetric losses, hyper-parameter sweeps beyond the tiny grid in PLAN.md §6, ablations, probabilistic outputs, extra datasets, packaging/CI. If you think something is missing, write it under "Suggestions" in your final message instead of building it.

## Code style
Small files, plain functions, type hints, a docstring saying *why* (not just what). No frameworks beyond PyTorch, NumPy, pandas, SciPy, Matplotlib. Use `pathlib`; the machine is Windows (also keep it Linux-safe). No notebooks required (an optional walkthrough notebook may be added at the end).

## UI work
Before touching `report/`, load the `impeccable` skill if it exists in your session (else `frontend-design`), then follow `docs/DESIGN_BRIEF.md`. The brief is the floor, not the ceiling.

## Final message format
What you built · acceptance checks (pass/fail with numbers) · anything you changed from PLAN.md and why · suggestions. Keep it short.
