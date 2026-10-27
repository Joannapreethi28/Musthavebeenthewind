# GEO electron-flux forecasting

Course project by Jabin Joseph M and Joanna Preethi J, LICET. Forecast electron flux using an LSTM and compare it with persistence and a daily-cycle baseline.

**Status: paused at the user's request; implementation is in progress.**

Environment setup, data download/validation, dataset construction, baseline checks, scientific tests and the overfit diagnostic have passed. A completed LSTM run is saved; the remaining training runs and final evaluation are pending. Evaluation, figure, offline report and screenshot code has been written and syntax-checked, but its final end-to-end execution is still pending.

- [Exact stopping point and resume commands](docs/HANDOFF.md)
- [Authoritative production protocol](docs/PROTOCOL_AMENDMENTS.md)
- [Data-quality measurements](results/data_quality.json)
- [Measured validation baselines](results/baselines.json)
- [Completed training run](results/runs/lstm_seed0.json)
- [Implementation plan](docs/IMPLEMENTATION_PLAN.md)

No final test performance is claimed. The final results table, figures, README and viva handoff will be generated from saved metrics after all required training completes.

Raw/processed data, the Python environment and model checkpoints remain local and Git-ignored. Source, methodology, download provenance, completed-run metadata and training history are tracked.
