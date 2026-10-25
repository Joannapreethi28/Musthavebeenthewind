**Step 1: environment setup**

Step 1 passed on 25 October 2026. The measured package versions, device, and individual checks are recorded in [environment.json](../results/setup/environment.json); the complete dependency snapshot is [requirements.freeze.txt](../results/setup/requirements.freeze.txt).

The working repository is `source/` inside the selected workspace. The working branch is `joanna`. The Git commit author is configured locally from the user's supplied identity. A dry-run push to `origin/joanna` succeeded before publishing the setup changes.

This machine reports Intel UHD Graphics. The environment therefore uses CPU PyTorch, which is allowed by the project plan. Python 3.13.15 and the project `.venv` are available. The RTX 5060 mentioned in the initial brief is not in the device listing.

Run this in ordinary Windows PowerShell from any directory:

```powershell
python "C:\Users\Jeevan\Documents\Musthavebeenthewind\source\scripts\setup_environment.py"
```

The script installs torch 2.11.0 from the official PyTorch CPU wheel index, then the remaining dependencies from PyPI. It uses binary wheels with TLS verification enabled and saves pip's installation reports inside `.venv`. The reports retain download URLs and available archive hashes; not every dependency has a hash recorded. The selected CPU build is listed in [PyTorch's official version matrix](https://pytorch.org/get-started/previous-versions/).

It checks dependency compatibility, imports all six required packages, and exercises an LSTM forward pass, backward pass, gradient clipping, and an AdamW update. The inputs are synthetic and are used only to check the environment. This does not download NOAA/NASA data or run a forecasting experiment. Playwright's browser download belongs to the later report step.

The checks write their actual results to `results/setup/environment.json`. Successful checks also write the installed versions to `results/setup/requirements.freeze.txt`. A failed check exits with a nonzero code and cannot mark setup complete.

To recheck without downloading or installing packages:

```powershell
python "C:\Users\Jeevan\Documents\Musthavebeenthewind\source\scripts\setup_environment.py" --check-only
```

**Verification outcome.** Dependency compatibility, all six package imports, and all seven LSTM checks passed. The user installed the packages from their own PowerShell because downloads from the agent session failed with Windows socket permission error 10013. The agent then independently reran the checks successfully. GPU availability is false; CPU execution is the accepted configuration. The diagnostic loss is from random synthetic inputs and is not a forecasting result.

For an exact dependency replay, install the pinned CPU torch build first, then install `results/setup/requirements.freeze.txt` with pip instead of the unpinned general requirements. Keep this snapshot with the measured environment report. Browser installation remains deferred until the report step.

**Data handling requirement for Step 2.** Download only the selected NOAA/NASA products over HTTPS, check the expected NetCDF/text formats and schemas, check dates and coverage, record SHA-256 hashes, and reject incomplete or unexpected responses before processing. A locally computed hash records integrity/provenance; it is not an independent guarantee that a file is harmless. Treat these files only as data and never execute downloaded content.
