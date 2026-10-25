"""Record real import and LSTM gradient checks before starting the data pipeline."""

from __future__ import annotations

from datetime import datetime, timezone
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "setup" / "environment.json"
PACKAGES = ("numpy", "pandas", "scipy", "matplotlib", "torch", "playwright")


def git_value(*arguments: str) -> str:
    """Record public repository context without reading authentication credentials."""
    result = subprocess.run(
        ["git", *arguments], cwd=ROOT, capture_output=True, text=True, check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def save_report(report: dict) -> None:
    """Write failures as well as passes so a prior success cannot hide a failed recheck."""
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main() -> int:
    """Exercise the required network architecture using explicitly synthetic inputs."""
    report = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "failed",
        "scope": "Environment check only; synthetic inputs, no scientific data or forecast scores.",
        "python": platform.python_version(),
        "platform": platform.system(),
        "machine": platform.machine(),
        "project_venv": Path(sys.prefix).resolve() == (ROOT / ".venv").resolve(),
        "git": {
            "branch": git_value("branch", "--show-current"),
            "commit": git_value("rev-parse", "HEAD"),
            "author_name": git_value("config", "user.name"),
            "author_email_configured": bool(git_value("config", "user.email")),
        },
        "packages": {},
    }
    if not report["project_venv"]:
        report["error"] = "Run this check with the project's .venv interpreter."
        save_report(report)
        print("FAIL: " + report["error"])
        return 1

    failures = {}
    for name in PACKAGES:
        try:
            importlib.import_module(name)
            report["packages"][name] = importlib.metadata.version(name)
            print(f"PASS import {name} {report['packages'][name]}", flush=True)
        except (ImportError, OSError, RuntimeError) as error:
            failures[name] = f"{type(error).__name__}: {error}"
    if failures:
        report["import_failures"] = failures
        save_report(report)
        print("FAIL: missing or unusable dependencies: " + ", ".join(failures), flush=True)
        print(f"Details: {OUTPUT}", flush=True)
        return 1

    try:
        import numpy as np
        import torch
        from torch import nn

        seed = 0
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
        torch.use_deterministic_algorithms(True)
        # CPU is intentional: the detected machine has Intel UHD Graphics.
        device = torch.device("cpu")
        report["device"] = {
            "selected": str(device),
            "cuda_available": torch.cuda.is_available(),
            "torch_cuda_build": torch.version.cuda,
            "cpu_threads": torch.get_num_threads(),
        }
        inputs = torch.from_numpy(np.random.standard_normal((8, 72, 6)).astype("float32"))
        targets = torch.randn(8, 1, device=device)
        lstm = nn.LSTM(6, 64, num_layers=2, dropout=0.2, batch_first=True).to(device)
        head = nn.Linear(64, 1).to(device)
        parameters = list(lstm.parameters()) + list(head.parameters())
        optimizer = torch.optim.AdamW(parameters, lr=0.001, weight_decay=0.0001)
        before = head.weight.detach().clone()
        started = time.perf_counter()
        sequence, _ = lstm(inputs.to(device))
        prediction = head(sequence[:, -1, :])
        loss = nn.functional.mse_loss(prediction, targets)
        loss.backward()
        finite_gradients = all(
            parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
            for parameter in parameters
        )
        gradient_norm = float(nn.utils.clip_grad_norm_(parameters, 1.0).item())
        optimizer.step()
        checks = {
            "output_shape_correct": list(prediction.shape) == [8, 1],
            "loss_finite": math.isfinite(float(loss.item())),
            "gradients_finite": finite_gradients and math.isfinite(gradient_norm),
            "gradient_nonzero": gradient_norm > 0,
            "optimizer_changed_weights": not torch.equal(before, head.weight.detach()),
            "updated_parameters_finite": all(bool(torch.isfinite(p).all()) for p in parameters),
            "parameter_count_correct": sum(p.numel() for p in parameters) == 51777,
        }
        report["smoke_check"] = {
            "seed": seed,
            "input_shape": list(inputs.shape),
            "output_shape": list(prediction.shape),
            "parameter_count": sum(p.numel() for p in parameters),
            "synthetic_mse": float(loss.item()) if math.isfinite(float(loss.item())) else None,
            "gradient_norm_before_clip": gradient_norm if math.isfinite(gradient_norm) else None,
            "elapsed_seconds": time.perf_counter() - started,
            "checks": checks,
        }
        for name, passed in checks.items():
            print(f"{'PASS' if passed else 'FAIL'} {name}", flush=True)
        if not all(checks.values()):
            report["error"] = "At least one LSTM forward/backward/update check failed."
            save_report(report)
            return 1

        freeze = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=True,
        ).stdout
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        (OUTPUT.parent / "requirements.freeze.txt").write_text(freeze, encoding="utf-8")
        report["status"] = "passed"
        save_report(report)
        print(f"PASS environment: CPU LSTM with {report['smoke_check']['parameter_count']:,} parameters", flush=True)
        print(f"Recorded versions and checks: {OUTPUT}", flush=True)
        return 0
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        save_report(report)
        print("FAIL: " + report["error"], file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
