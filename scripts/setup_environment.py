"""Prepare an isolated CPU environment without downloading any scientific data."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
VENV = ROOT / ".venv"


def run(command: list[str]) -> None:
    """Stop at the first failed setup stage so partial setup cannot look complete."""
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> int:
    """Use official HTTPS package indexes and record the environment after checking it."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true", help="Verify without installing.")
    args = parser.parse_args()
    if sys.version_info < (3, 10):
        parser.error("Python 3.10 or later is required.")

    python = VENV / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    try:
        if not python.is_file():
            if args.check_only:
                parser.error("The project .venv does not exist yet.")
            print("Creating the project .venv...", flush=True)
            run([sys.executable, "-m", "venv", str(VENV)])

        if not args.check_only:
            # No source distributions, user-site installs, or unsigned script downloads.
            # TLS verification remains enabled. A pip report records archive SHA-256 hashes.
            pip = [
                str(python), "-m", "pip", "--isolated", "install",
                "--disable-pip-version-check", "--no-cache-dir", "--only-binary=:all:",
                "--retries", "2", "--timeout", "30",
            ]
            print("Installing CPU PyTorch from download.pytorch.org...", flush=True)
            run(pip + [
                "--index-url", "https://download.pytorch.org/whl/cpu",
                "--report", str(VENV / "torch_install_report.json"), "torch==2.11.0",
            ])
            print("Installing the remaining dependencies from PyPI...", flush=True)
            run(pip + [
                "--index-url", "https://pypi.org/simple",
                "--report", str(VENV / "dependencies_install_report.json"),
                "-r", str(ROOT / "requirements.txt"),
            ])

        run([str(python), "-m", "pip", "check"])
        run([str(python), str(ROOT / "scripts" / "check_environment.py")])
        print("Environment checks passed. Scientific data has not been downloaded.", flush=True)
        return 0
    except subprocess.CalledProcessError as error:
        print(f"Setup stopped: a required command exited with code {error.returncode}.", file=sys.stderr)
        print("Fix the reported error, then rerun this same script; completed installs are reusable.", file=sys.stderr)
        return error.returncode or 1


if __name__ == "__main__":
    raise SystemExit(main())
