import subprocess
import sys
from pathlib import Path

import pytest

from passos_wearable.config import load_config

ROOT = Path(__file__).resolve().parents[1]


def test_config_loads_and_env_override(monkeypatch):
    assert load_config()["estimation"]["window_s"] == 8.0
    monkeypatch.setenv("PASSOS_DATA_DIR", "/somewhere")
    assert load_config()["data"]["dir"] == "/somewhere"
    assert load_config(overrides={"estimation": {"window_s": 6.0}})["estimation"]["window_s"] == 6.0
    with pytest.raises(FileNotFoundError):
        load_config("nope.yaml")


def _cli(*a):
    return subprocess.run([sys.executable, str(ROOT / "app" / "cli.py"), *a], capture_output=True, text=True, cwd=ROOT)


def test_cli_synthetic_analyze():
    r = _cli("analyze", "--synthetic", "--start", "100", "--duration", "24")
    assert r.returncode == 0 and "SYNTHETIC" in r.stdout and "Estimated HR" in r.stdout


def test_cli_missing_dataset_exit_code(tmp_path):
    r = _cli("analyze", "--data-dir", str(tmp_path))
    assert r.returncode == 3 and "PPG-DaLiA" in r.stderr
    r = _cli("info", "--data-dir", str(tmp_path))
    assert r.returncode == 3 and "not found" in r.stderr
