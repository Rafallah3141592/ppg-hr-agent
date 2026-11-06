"""Signal filtering and motion-level helpers (no reference data is used here)."""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt


def bandpass(x: np.ndarray, fs: float, low: float, high: float, order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth band-pass."""
    x = np.asarray(x, dtype=float)
    nyq = fs / 2.0
    if not (0 < low < high < nyq):
        raise ValueError(f"Invalid band [{low}, {high}] Hz for fs={fs} Hz")
    if not np.all(np.isfinite(x)):
        raise ValueError("Signal contains non-finite values")
    sos = butter(order, [low / nyq, high / nyq], btype="band", output="sos")
    min_len = 3 * (2 * sos.shape[0] + 1) + 1
    if x.size < min_len:
        raise ValueError(f"Signal too short ({x.size} samples) for the filter (needs >= {min_len})")
    return sosfiltfilt(sos, x)


def preprocess_ppg(ppg: np.ndarray, fs: float, config: dict) -> np.ndarray:
    """Band-pass (default 0.5-4 Hz = 30-240 bpm) and standardise a PPG window."""
    lo, hi = config["preprocessing"]["ppg_bandpass_hz"]
    y = bandpass(ppg, fs, lo, hi, config["preprocessing"]["filter_order"])
    s = np.std(y)
    return y / s if s > 1e-12 else y


def acc_magnitude(acc: np.ndarray) -> np.ndarray:
    acc = np.asarray(acc, dtype=float)
    if acc.ndim != 2 or acc.shape[1] != 3:
        raise ValueError(f"ACC must have shape (N, 3); got {acc.shape}")
    return np.linalg.norm(acc, axis=1)


def dynamic_acc_magnitude(acc: np.ndarray, fs: float) -> np.ndarray:
    """Magnitude with gravity/posture removed (0.3-8 Hz band-pass)."""
    return bandpass(acc_magnitude(acc), fs, 0.3, min(8.0, fs / 2 - 0.5), order=2)


def motion_level(acc: np.ndarray, fs: float) -> float:
    """Std of dynamic acceleration magnitude (g). Larger = more movement."""
    return float(np.std(dynamic_acc_magnitude(acc, fs)))


def motion_label(level_g: float, config: dict) -> str:
    m = config["motion"]
    return "low" if level_g < m["low_std_g"] else ("high" if level_g >= m["high_std_g"] else "medium")
