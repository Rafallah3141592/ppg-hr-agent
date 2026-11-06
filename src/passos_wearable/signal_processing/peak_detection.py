"""PPG peak detection and inter-beat-interval (IBI) processing."""
from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks


def detect_ppg_peaks(x: np.ndarray, fs: float, hr_range_bpm=(40, 200), prominence_std: float = 0.3
                     ) -> np.ndarray:
    """Systolic peak positions (fractional sample indices) in a band-passed PPG window."""
    x = np.asarray(x, dtype=float)
    min_dist = max(1, int(np.floor(fs * 60.0 / hr_range_bpm[1])))
    peaks, _ = find_peaks(x, distance=min_dist, prominence=prominence_std * np.std(x))
    return _refine(x, peaks)


def _refine(x: np.ndarray, peaks: np.ndarray) -> np.ndarray:
    """Parabolic interpolation around each peak for sub-sample timing."""
    out = peaks.astype(float)
    for k, i in enumerate(peaks):
        if 0 < i < len(x) - 1:
            y0, y1, y2 = x[i - 1], x[i], x[i + 1]
            den = y0 - 2 * y1 + y2
            if den != 0:
                out[k] = i + np.clip(0.5 * (y0 - y2) / den, -0.5, 0.5)
    return out


def peaks_to_ibi(peaks: np.ndarray, fs: float) -> np.ndarray:
    """Inter-beat intervals in milliseconds."""
    return np.diff(np.asarray(peaks, dtype=float)) / fs * 1000.0


def clean_ibi(ibi_ms: np.ndarray, ibi_range_ms=(300, 1500), max_rel_dev: float = 0.25
              ) -> tuple[np.ndarray, np.ndarray]:
    """Plausibility filter: keep IBIs in the physiological range and within ``max_rel_dev`` of the
    median (rejects missed beats = ~2x IBI and spurious beats = short IBI). Returns (clean, mask)."""
    ibi = np.asarray(ibi_ms, dtype=float)
    mask = np.isfinite(ibi) & (ibi >= ibi_range_ms[0]) & (ibi <= ibi_range_ms[1])
    if mask.any():
        med = np.median(ibi[mask])
        mask &= np.abs(ibi - med) <= max_rel_dev * med
    return ibi[mask], mask


def hr_from_ibi(ibi_ms: np.ndarray) -> float:
    """HR (bpm) from the median IBI; NaN if no IBI."""
    ibi = np.asarray(ibi_ms, dtype=float)
    return float(60000.0 / np.median(ibi)) if ibi.size else float("nan")
