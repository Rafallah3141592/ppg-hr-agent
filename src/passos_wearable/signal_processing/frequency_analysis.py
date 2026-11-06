"""Spectral HR estimation, including accelerometer-guided motion-frequency suppression."""
from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks, periodogram


def power_spectrum(x: np.ndarray, fs: float, nfft: int = 8192) -> tuple[np.ndarray, np.ndarray]:
    """Hann-windowed, zero-padded periodogram (zero padding only interpolates; true resolution is 1/T)."""
    x = np.asarray(x, dtype=float)
    return periodogram(x - x.mean(), fs=fs, window="hann", nfft=max(nfft, len(x)), detrend=False)


def spectral_peaks(f: np.ndarray, p: np.ndarray, band_hz: tuple[float, float], n: int = 5
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Up to ``n`` strongest local maxima inside ``band_hz``, sorted by power (descending)."""
    m = (f >= band_hz[0]) & (f <= band_hz[1])
    idx_band = np.flatnonzero(m)
    if idx_band.size < 3:
        return np.array([]), np.array([])
    loc, _ = find_peaks(p[idx_band])
    if loc.size == 0:
        k = int(np.argmax(p[idx_band]))
        return np.array([f[idx_band][k]]), np.array([p[idx_band][k]])
    idx = idx_band[loc]
    order = np.argsort(p[idx])[::-1][:n]
    return f[idx][order], p[idx][order]


def spectral_hr(ppg_filt: np.ndarray, fs: float, hr_range_bpm=(40, 200), nfft: int = 8192) -> dict:
    """Dominant-frequency HR. Returns hr_bpm, freq_hz, and the spectrum for reuse."""
    f, p = power_spectrum(ppg_filt, fs, nfft)
    fr, pr = spectral_peaks(f, p, (hr_range_bpm[0] / 60.0, hr_range_bpm[1] / 60.0), n=6)
    if fr.size == 0:
        return {"hr_bpm": float("nan"), "freq_hz": float("nan"), "freqs": f, "psd": p,
                "peak_freqs": fr, "peak_powers": pr}
    return {"hr_bpm": float(fr[0] * 60.0), "freq_hz": float(fr[0]), "freqs": f, "psd": p,
            "peak_freqs": fr, "peak_powers": pr}


def motion_frequencies(acc: np.ndarray, fs_acc: float, band_hz=(0.5, 4.0), n: int = 2,
                       min_share: float = 0.15, nfft: int = 4096) -> list[float]:
    """Dominant movement frequencies from the 3-axis ACC spectrum (summed over axes).

    A peak counts only if it holds at least ``min_share`` of the in-band power, so near-static
    recordings (white-ish ACC noise) yield no motion frequencies."""
    acc = np.asarray(acc, dtype=float)
    psd = None
    for c in range(3):
        f, p = power_spectrum(acc[:, c], fs_acc, nfft)
        psd = p if psd is None else psd + p
    m = (f >= band_hz[0]) & (f <= band_hz[1])
    total = psd[m].sum()
    if total <= 0:
        return []
    fr, pr = spectral_peaks(f, psd, band_hz, n=n)
    # share = power in +-0.1 Hz around the peak / in-band power
    out = []
    for fk in fr:
        share = psd[m & (np.abs(f - fk) <= 0.1)].sum() / total
        if share >= min_share:
            out.append(float(fk))
    return out


def motion_suppressed_hr(ppg_spec: dict, motion_freqs_hz: list[float], tol_hz: float = 0.12) -> dict:
    """Pick the strongest PPG spectral peak that is NOT at a motion frequency or its 2nd harmonic.

    Known failure: if the true HR coincides with the movement cadence (e.g. brisk walking), the
    correct peak is rejected. When every candidate is rejected we fall back to the plain peak."""
    fr, pr = ppg_spec["peak_freqs"], ppg_spec["peak_powers"]
    if fr.size == 0:
        return {"hr_bpm": float("nan"), "freq_hz": float("nan"), "fallback": True, "suppressed": []}
    banned = [h * m for m in motion_freqs_hz for h in (1.0, 2.0)]
    suppressed = []
    for fk, pk in zip(fr, pr):
        if any(abs(fk - b) <= tol_hz for b in banned):
            suppressed.append(float(fk * 60.0))
            continue
        return {"hr_bpm": float(fk * 60.0), "freq_hz": float(fk), "fallback": False, "suppressed": suppressed}
    return {"hr_bpm": float(fr[0] * 60.0), "freq_hz": float(fr[0]), "fallback": True, "suppressed": suppressed}
