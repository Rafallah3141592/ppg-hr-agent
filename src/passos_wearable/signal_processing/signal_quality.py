"""PPG signal-quality indicators (SQI). Thresholds are heuristic and configurable."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np


@dataclass
class SignalQuality:
    spectral_concentration: float      # share of in-band power around the HR frequency (+ 2nd harmonic)
    template_correlation: float        # mean correlation of single beats with the average beat (NaN if <3 beats)
    ibi_cv: float                      # coefficient of variation of valid IBIs (NaN if <3)
    n_peaks: int
    valid_ibi_fraction: float
    score: float                       # 0..1 weighted combination
    label: str                         # high | medium | low
    flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        return {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in d.items()}


def spectral_concentration(f: np.ndarray, p: np.ndarray, f0: float, halfwidth: float,
                           band_hz=(0.5, 4.0)) -> float:
    m = (f >= band_hz[0]) & (f <= band_hz[1])
    total = p[m].sum()
    if not np.isfinite(f0) or total <= 0:
        return 0.0
    near = (np.abs(f - f0) <= halfwidth) | (np.abs(f - 2 * f0) <= halfwidth)
    return float(min(1.0, p[m & near].sum() / total))


def template_correlation(x: np.ndarray, peaks: np.ndarray, fs: float, median_ibi_ms: float) -> float:
    """Mean Pearson correlation between each beat (peak -0.35T .. +0.65T) and the mean beat."""
    T = median_ibi_ms / 1000.0
    pre, post = int(0.35 * T * fs), int(0.65 * T * fs)
    beats = [x[int(round(pk)) - pre: int(round(pk)) + post] for pk in peaks
             if int(round(pk)) - pre >= 0 and int(round(pk)) + post <= len(x)]
    beats = [b for b in beats if len(b) == pre + post]
    if len(beats) < 3:
        return float("nan")
    B = np.vstack(beats)
    tmpl = B.mean(axis=0)
    if np.std(tmpl) < 1e-12:
        return float("nan")
    cors = [np.corrcoef(b, tmpl)[0, 1] for b in B if np.std(b) > 1e-12]
    return float(np.mean(cors)) if cors else float("nan")


def compute_sqi(x_filt: np.ndarray, fs: float, peaks: np.ndarray, ibi_all_ms: np.ndarray,
                ibi_valid_mask: np.ndarray, ibi_clean_ms: np.ndarray, f: np.ndarray, p: np.ndarray,
                hr_bpm: float, config: dict) -> SignalQuality:
    q = config["quality"]
    conc = spectral_concentration(f, p, hr_bpm / 60.0 if np.isfinite(hr_bpm) else np.nan,
                                  q["spectral_halfwidth_hz"])
    n_valid = int(len(ibi_clean_ms))
    valid_frac = float(np.mean(ibi_valid_mask)) if len(ibi_valid_mask) else 0.0
    cv = float(np.std(ibi_clean_ms) / np.mean(ibi_clean_ms)) if n_valid >= 3 else float("nan")
    tc = (template_correlation(x_filt, peaks, fs, float(np.median(ibi_clean_ms)))
          if n_valid >= 3 else float("nan"))
    c_tc = 0.0 if np.isnan(tc) else max(0.0, tc)
    c_cv = 0.0 if np.isnan(cv) else 1.0 - min(cv / 0.25, 1.0)
    score = float(0.4 * conc + 0.3 * c_tc + 0.3 * c_cv)
    label = "high" if score >= q["sqi_high"] else ("medium" if score >= q["sqi_medium"] else "low")
    flags = []
    if conc < 0.3:
        flags.append("low_spectral_concentration")
    if n_valid < 3:
        flags.append("insufficient_valid_beats")
    elif cv > 0.15:
        flags.append("irregular_ibi")
    if valid_frac < 0.6 and len(ibi_valid_mask):
        flags.append("many_rejected_ibis")
    return SignalQuality(conc, tc, cv, int(len(peaks)), valid_frac, score, label, flags)
