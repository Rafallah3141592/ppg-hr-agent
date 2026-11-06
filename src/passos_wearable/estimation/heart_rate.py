"""Heart-rate estimation from wrist PPG (+ optional ACC).

Inputs are raw signal arrays only. Reference HR / ECG never enters this module (see
``tests/test_estimation.py::test_no_reference_leakage``).

Methods
-------
peaks            band-pass -> peak detection -> IBI -> plausibility filter -> median IBI -> bpm
spectral         dominant spectral peak of the band-passed PPG
spectral_motion  spectral peak after suppressing ACC-derived motion frequencies (needs ACC)
fusion           peaks + spectral agree -> mean; else motion-suppressed spectrum (or spectral if no ACC)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from passos_wearable.config import load_config
from passos_wearable.preprocessing.filters import motion_label, motion_level, preprocess_ppg
from passos_wearable.signal_processing.frequency_analysis import (
    motion_frequencies, motion_suppressed_hr, spectral_hr)
from passos_wearable.signal_processing.peak_detection import (
    clean_ibi, detect_ppg_peaks, hr_from_ibi, peaks_to_ibi)
from passos_wearable.signal_processing.signal_quality import SignalQuality, compute_sqi

log = logging.getLogger(__name__)
METHODS = ("peaks", "spectral", "spectral_motion", "fusion")


@dataclass
class HREstimate:
    hr_bpm: float
    method: str
    valid: bool
    confidence: str                    # high | medium | low | none
    candidates: dict[str, float] = field(default_factory=dict)
    sqi: SignalQuality | None = None
    motion_level_g: float | None = None
    motion_label: str = "unknown"
    motion_freqs_hz: list[float] = field(default_factory=list)
    n_peaks: int = 0
    n_valid_ibi: int = 0
    ibi_ms: list[float] = field(default_factory=list)
    method_disagreement_bpm: float | None = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "hr_bpm": None if not np.isfinite(self.hr_bpm) else round(float(self.hr_bpm), 2),
            "method": self.method, "valid": self.valid, "confidence": self.confidence,
            "candidates": {k: (None if not np.isfinite(v) else round(float(v), 2))
                           for k, v in self.candidates.items()},
            "sqi": self.sqi.to_dict() if self.sqi else None,
            "motion_level_g": self.motion_level_g, "motion_label": self.motion_label,
            "motion_freqs_hz": self.motion_freqs_hz, "n_peaks": self.n_peaks,
            "n_valid_ibi": self.n_valid_ibi,
            "method_disagreement_bpm": self.method_disagreement_bpm, "warnings": list(self.warnings),
        }


def _invalid(method: str, warning: str, **kw) -> HREstimate:
    return HREstimate(float("nan"), method, False, "none", warnings=[warning], **kw)


def estimate_hr(ppg: np.ndarray, fs_ppg: float, acc: np.ndarray | None = None, fs_acc: float | None = None,
                method: str | None = None, config: dict | None = None) -> HREstimate:
    """Estimate HR for ONE window of PPG (and optional ACC). Raises ValueError on invalid input."""
    cfg = config or load_config()
    e = cfg["estimation"]
    method = method or e["default_method"]
    if method not in METHODS:
        raise ValueError(f"Unknown method {method!r}; choose from {METHODS}")
    ppg = np.asarray(ppg, dtype=float)
    if ppg.ndim != 1:
        raise ValueError("ppg must be a 1-D array")
    if not np.all(np.isfinite(ppg)):
        raise ValueError("ppg contains non-finite values")
    if ppg.size < e["min_window_s"] * fs_ppg:
        raise ValueError(f"PPG window too short: {ppg.size / fs_ppg:.1f}s < {e['min_window_s']}s")
    have_acc = acc is not None and fs_acc is not None
    if method == "spectral_motion" and not have_acc:
        raise ValueError("Method 'spectral_motion' requires accelerometer data")
    warnings: list[str] = []
    if np.ptp(ppg) < 1e-9:
        return _invalid(method, "flat_signal")

    hr_lo, hr_hi = e["hr_range_bpm"]
    filt = preprocess_ppg(ppg, fs_ppg, cfg)

    # -- peaks -> IBI
    peaks = detect_ppg_peaks(filt, fs_ppg, (hr_lo, hr_hi), e["peak_prominence_std"])
    ibi_all = peaks_to_ibi(peaks, fs_ppg)
    ibi_clean, ibi_mask = clean_ibi(ibi_all, e["ibi_range_ms"], e["ibi_max_rel_dev"])
    hr_peaks = hr_from_ibi(ibi_clean) if len(ibi_clean) >= e["min_valid_ibis"] else float("nan")

    # -- spectral
    spec = spectral_hr(filt, fs_ppg, (hr_lo, hr_hi), e["spectral_nfft"])
    hr_spec = spec["hr_bpm"]

    # -- motion
    m_level, m_label, m_freqs, hr_motion = None, "unknown", [], float("nan")
    if have_acc:
        acc = np.asarray(acc, dtype=float)
        if acc.ndim == 2 and acc.shape[1] == 3 and np.all(np.isfinite(acc)) and acc.shape[0] > 0:
            try:
                m_level = motion_level(acc, fs_acc)
                m_label = motion_label(m_level, cfg)
                # Only trust ACC spectral peaks when there is real movement (absolute-level gate);
                # otherwise sensor noise would be mistaken for a movement frequency.
                m_freqs = ([] if m_label == "low" else
                           motion_frequencies(acc, fs_acc, min_share=e["motion_min_peak_share"]))
                sup = motion_suppressed_hr(spec, m_freqs, e["motion_exclusion_tol_hz"])
                hr_motion = sup["hr_bpm"]
                if sup["fallback"] and m_freqs:
                    warnings.append("all spectral candidates coincide with motion frequencies")
            except ValueError as err:
                warnings.append(f"accelerometer unusable: {err}")
                have_acc = False
        else:
            warnings.append("accelerometer invalid (shape or non-finite); motion analysis skipped")
            have_acc = False
        if method == "spectral_motion" and not have_acc:
            raise ValueError("Method 'spectral_motion' requires valid accelerometer data")

    cands = {"peaks": hr_peaks, "spectral": hr_spec}
    if have_acc:
        cands["spectral_motion"] = hr_motion
    disagreement = (abs(hr_peaks - hr_spec) if np.isfinite(hr_peaks) and np.isfinite(hr_spec) else None)

    # -- fusion
    if np.isfinite(hr_peaks) and np.isfinite(hr_spec):
        if disagreement <= e["agreement_tol_bpm"] and m_label != "high":
            fused = 0.5 * (hr_peaks + hr_spec)
        else:
            fused = hr_motion if (have_acc and np.isfinite(hr_motion)) else hr_spec
    else:
        fused = next((v for v in (hr_motion, hr_spec, hr_peaks) if np.isfinite(v)), float("nan"))
    cands["fusion"] = fused
    hr = cands.get(method, float("nan"))

    # -- plausibility + SQI
    plausible = np.isfinite(hr) and hr_lo <= hr <= hr_hi
    if not np.isfinite(hr):
        warnings.append(f"method '{method}' produced no estimate")
    elif not plausible:
        warnings.append(f"estimate {hr:.1f} bpm outside plausible range {hr_lo}-{hr_hi}")
    if disagreement is not None and disagreement > e["agreement_tol_bpm"]:
        warnings.append(f"peak-based and spectral HR disagree by {disagreement:.1f} bpm")
    sqi = compute_sqi(filt, fs_ppg, peaks, ibi_all, ibi_mask, ibi_clean, spec["freqs"], spec["psd"], hr, cfg)

    valid = bool(plausible)
    if not valid:
        conf = "none"
    elif sqi.label == "low" or (disagreement is not None and disagreement > e["agreement_tol_bpm"]
                                and m_label == "high"):
        conf = "low"
    elif sqi.label == "high" and (disagreement is None or disagreement <= e["agreement_tol_bpm"]) \
            and m_label != "high":
        conf = "high"
    else:
        conf = "medium"
    return HREstimate(float(hr) if valid else float("nan"), method, valid, conf, cands, sqi, m_level,
                      m_label, m_freqs, int(len(peaks)), int(len(ibi_clean)),
                      [round(float(v), 1) for v in ibi_clean], disagreement, warnings)


def estimate_hr_series(ppg: np.ndarray, fs_ppg: float, acc: np.ndarray | None, fs_acc: float | None,
                       method: str | None = None, config: dict | None = None,
                       window_s: float | None = None, step_s: float | None = None
                       ) -> tuple[np.ndarray, list[HREstimate]]:
    """Sliding-window estimation. Returns (window start times in s, estimates). A window that
    cannot be processed (e.g. NaN gap) yields an invalid estimate instead of aborting."""
    cfg = config or load_config()
    W = window_s or cfg["estimation"]["window_s"]
    S = step_s or cfg["estimation"]["step_s"]
    n_w, n_s = int(round(W * fs_ppg)), int(round(S * fs_ppg))
    if len(ppg) < n_w:
        raise ValueError("Signal shorter than one window")
    starts_idx = range(0, len(ppg) - n_w + 1, n_s)
    starts, out = [], []
    for i in starts_idx:
        t0 = i / fs_ppg
        a = None
        if acc is not None and fs_acc:
            j0 = int(round(t0 * fs_acc))
            a = acc[j0: j0 + int(round(W * fs_acc))]
        try:
            est = estimate_hr(ppg[i:i + n_w], fs_ppg, a, fs_acc, method, cfg)
        except ValueError as err:
            est = _invalid(method or cfg["estimation"]["default_method"], f"window rejected: {err}")
        starts.append(t0)
        out.append(est)
    return np.array(starts), out


def apply_temporal_continuity(hr: np.ndarray, max_jump_bpm: float = 12.0, max_hold: int = 4
                              ) -> tuple[np.ndarray, np.ndarray]:
    """Plausibility across time: an estimate that jumps more than ``max_jump_bpm`` from the last
    accepted value is replaced by that value (held) for at most ``max_hold`` consecutive windows,
    after which the new level is accepted. NaNs are filled the same way. Returns (hr, held_mask)."""
    hr = np.asarray(hr, dtype=float)
    out = hr.copy()
    held = np.zeros(len(hr), bool)
    last, n_hold = float("nan"), 0
    for i, v in enumerate(hr):
        if not np.isfinite(last):
            if np.isfinite(v):
                last = v
            continue
        if (not np.isfinite(v)) or abs(v - last) > max_jump_bpm:
            if n_hold < max_hold:
                out[i], held[i], n_hold = last, True, n_hold + 1
                continue
        last, n_hold = v, 0
    return out, held
