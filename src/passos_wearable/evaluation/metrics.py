"""Reference HR extraction and error metrics.

This is the ONLY place where reference HR is turned into numbers. The estimation pipeline never
imports this module.
"""
from __future__ import annotations

import numpy as np

from passos_wearable.data.loader import ReferenceData
from passos_wearable.signal_processing.peak_detection import clean_ibi


# ----------------------------------------------------------------------------- reference extraction
def reference_hr_from_labels(ref: ReferenceData, start_s: float, duration_s: float) -> dict:
    """Mean dataset-label HR over label windows fully inside [start, start+duration]. If none fit
    (segment shorter than one label window or misaligned), the label whose centre is nearest the
    segment centre is used and the result is flagged ``approximate``."""
    W, S = ref.label_window_s, ref.label_step_s
    end = start_s + duration_s
    i = np.arange(len(ref.hr_labels))
    inside = (i * S >= start_s - 1e-9) & (i * S + W <= end + 1e-9) & np.isfinite(ref.hr_labels)
    if inside.any():
        return {"hr_bpm": float(np.mean(ref.hr_labels[inside])), "n_labels": int(inside.sum()),
                "approximate": False}
    c = start_s + duration_s / 2.0
    k = int(np.argmin(np.abs(ref.label_centers_s - c))) if len(i) else None
    if k is None or not np.isfinite(ref.hr_labels[k]):
        return {"hr_bpm": float("nan"), "n_labels": 0, "approximate": True}
    return {"hr_bpm": float(ref.hr_labels[k]), "n_labels": 1, "approximate": True}


def reference_hr_from_rpeaks(ref: ReferenceData, start_s: float, duration_s: float) -> dict:
    """HR from ECG R-peak intervals inside the segment (median IBI after plausibility filtering)."""
    if ref.rpeaks is None or len(ref.rpeaks) == 0:
        return {"hr_bpm": float("nan"), "n_beats": 0}
    t = ref.rpeaks / ref.fs_ecg
    tt = t[(t >= start_s) & (t <= start_s + duration_s)]
    if len(tt) < 4:
        return {"hr_bpm": float("nan"), "n_beats": int(len(tt))}
    ibi, _ = clean_ibi(np.diff(tt) * 1000.0, (300, 2000), 0.3)
    return {"hr_bpm": float(60000.0 / np.median(ibi)) if len(ibi) else float("nan"), "n_beats": int(len(tt))}


def reference_for_windows(ref: ReferenceData, starts_s: np.ndarray, window_s: float) -> tuple[np.ndarray, np.ndarray]:
    """Per-window reference HR (dataset labels, ECG R-peaks) for windows [start, start+window]."""
    lab = np.array([reference_hr_from_labels(ref, s, window_s)["hr_bpm"] for s in starts_s])
    if ref.rpeaks is None or len(ref.rpeaks) == 0:
        return lab, np.full(len(starts_s), np.nan)
    ecg = np.array([reference_hr_from_rpeaks(ref, s, window_s)["hr_bpm"] for s in starts_s])
    return lab, ecg


# ----------------------------------------------------------------------------- metrics
def error_metrics(est, ref, tol_bpm: float = 5.0) -> dict:
    """MAE, RMSE, bias (est - ref), median absolute error, SD of error, Pearson r, share within tol.
    Pairs with NaN in either series are dropped; coverage = valid pairs / reference-available pairs."""
    est, ref = np.asarray(est, float), np.asarray(ref, float)
    if est.shape != ref.shape:
        raise ValueError(f"Shape mismatch: est {est.shape} vs ref {ref.shape}")
    has_ref = np.isfinite(ref)
    ok = has_ref & np.isfinite(est)
    n = int(ok.sum())
    out = {"n_windows": int(has_ref.sum()), "n_valid": n,
           "coverage": float(n / has_ref.sum()) if has_ref.any() else float("nan")}
    if n == 0:
        out.update({k: float("nan") for k in ("mae", "rmse", "bias", "medae", "std_error", "pearson_r", "within_tol")})
        return out
    e = est[ok] - ref[ok]
    out.update({
        "mae": float(np.mean(np.abs(e))), "rmse": float(np.sqrt(np.mean(e ** 2))), "bias": float(np.mean(e)),
        "medae": float(np.median(np.abs(e))), "std_error": float(np.std(e, ddof=1)) if n > 1 else float("nan"),
        "pearson_r": float(np.corrcoef(est[ok], ref[ok])[0, 1]) if n > 2 and np.std(est[ok]) > 0 and np.std(ref[ok]) > 0 else float("nan"),
        "within_tol": float(np.mean(np.abs(e) <= tol_bpm)),
    })
    return out


def group_metrics(est, ref, groups, tol_bpm: float = 5.0) -> dict:
    """``error_metrics`` per group label."""
    est, ref, groups = np.asarray(est, float), np.asarray(ref, float), np.asarray(groups)
    return {str(g): error_metrics(est[groups == g], ref[groups == g], tol_bpm) for g in np.unique(groups)}


def bland_altman(est, ref) -> dict:
    est, ref = np.asarray(est, float), np.asarray(ref, float)
    ok = np.isfinite(est) & np.isfinite(ref)
    d = est[ok] - ref[ok]
    if d.size < 2:
        return {"mean_diff": float("nan"), "loa_low": float("nan"), "loa_high": float("nan")}
    m, s = float(d.mean()), float(d.std(ddof=1))
    return {"mean_diff": m, "loa_low": m - 1.96 * s, "loa_high": m + 1.96 * s}
