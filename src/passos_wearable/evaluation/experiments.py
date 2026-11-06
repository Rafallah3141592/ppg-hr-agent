"""Sliding-window experiment runner: estimation first, THEN reference-based evaluation.

Stage 1 (``run_estimation``) sees only PPG/ACC. Stage 2 (``attach_reference``) adds reference HR
and activity context to the result table; no estimate is modified in stage 2.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from passos_wearable.config import load_config
from passos_wearable.data.loader import ACTIVITY_NAMES, Recording
from passos_wearable.estimation.heart_rate import apply_temporal_continuity, estimate_hr_series
from passos_wearable.evaluation.metrics import error_metrics, group_metrics, reference_for_windows

BASE_METHODS = ["peaks", "spectral", "spectral_motion", "fusion"]
ALL_METHODS = BASE_METHODS + ["fusion_continuity"]


def run_estimation(rec: Recording, config: dict | None = None, t_start: float = 0.0,
                   t_end: float | None = None, max_windows: int | None = None) -> pd.DataFrame:
    """Estimate HR for every window. Uses ``rec.signals`` only."""
    cfg = config or load_config()
    W, S = cfg["estimation"]["window_s"], cfg["estimation"]["step_s"]
    s = rec.signals
    t_end = min(t_end if t_end is not None else s.duration_s, s.duration_s)
    seg = s.segment(t_start, t_end - t_start)
    starts, ests = estimate_hr_series(seg.bvp, seg.fs_bvp, seg.acc, seg.fs_acc, "fusion", cfg)
    if max_windows:
        starts, ests = starts[:max_windows], ests[:max_windows]
    df = pd.DataFrame({
        "start_s": starts + t_start,
        **{f"hr_{m}": [e.candidates.get(m, np.nan) for e in ests] for m in BASE_METHODS},
        "sqi": [e.sqi.score if e.sqi else np.nan for e in ests],
        "sqi_label": [e.sqi.label if e.sqi else "none" for e in ests],
        "motion_g": [e.motion_level_g if e.motion_level_g is not None else np.nan for e in ests],
        "motion_label": [e.motion_label for e in ests],
        "confidence": [e.confidence for e in ests],
        "disagreement_bpm": [e.method_disagreement_bpm if e.method_disagreement_bpm is not None else np.nan
                             for e in ests],
    })
    df["center_s"] = df["start_s"] + W / 2.0
    e = cfg["estimation"]
    df["hr_fusion_continuity"], _ = apply_temporal_continuity(
        df["hr_fusion"].to_numpy(), e["continuity_max_jump_bpm"], e["continuity_max_hold"])
    return df


def attach_reference(df: pd.DataFrame, rec: Recording, config: dict | None = None) -> pd.DataFrame:
    """Add reference HR (dataset labels + ECG R-peaks) and activity annotation (evaluation context)."""
    cfg = config or load_config()
    W = cfg["estimation"]["window_s"]
    out = df.copy()
    out["ref_label_hr"], out["ref_ecg_hr"] = reference_for_windows(rec.reference, out["start_s"].to_numpy(), W)
    s = rec.signals
    if s.activity is not None:
        modes = []
        for t0 in out["start_s"]:
            a = s.activity[int(t0 * s.fs_activity): int((t0 + W) * s.fs_activity)]
            modes.append(int(np.bincount(a).argmax()) if len(a) else -1)
        out["activity_code"] = modes
        out["activity"] = [ACTIVITY_NAMES.get(m, "unknown") for m in modes]
    else:
        out["activity_code"], out["activity"] = -1, "unknown"
    return out


def summarize(df: pd.DataFrame, methods=ALL_METHODS, ref_col: str = "ref_label_hr", tol: float = 5.0
              ) -> pd.DataFrame:
    rows = []
    for m in methods:
        r = error_metrics(df[f"hr_{m}"], df[ref_col], tol)
        rows.append({"method": m, **r})
    return pd.DataFrame(rows).set_index("method")


def summarize_by(df: pd.DataFrame, group_col: str, methods=ALL_METHODS, ref_col: str = "ref_label_hr",
                 tol: float = 5.0) -> pd.DataFrame:
    rows = []
    for m in methods:
        for g, r in group_metrics(df[f"hr_{m}"], df[ref_col], df[group_col], tol).items():
            rows.append({"method": m, group_col: g, **r})
    return pd.DataFrame(rows)


def sqi_gating(df: pd.DataFrame, method: str = "fusion", ref_col: str = "ref_label_hr",
               tol: float = 5.0) -> pd.DataFrame:
    """Accuracy vs coverage when low-confidence windows are rejected."""
    rows = []
    for name, keep in [("all windows", df["confidence"] != "__none__"),
                       ("confidence >= medium", df["confidence"].isin(["high", "medium"])),
                       ("confidence == high", df["confidence"] == "high")]:
        est = df[f"hr_{method}"].where(keep)
        r = error_metrics(est, df[ref_col], tol)
        rows.append({"filter": name, **r, "kept_fraction": float(keep.mean())})
    return pd.DataFrame(rows)


def worst_windows(df: pd.DataFrame, method: str = "fusion", ref_col: str = "ref_label_hr", n: int = 10
                  ) -> pd.DataFrame:
    d = df.copy()
    d["abs_error"] = (d[f"hr_{method}"] - d[ref_col]).abs()
    cols = ["start_s", "activity", f"hr_{method}", ref_col, "abs_error", "sqi", "motion_g", "confidence"]
    return d.sort_values("abs_error", ascending=False).head(n)[[c for c in cols if c in d.columns]]
