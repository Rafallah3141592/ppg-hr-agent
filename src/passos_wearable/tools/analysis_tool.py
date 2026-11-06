"""The analysis function exposed to the AI agent (and callable directly).

``analyze_hr_segment`` always returns a JSON-serialisable dict. Failures are returned as
``status="error"`` results (never raised) so an LLM can read and relay them. Estimation and
reference evaluation are separate steps: the reference is only attached after the estimate is final
and is placed under the ``evaluation`` key.
"""
from __future__ import annotations

import logging
import math
import re
from typing import Any

import numpy as np

from passos_wearable.config import load_config
from passos_wearable.data.loader import (ACTIVITY_NAMES, DataError, Recording, load_subject,
                                         validate_subject)
from passos_wearable.data.synthetic import make_synthetic_recording
from passos_wearable.estimation.heart_rate import METHODS, estimate_hr_series
from passos_wearable.evaluation.metrics import reference_hr_from_labels, reference_hr_from_rpeaks

log = logging.getLogger(__name__)

TOOL_SCHEMA = {
    "name": "analyze_hr_segment",
    "description": (
        "Estimate heart rate (bpm) from the wrist PPG of PPG-DaLiA participant S1 over a time segment. "
        "Returns the estimate, alternative methods, signal quality, movement level, plausibility checks, "
        "and (separately) a comparison with the ECG-derived reference HR when requested. "
        "Not a medical device; do not use for diagnosis."),
    "input_schema": {
        "type": "object",
        "properties": {
            "subject": {"type": "string", "description": "Participant id; only 'S1' is supported.", "default": "S1"},
            "start_s": {"type": "number", "description": "Segment start in seconds from recording start (>= 0)."},
            "duration_s": {"type": "number", "description": "Segment length in seconds (8 to 300)."},
            "method": {"type": "string", "enum": list(METHODS),
                       "description": "Estimation method; default 'fusion'."},
            "include_reference": {"type": "boolean",
                                  "description": "Also compare with the reference HR (evaluation only)."},
        },
        "required": ["start_s", "duration_s"],
    },
}

_BASE_CAVEATS = [
    "Classical signal-processing estimate from wrist PPG; not validated for clinical use.",
    "Window-level HR is derived from 8 s windows, so rapid HR changes are smoothed.",
]


def _error(msg: str, request: dict) -> dict:
    return {"status": "error", "error_message": msg, "request": request}


def _num(x: Any) -> float | None:
    return None if x is None or not np.isfinite(x) else round(float(x), 2)


def analyze_hr_segment(subject: str = "S1", start_s: float = 0.0, duration_s: float = 30.0,
                       method: str | None = None, include_reference: bool = True,
                       data_dir: str | None = None, synthetic: bool = False,
                       config: dict | None = None, recording: Recording | None = None) -> dict:
    """Analyse one segment. ``recording`` may be injected (tests, notebooks); ``synthetic=True``
    uses generated data that is labelled as such in the result."""
    request = {"subject": subject, "start_s": start_s, "duration_s": duration_s, "method": method,
               "include_reference": include_reference}
    try:
        cfg = config or load_config()
        a = cfg["analysis"]
        method = method or cfg["estimation"]["default_method"]
        request["method"] = method
        if method not in METHODS:
            return _error(f"Unknown method {method!r}. Choose one of {list(METHODS)}.", request)
        try:
            start_s, duration_s = float(start_s), float(duration_s)
        except (TypeError, ValueError):
            return _error("start_s and duration_s must be numbers.", request)
        if not (math.isfinite(start_s) and math.isfinite(duration_s)):
            return _error("start_s and duration_s must be finite numbers.", request)
        if start_s < 0:
            return _error("start_s must be >= 0.", request)
        if not (a["min_duration_s"] <= duration_s <= a["max_duration_s"]):
            return _error(f"duration_s must be between {a['min_duration_s']:g} and {a['max_duration_s']:g} seconds.",
                          request)
        validate_subject(subject, cfg)

        rec = recording
        if rec is None:
            rec = (make_synthetic_recording(config=cfg, subject=subject) if synthetic
                   else load_subject(subject, data_dir, cfg))
        if start_s + duration_s > rec.signals.duration_s + 1e-6:
            return _error(f"Segment ends at {start_s + duration_s:.1f}s but the recording is only "
                          f"{rec.signals.duration_s:.1f}s long.", request)

        # ---- stage 1: estimation (signals only) ----
        seg = rec.signals.segment(start_s, duration_s)
        starts, ests = estimate_hr_series(seg.bvp, seg.fs_bvp, seg.acc, seg.fs_acc, method, cfg)
        hrs = np.array([e.hr_bpm for e in ests])
        valid = np.isfinite(hrs)
        n_win = len(ests)
        hr = float(np.median(hrs[valid])) if valid.any() else float("nan")
        spread = float(np.std(hrs[valid])) if valid.sum() > 1 else float("nan")
        conf = [e.confidence for e in ests]
        frac_high, frac_good = conf.count("high") / n_win, (conf.count("high") + conf.count("medium")) / n_win
        valid_frac = float(valid.mean())
        if not valid.any():
            overall = "none"
        elif frac_high >= 0.6 and (not np.isfinite(spread) or spread <= 5):
            overall = "high"
        elif frac_good < 0.4 or valid_frac < 0.5 or (np.isfinite(spread) and spread > 10):
            overall = "low"
        else:
            overall = "medium"

        alts = {}
        for m in METHODS:
            vals = np.array([e.candidates.get(m, np.nan) for e in ests], float)
            alts[m] = _num(np.nanmedian(vals)) if np.isfinite(vals).any() else None
        sqi_scores = [e.sqi.score for e in ests if e.sqi]
        mot = [e.motion_level_g for e in ests if e.motion_level_g is not None]
        mean_motion = float(np.mean(mot)) if mot else float("nan")
        from passos_wearable.preprocessing.filters import motion_label as _ml
        quality = {
            "mean_sqi": _num(np.mean(sqi_scores)) if sqi_scores else None,
            "label": max(set(e.sqi.label for e in ests if e.sqi), key=[e.sqi.label for e in ests if e.sqi].count)
            if sqi_scores else "unknown",
            "fraction_low_quality_windows": _num(np.mean([e.sqi.label == "low" for e in ests if e.sqi])) if sqi_scores else None,
            "flag_counts": _count([f for e in ests if e.sqi for f in e.sqi.flags]),
        }
        motion = {"mean_level_g": _num(mean_motion), "label": _ml(mean_motion, cfg) if np.isfinite(mean_motion) else "unknown",
                  "fraction_high_motion_windows": _num(np.mean([e.motion_label == "high" for e in ests]))}
        warn_counts = _count([re.sub(r"\d+(?:\.\d+)?", "N", w.split(":")[0] if w.startswith("window rejected") else w)
                              for e in ests for w in e.warnings])
        plaus = {"valid_window_fraction": _num(valid_frac), "n_windows": n_win,
                 "in_range_bpm": cfg["estimation"]["hr_range_bpm"],
                 "mean_method_disagreement_bpm": _num(np.nanmean([e.method_disagreement_bpm for e in ests
                                                                   if e.method_disagreement_bpm is not None]))
                 if any(e.method_disagreement_bpm is not None for e in ests) else None,
                 "warnings": warn_counts}

        context = {}
        if seg.activity is not None and len(seg.activity):
            codes, counts = np.unique(seg.activity, return_counts=True)
            context = {"annotated_activity_fractions": {ACTIVITY_NAMES.get(int(c), str(c)): round(float(n / counts.sum()), 2)
                                                         for c, n in zip(codes, counts)}}

        result: dict[str, Any] = {
            "status": "ok", "request": request,
            "data_source": rec.source_tag, "is_synthetic": rec.is_synthetic,
            "segment": {"start_s": start_s, "end_s": start_s + duration_s, "duration_s": duration_s,
                        "n_windows": n_win, "window_s": cfg["estimation"]["window_s"],
                        "step_s": cfg["estimation"]["step_s"]},
            "estimate": {"method": method, "hr_bpm": _num(hr), "window_to_window_std_bpm": _num(spread),
                         "window_hr_min_bpm": _num(np.min(hrs[valid])) if valid.any() else None,
                         "window_hr_max_bpm": _num(np.max(hrs[valid])) if valid.any() else None,
                         "confidence": overall},
            "alternatives_bpm": alts, "signal_quality": quality, "motion": motion,
            "plausibility": plaus, "context": context, "evaluation": None,
            "caveats": list(_BASE_CAVEATS),
        }
        if rec.is_synthetic:
            result["caveats"].append("SYNTHETIC data: this result says nothing about real PPG-DaLiA performance.")
        if motion["label"] == "high":
            result["caveats"].append("High movement: wrist PPG is strongly affected by motion artefacts; treat with caution.")
        if quality["label"] == "low" or overall == "low":
            result["caveats"].append("Low signal quality or consistency: the estimate may be unreliable.")
        if overall == "none":
            result["caveats"].append("No valid estimate could be produced for this segment.")

        # ---- stage 2: reference evaluation (never feeds back into the estimate) ----
        if include_reference:
            lab = reference_hr_from_labels(rec.reference, start_s, duration_s)
            ecg = reference_hr_from_rpeaks(rec.reference, start_s, duration_s)
            ev: dict[str, Any] = {"note": "Reference used for evaluation only; it was not an input to the estimate.",
                                  "dataset_label_hr_bpm": _num(lab["hr_bpm"]), "label_alignment_approximate": lab["approximate"],
                                  "ecg_rpeak_hr_bpm": _num(ecg["hr_bpm"])}
            if np.isfinite(hr) and np.isfinite(lab["hr_bpm"]):
                err = hr - lab["hr_bpm"]
                ev["signed_error_bpm"], ev["abs_error_bpm"] = _num(err), _num(abs(err))
            result["evaluation"] = ev
        return result
    except (DataError, ValueError) as e:
        log.warning("analysis failed: %s", e)
        return _error(str(e), request)
    except FileNotFoundError as e:
        return _error(str(e), request)


def _count(items: list[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for i in items:
        out[i] = out.get(i, 0) + 1
    return out
