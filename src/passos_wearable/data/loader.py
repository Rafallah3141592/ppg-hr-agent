"""PPG-DaLiA loading, validation, synchronisation and segment selection.

Dataset layout expected (as distributed by UCI, dataset 495)::

    <data.dir>/S1/S1.pkl

The pickle holds (Reiss et al., 2019): ``signal.wrist.BVP`` (64 Hz), ``signal.wrist.ACC`` (32 Hz, Nx3),
``label`` (reference HR, one value per 8 s window shifted by 2 s), ``activity`` (4 Hz),
``rpeaks`` (ECG R-peak sample indices at 700 Hz).

Design rule: *signals* and *reference* are separate containers. Everything the estimation
pipeline may see lives in :class:`WristSignals` / :class:`SignalSegment`; the reference HR lives in
:class:`ReferenceData` and is only consumed by ``passos_wearable.evaluation``.

Security note: ``.pkl`` files are Python pickles; only load the official dataset you downloaded yourself.
"""
from __future__ import annotations

import logging
import pickle
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from passos_wearable.config import load_config, resolve_path

log = logging.getLogger(__name__)

ACTIVITY_NAMES = {
    0: "transient", 1: "sitting", 2: "stairs", 3: "table_soccer", 4: "cycling",
    5: "driving", 6: "lunch", 7: "walking", 8: "working",
}
_SUBJECT_RE = re.compile(r"^S\d{1,2}$")

DATASET_HELP = (
    "PPG-DaLiA was not found. Download 'PPG-DaLiA' (UCI ML Repository, dataset 495, "
    "https://archive.ics.uci.edu/dataset/495/ppg+dalia), unzip it, and place the folder so that "
    "<data.dir>/S1/S1.pkl exists (default data.dir = data/raw/PPG_FieldStudy, or set PASSOS_DATA_DIR). "
    "See data/README.md. Use --synthetic only for smoke tests (NOT real results)."
)


class DataError(ValueError):
    """Raised when data are malformed or fail validation."""


class DatasetNotFoundError(FileNotFoundError):
    """Raised when the dataset (or a requested subject) is not on disk."""


@dataclass
class SignalSegment:
    """Signals for one time segment. Contains NO reference HR."""
    start_s: float
    duration_s: float
    bvp: np.ndarray
    fs_bvp: float
    acc: np.ndarray
    fs_acc: float
    activity: np.ndarray | None = None
    fs_activity: float = 4.0


@dataclass
class WristSignals:
    bvp: np.ndarray
    fs_bvp: float
    acc: np.ndarray
    fs_acc: float
    activity: np.ndarray | None = None
    fs_activity: float = 4.0

    @property
    def duration_s(self) -> float:
        return float(min(len(self.bvp) / self.fs_bvp, len(self.acc) / self.fs_acc))

    def segment(self, start_s: float, duration_s: float) -> SignalSegment:
        if not np.isfinite(start_s) or not np.isfinite(duration_s):
            raise ValueError("start_s and duration_s must be finite numbers")
        if start_s < 0 or duration_s <= 0:
            raise ValueError("start_s must be >= 0 and duration_s must be > 0")
        if start_s + duration_s > self.duration_s + 1e-6:
            raise ValueError(
                f"Segment [{start_s:.1f}, {start_s + duration_s:.1f}] s exceeds recording length "
                f"{self.duration_s:.1f} s")
        i0, n = int(round(start_s * self.fs_bvp)), int(round(duration_s * self.fs_bvp))
        j0, m = int(round(start_s * self.fs_acc)), int(round(duration_s * self.fs_acc))
        act = None
        if self.activity is not None:
            a0, an = int(round(start_s * self.fs_activity)), int(round(duration_s * self.fs_activity))
            act = self.activity[a0:a0 + an]
        return SignalSegment(start_s, duration_s, self.bvp[i0:i0 + n], self.fs_bvp,
                             self.acc[j0:j0 + m], self.fs_acc, act, self.fs_activity)


@dataclass
class ReferenceData:
    """Reference HR. Used for evaluation only - never by the estimation pipeline."""
    hr_labels: np.ndarray            # bpm, one per label window
    label_window_s: float = 8.0
    label_step_s: float = 2.0
    rpeaks: np.ndarray | None = None  # sample indices at fs_ecg
    fs_ecg: float = 700.0

    @property
    def label_centers_s(self) -> np.ndarray:
        return np.arange(len(self.hr_labels)) * self.label_step_s + self.label_window_s / 2.0


@dataclass
class Recording:
    subject: str
    source: str                       # "ppg-dalia" or "synthetic"
    signals: WristSignals
    reference: ReferenceData
    warnings: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def is_synthetic(self) -> bool:
        return self.source != "ppg-dalia"

    @property
    def source_tag(self) -> str:
        return "SYNTHETIC (not PPG-DaLiA)" if self.is_synthetic else "PPG-DaLiA"


# ----------------------------------------------------------------------------- helpers
def validate_subject(subject: str, config: dict | None = None) -> str:
    if not isinstance(subject, str) or not _SUBJECT_RE.match(subject):
        raise ValueError(f"Invalid subject id {subject!r}; expected like 'S1'")
    allowed = (config or {}).get("data", {}).get("allowed_subjects")
    if allowed and subject not in allowed:
        raise ValueError(f"Subject {subject} is outside the configured scope {allowed}")
    return subject


def find_subject_file(subject: str, data_dir: str | Path) -> Path:
    d = resolve_path(data_dir)
    for cand in (d / subject / f"{subject}.pkl", d / "PPG_FieldStudy" / subject / f"{subject}.pkl",
                 d / f"{subject}.pkl"):
        if cand.is_file():
            return cand
    raise DatasetNotFoundError(f"{subject}.pkl not found under {d}. {DATASET_HELP}")


def dataset_available(subject: str = "S1", config: dict | None = None) -> bool:
    cfg = config or load_config()
    try:
        find_subject_file(subject, cfg["data"]["dir"])
        return True
    except FileNotFoundError:
        return False


def clean_missing(x: np.ndarray, fs: float, name: str, max_gap_s: float, max_frac: float
                  ) -> tuple[np.ndarray, dict]:
    """Interpolate short non-finite gaps; leave long gaps as NaN (windows touching them are rejected
    downstream). Raises DataError if too much of the signal is missing."""
    x = np.array(x, dtype=float, copy=True)
    bad = ~np.isfinite(x) if x.ndim == 1 else ~np.isfinite(x).all(axis=1)
    frac = float(bad.mean()) if bad.size else 1.0
    report = {"signal": name, "nan_fraction": frac, "interpolated_samples": 0, "remaining_nan_samples": 0}
    if frac > max_frac:
        raise DataError(f"{name}: {frac:.1%} of samples are missing/non-finite (limit {max_frac:.0%})")
    if bad.any():
        idx = np.flatnonzero(bad)
        runs = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
        max_len = int(round(max_gap_s * fs))
        t = np.arange(len(x))
        for run in runs:
            ok_edges = run[0] > 0 and run[-1] < len(x) - 1
            if len(run) <= max_len and ok_edges:
                good = ~bad
                if x.ndim == 1:
                    x[run] = np.interp(run, t[good], x[good])
                else:
                    for c in range(x.shape[1]):
                        x[run, c] = np.interp(run, t[good], x[good, c])
                report["interpolated_samples"] += len(run)
            else:
                x[run] = np.nan
                report["remaining_nan_samples"] += len(run)
    return x, report


def _normalize_acc_units(acc: np.ndarray, warnings: list[str]) -> np.ndarray:
    """Return acceleration in g. Heuristic: gravity (~1 g) means median |a| ~ 1; ~64 means raw E4 counts."""
    med = float(np.nanmedian(np.linalg.norm(acc, axis=1)))
    if med > 10:
        warnings.append(f"ACC median magnitude {med:.1f} suggests raw E4 counts; divided by 64 to get g.")
        return acc / 64.0
    return acc


def build_recording(subject: str, source: str, bvp, acc, activity, hr_labels, rpeaks, config: dict,
                    meta: dict | None = None) -> Recording:
    """Validate, clean and synchronise raw arrays into a Recording."""
    d = config["data"]
    warnings: list[str] = []
    bvp = np.asarray(bvp, dtype=float).reshape(-1)
    acc = np.asarray(acc, dtype=float)
    if acc.ndim != 2 or acc.shape[1] != 3:
        raise DataError(f"ACC must have shape (N, 3); got {acc.shape}")
    if bvp.size == 0 or acc.shape[0] == 0:
        raise DataError("Empty BVP or ACC signal")
    fs_b, fs_a = float(d["fs_ppg"]), float(d["fs_acc"])
    if fs_b <= 0 or fs_a <= 0:
        raise DataError("Sampling frequencies must be positive")

    bvp, rep_b = clean_missing(bvp, fs_b, "BVP", d["max_interp_gap_s"], d["max_nan_fraction"])
    acc, rep_a = clean_missing(acc, fs_a, "ACC", d["max_interp_gap_s"], d["max_nan_fraction"])
    for rep in (rep_b, rep_a):
        if rep["nan_fraction"] > 0:
            warnings.append(f"{rep['signal']}: {rep['nan_fraction']:.2%} missing, "
                            f"{rep['interpolated_samples']} interpolated, {rep['remaining_nan_samples']} left as NaN")
    acc = _normalize_acc_units(acc, warnings)

    # Synchronisation: both wrist streams start at t=0 in PPG-DaLiA; trim to the common duration.
    dur_b, dur_a = len(bvp) / fs_b, len(acc) / fs_a
    mismatch = abs(dur_b - dur_a)
    if mismatch > d["max_stream_duration_mismatch_s"]:
        raise DataError(f"BVP ({dur_b:.1f}s) and ACC ({dur_a:.1f}s) durations differ by {mismatch:.1f}s")
    common = min(dur_b, dur_a)
    if mismatch > 1.0 / min(fs_b, fs_a):
        warnings.append(f"Trimmed streams to common duration {common:.1f}s (mismatch {mismatch:.2f}s)")
    bvp, acc = bvp[: int(common * fs_b)], acc[: int(common * fs_a)]

    act = None
    if activity is not None:
        act = np.asarray(activity).reshape(-1).astype(int)
        act = act[: int(common * d["fs_activity"])]

    hr = np.asarray(hr_labels, dtype=float).reshape(-1)
    bad_hr = ~np.isfinite(hr) | (hr < 20) | (hr > 240)
    if bad_hr.any():
        warnings.append(f"{int(bad_hr.sum())} reference HR labels outside 20-240 bpm set to NaN")
        hr = np.where(bad_hr, np.nan, hr)
    expected = int((common - d["label_window_s"]) // d["label_step_s"]) + 1
    if abs(len(hr) - expected) > 2:
        warnings.append(f"Reference label count {len(hr)} differs from expected ~{expected} for {common:.0f}s")

    rp = None
    if rpeaks is not None:
        rp = np.asarray(rpeaks).reshape(-1).astype(np.int64)
        rp = np.sort(rp[rp >= 0])
    ref = ReferenceData(hr, d["label_window_s"], d["label_step_s"], rp, d["fs_ecg"])
    sig = WristSignals(bvp, fs_b, acc, fs_a, act, d["fs_activity"])
    for w in warnings:
        log.warning("[%s] %s", subject, w)
    return Recording(subject, source, sig, ref, warnings, meta or {})


def load_subject(subject: str = "S1", data_dir: str | Path | None = None, config: dict | None = None
                 ) -> Recording:
    """Load one PPG-DaLiA participant from ``<data_dir>/<subject>/<subject>.pkl``."""
    cfg = config or load_config()
    validate_subject(subject, cfg)
    path = find_subject_file(subject, data_dir or cfg["data"]["dir"])
    log.info("Loading %s", path)
    try:
        with open(path, "rb") as f:
            raw = pickle.load(f, encoding="latin1")
        wrist = raw["signal"]["wrist"]
        bvp, acc = wrist["BVP"], wrist["ACC"]
        labels, activity = raw["label"], raw.get("activity")
        rpeaks = raw.get("rpeaks")
    except (KeyError, TypeError, pickle.UnpicklingError, EOFError) as e:
        raise DataError(f"{path} does not look like a PPG-DaLiA pickle: {e!r}") from e
    return build_recording(subject, "ppg-dalia", bvp, acc, activity, labels, rpeaks, cfg,
                           meta={"path": path.name})


def describe_recording(rec: Recording) -> dict[str, Any]:
    """Dataset-investigation summary (durations, rates, activity coverage, reference HR stats)."""
    s, r = rec.signals, rec.reference
    out: dict[str, Any] = {
        "subject": rec.subject, "source": rec.source_tag, "duration_s": round(s.duration_s, 1),
        "fs_bvp_hz": s.fs_bvp, "fs_acc_hz": s.fs_acc, "n_bvp": len(s.bvp), "n_acc": len(s.acc),
        "n_reference_labels": len(r.hr_labels),
        "reference_hr_bpm": {"min": float(np.nanmin(r.hr_labels)), "median": float(np.nanmedian(r.hr_labels)),
                             "max": float(np.nanmax(r.hr_labels))},
        "n_rpeaks": None if r.rpeaks is None else int(len(r.rpeaks)),
        "warnings": list(rec.warnings),
    }
    if s.activity is not None:
        codes, counts = np.unique(s.activity, return_counts=True)
        out["activity_seconds"] = {ACTIVITY_NAMES.get(int(c), str(c)): round(float(n) / s.fs_activity, 1)
                                   for c, n in zip(codes, counts)}
    return out
