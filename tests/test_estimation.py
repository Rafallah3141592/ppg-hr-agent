from pathlib import Path

import numpy as np
import pytest

from passos_wearable.estimation.heart_rate import (METHODS, apply_temporal_continuity, estimate_hr,
                                                   estimate_hr_series)
from passos_wearable.signal_processing.peak_detection import clean_ibi, hr_from_ibi, peaks_to_ibi

FS = 64


def clean_ppg(hr, dur=10, noise=0.05, seed=0):
    t = np.arange(int(dur * FS)) / FS
    ph = 2 * np.pi * hr / 60 * t
    return np.sin(ph) + 0.4 * np.sin(2 * ph - 1) + np.random.default_rng(seed).normal(0, noise, t.size)


@pytest.mark.parametrize("hr", [55, 75, 100, 140, 170])
@pytest.mark.parametrize("method", ["peaks", "spectral", "fusion"])
def test_clean_synthetic_pulse_recovered(hr, method, cfg):
    est = estimate_hr(clean_ppg(hr), FS, method=method, config=cfg)
    assert est.valid and abs(est.hr_bpm - hr) < 3, (hr, method, est.hr_bpm)
    assert est.confidence in ("high", "medium")


def test_all_methods_run_with_acc(cfg):
    acc = np.random.default_rng(0).normal(0, 0.01, (320, 3)) + [0, 0, 1]
    for m in METHODS:
        est = estimate_hr(clean_ppg(80), FS, acc, 32, m, cfg)
        assert abs(est.hr_bpm - 80) < 3 and est.motion_label == "low"


def test_motion_suppression_removes_motion_peak(cfg):
    t = np.arange(10 * FS) / FS
    ppg = 0.6 * np.sin(2 * np.pi * 1.2 * t) + 1.0 * np.sin(2 * np.pi * 2.6 * t)  # HR 72 bpm + motion 2.6 Hz
    ta = np.arange(10 * 32) / 32
    acc = np.c_[0.5 * np.sin(2 * np.pi * 2.6 * ta), 0.3 * np.sin(2 * np.pi * 2.6 * ta), np.ones_like(ta)]
    assert abs(estimate_hr(ppg, FS, acc, 32, "spectral", cfg).hr_bpm - 156) < 4        # locks onto motion
    est = estimate_hr(ppg, FS, acc, 32, "spectral_motion", cfg)
    assert abs(est.hr_bpm - 72) < 4 and est.motion_label in ("medium", "high") and est.motion_freqs_hz


def test_noise_only_is_flagged_low_quality(cfg):
    est = estimate_hr(np.random.default_rng(3).normal(size=10 * FS), FS, config=cfg)
    assert est.sqi.label in ("low", "medium") and est.confidence != "high"


def test_flat_and_invalid_inputs(cfg):
    flat = estimate_hr(np.ones(640), FS, config=cfg)
    assert not flat.valid and np.isnan(flat.hr_bpm) and flat.confidence == "none"
    with pytest.raises(ValueError):
        estimate_hr(np.r_[np.ones(300), np.nan, np.ones(339)], FS, config=cfg)
    with pytest.raises(ValueError):
        estimate_hr(np.ones(64), FS, config=cfg)               # too short
    with pytest.raises(ValueError):
        estimate_hr(clean_ppg(80), FS, method="bogus", config=cfg)
    with pytest.raises(ValueError):
        estimate_hr(clean_ppg(80), FS, method="spectral_motion", config=cfg)  # needs ACC
    with pytest.raises(ValueError):
        estimate_hr(np.ones((10, 2)), FS, config=cfg)


def test_implausible_rate_is_rejected(cfg):
    est = estimate_hr(clean_ppg(300, noise=0.0), FS, method="spectral", config=cfg)
    assert not est.valid or est.hr_bpm <= cfg["estimation"]["hr_range_bpm"][1]


def test_ibi_cleaning_rejects_missed_and_spurious_beats():
    ibi = np.array([800, 810, 1600, 790, 300 + 5, 805, 2500])
    clean, mask = clean_ibi(ibi, (300, 1500), 0.25)
    assert list(clean) == [800, 810, 790, 805] and abs(hr_from_ibi(clean) - 60000 / 802.5) < 1e-6


def test_peaks_to_ibi():
    assert np.allclose(peaks_to_ibi(np.array([0, 64, 128.0]), 64), [1000, 1000])


def test_series_and_continuity(cfg):
    x = np.r_[clean_ppg(70, 30), clean_ppg(72, 30, seed=1)]
    starts, ests = estimate_hr_series(x, FS, None, None, "peaks", cfg)
    assert len(starts) == (60 - 8) // 2 + 1 and np.nanmedian([e.hr_bpm for e in ests]) > 65
    out, held = apply_temporal_continuity(np.array([70, 71, 120, 72, np.nan, 73]), 12, 4)
    assert list(out) == [70, 71, 71, 72, 72, 73] and held.sum() == 2


def test_no_reference_leakage():
    """Estimation-side modules must not import or mention reference data."""
    root = Path(__file__).resolve().parents[1] / "src" / "passos_wearable"
    banned = ["hr_labels", "rpeaks", "ReferenceData", "passos_wearable.evaluation", "passos_wearable.data",
              "ref_label", "reference_hr"]
    for pkg in ("estimation", "signal_processing", "preprocessing"):
        for f in (root / pkg).glob("*.py"):
            src = f.read_text()
            for tok in banned:
                assert tok not in src, f"{f.name} mentions {tok}"
    import inspect
    assert set(inspect.signature(estimate_hr).parameters) == {"ppg", "fs_ppg", "acc", "fs_acc", "method", "config"}


def test_estimate_independent_of_reference(rec, cfg):
    """Changing the reference HR must not change the estimate."""
    import copy
    from passos_wearable.tools.analysis_tool import analyze_hr_segment
    r1 = analyze_hr_segment(start_s=100, duration_s=20, recording=rec, config=cfg)
    rec2 = copy.deepcopy(rec); rec2.reference.hr_labels[:] = 150.0; rec2.reference.rpeaks = None
    r2 = analyze_hr_segment(start_s=100, duration_s=20, recording=rec2, config=cfg)
    assert r1["estimate"] == r2["estimate"] and r1["evaluation"] != r2["evaluation"]
