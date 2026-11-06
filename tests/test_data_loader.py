import pickle

import numpy as np
import pytest

from passos_wearable.data.loader import (ACTIVITY_NAMES, DataError, DatasetNotFoundError, clean_missing,
                                         describe_recording, load_subject, validate_subject)


def _write_fake_pkl(tmp_path, n_s=120, acc_scale=1.0):
    """Fake file in the exact PPG-DaLiA pickle layout (synthetic content)."""
    rng = np.random.default_rng(0)
    d = {"signal": {"wrist": {"BVP": rng.normal(size=(64 * n_s, 1)), "ACC": rng.normal(size=(32 * n_s, 3)) * 0.05 + [0, 0, acc_scale],
                              "EDA": np.zeros((4 * n_s, 1)), "TEMP": np.zeros((4 * n_s, 1))}, "chest": {}},
         "label": np.linspace(60, 90, (n_s - 8) // 2 + 1), "activity": np.ones((4 * n_s, 1)),
         "rpeaks": np.arange(0, 700 * n_s, 700), "subject": "S1"}
    (tmp_path / "S1").mkdir()
    with open(tmp_path / "S1" / "S1.pkl", "wb") as f:
        pickle.dump(d, f)
    return tmp_path


def test_load_dalia_layout(tmp_path, cfg):
    rec = load_subject("S1", _write_fake_pkl(tmp_path), cfg)
    assert rec.source == "ppg-dalia" and rec.signals.fs_bvp == 64 and rec.signals.fs_acc == 32
    assert rec.signals.bvp.ndim == 1 and rec.signals.acc.shape[1] == 3
    assert abs(rec.signals.duration_s - 120) < 0.1
    assert len(rec.reference.hr_labels) == 57
    assert describe_recording(rec)["activity_seconds"]["sitting"] > 100


def test_acc_raw_counts_are_converted(tmp_path, cfg):
    rec = load_subject("S1", _write_fake_pkl(tmp_path, acc_scale=64.0), cfg)
    assert 0.5 < np.median(np.linalg.norm(rec.signals.acc, axis=1)) < 1.5
    assert any("raw E4 counts" in w for w in rec.warnings)


def test_missing_dataset_gives_helpful_error(tmp_path, cfg):
    with pytest.raises(DatasetNotFoundError, match="PPG-DaLiA"):
        load_subject("S1", tmp_path, cfg)


def test_subject_validation(cfg):
    assert validate_subject("S1", cfg) == "S1"
    for bad in ("../S1", "S2", "s1", "", None):
        with pytest.raises(ValueError):
            validate_subject(bad, cfg)


def test_segment_selection_and_bounds(rec):
    seg = rec.signals.segment(10, 20)
    assert len(seg.bvp) == 20 * 64 and len(seg.acc) == 20 * 32 and len(seg.activity) == 20 * 4
    for s, d in [(-1, 5), (0, 0), (290, 20), (np.nan, 5)]:
        with pytest.raises(ValueError):
            rec.signals.segment(s, d)


def test_segment_object_has_no_reference(rec):
    seg = rec.signals.segment(0, 10)
    assert not hasattr(seg, "hr_labels") and not hasattr(seg, "reference")


def test_missing_values_short_gap_interpolated_long_gap_kept():
    x = np.sin(np.linspace(0, 20, 640)); x[100:110] = np.nan; x[300:500] = np.nan
    y, rep = clean_missing(x, 64, "BVP", max_gap_s=2.0, max_frac=0.5)
    assert np.isfinite(y[100:110]).all() and np.isnan(y[300:500]).all()
    assert rep["interpolated_samples"] == 10 and rep["remaining_nan_samples"] == 200


def test_too_many_missing_raises():
    x = np.full(100, np.nan)
    with pytest.raises(DataError):
        clean_missing(x, 64, "BVP", 2.0, 0.2)


def test_bad_acc_shape_raises(cfg):
    from passos_wearable.data.loader import build_recording
    with pytest.raises(DataError):
        build_recording("S1", "synthetic", np.zeros(640), np.zeros((320, 2)), None, [70], None, cfg)


def test_stream_synchronisation_trim(cfg):
    from passos_wearable.data.loader import build_recording
    rec = build_recording("S1", "synthetic", np.random.randn(64 * 62), np.random.randn(32 * 60, 3) + [0, 0, 1],
                          None, np.full(27, 70.0), None, cfg)
    assert abs(len(rec.signals.bvp) / 64 - len(rec.signals.acc) / 32) < 0.05
    assert any("Trimmed" in w for w in rec.warnings)


def test_corrupt_pickle_raises_dataerror(tmp_path, cfg):
    (tmp_path / "S1").mkdir()
    (tmp_path / "S1" / "S1.pkl").write_bytes(pickle.dumps({"nothing": 1}))
    with pytest.raises(DataError):
        load_subject("S1", tmp_path, cfg)


def test_activity_names_complete():
    assert ACTIVITY_NAMES[7] == "walking" and len(ACTIVITY_NAMES) == 9
