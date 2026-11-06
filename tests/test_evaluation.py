import numpy as np
import pytest

from passos_wearable.data.loader import ReferenceData
from passos_wearable.evaluation import experiments as X
from passos_wearable.evaluation.metrics import (bland_altman, error_metrics, group_metrics,
                                                reference_hr_from_labels, reference_hr_from_rpeaks)


def test_known_metrics():
    m = error_metrics([72, 80, 90, 100], [70, 82, 90, 96], tol_bpm=3)
    assert m["mae"] == pytest.approx(2.0) and m["bias"] == pytest.approx(1.0)
    assert m["rmse"] == pytest.approx(np.sqrt((4 + 4 + 0 + 16) / 4))
    assert m["medae"] == pytest.approx(2.0) and m["within_tol"] == pytest.approx(0.75)
    assert m["n_valid"] == 4 and m["coverage"] == 1.0


def test_nan_handling_and_coverage():
    m = error_metrics([70, np.nan, 90, 100], [70, 80, 92, np.nan])
    assert m["n_valid"] == 2 and m["n_windows"] == 3 and m["coverage"] == pytest.approx(2 / 3)
    assert np.isnan(error_metrics([np.nan], [70])["mae"])


def test_shape_mismatch():
    with pytest.raises(ValueError):
        error_metrics([1, 2], [1])


def test_group_metrics_and_bland_altman():
    g = group_metrics([70, 80, 100, 110], [70, 80, 90, 100], ["a", "a", "b", "b"])
    assert g["a"]["mae"] == 0 and g["b"]["mae"] == 10 and g["b"]["bias"] == 10
    ba = bland_altman([72, 78, 90], [70, 80, 90])
    assert ba["loa_low"] < ba["mean_diff"] < ba["loa_high"]


def test_reference_from_labels_window_logic():
    ref = ReferenceData(np.array([60., 62, 64, 66, 68]), 8.0, 2.0)   # label i covers [2i, 2i+8]
    r = reference_hr_from_labels(ref, 0, 12)                         # windows 0..2 fit inside [0,12]
    assert r["hr_bpm"] == pytest.approx(62.0) and r["n_labels"] == 3 and not r["approximate"]
    r = reference_hr_from_labels(ref, 3, 8)                          # none fit -> nearest, flagged
    assert r["approximate"] and r["n_labels"] == 1


def test_reference_from_rpeaks():
    ref = ReferenceData(np.array([75.]), rpeaks=(np.arange(0, 700 * 20, 560)), fs_ecg=700.0)  # 0.8 s -> 75 bpm
    assert reference_hr_from_rpeaks(ref, 2, 10)["hr_bpm"] == pytest.approx(75.0, abs=0.1)
    assert np.isnan(reference_hr_from_rpeaks(ref, 0, 1)["hr_bpm"])
    assert np.isnan(reference_hr_from_rpeaks(ReferenceData(np.array([70.])), 0, 10)["hr_bpm"])


def test_experiment_table_pipeline(rec, cfg):
    df = X.attach_reference(X.run_estimation(rec, cfg, 0, 120), rec, cfg)
    assert {"hr_peaks", "hr_fusion_continuity", "ref_label_hr", "ref_ecg_hr", "activity"} <= set(df.columns)
    tab = X.summarize(df)
    assert list(tab.index) == X.ALL_METHODS and (tab["mae"] < 40).all()
    assert len(X.summarize_by(df, "activity")) >= len(X.ALL_METHODS)
    assert len(X.sqi_gating(df)) == 3 and len(X.worst_windows(df, n=5)) == 5


def test_estimation_stage_unaffected_by_attach(rec, cfg):
    d1 = X.run_estimation(rec, cfg, 0, 60)
    d2 = X.attach_reference(d1, rec, cfg)
    assert d1["hr_fusion"].equals(d2["hr_fusion"]) and "ref_label_hr" not in d1
