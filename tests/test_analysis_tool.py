import json

import pytest

from passos_wearable.tools.analysis_tool import TOOL_SCHEMA, analyze_hr_segment


def run(rec, cfg, **kw):
    kw.setdefault("start_s", 60); kw.setdefault("duration_s", 30)
    return analyze_hr_segment(recording=rec, config=cfg, **kw)


def test_ok_result_structure_and_json(rec, cfg):
    r = run(rec, cfg)
    json.dumps(r)
    assert r["status"] == "ok" and r["is_synthetic"] and "SYNTHETIC" in r["data_source"]
    for k in ("segment", "estimate", "alternatives_bpm", "signal_quality", "motion", "plausibility", "evaluation", "caveats"):
        assert k in r
    assert r["segment"]["n_windows"] == (30 - 8) // 2 + 1
    assert r["estimate"]["confidence"] in ("high", "medium", "low", "none")
    assert any("SYNTHETIC" in c for c in r["caveats"])


def test_reference_is_optional_and_separate(rec, cfg):
    r = run(rec, cfg, include_reference=False)
    assert r["evaluation"] is None
    r2 = run(rec, cfg, include_reference=True)
    assert r2["estimate"] == r["estimate"] and "abs_error_bpm" in r2["evaluation"]
    assert "not an input" in r2["evaluation"]["note"]


@pytest.mark.parametrize("kw,frag", [
    ({"start_s": -5}, "start_s"), ({"duration_s": 2}, "duration_s"), ({"duration_s": 5000}, "duration_s"),
    ({"start_s": 290, "duration_s": 30}, "recording"), ({"method": "magic"}, "Unknown method"),
    ({"subject": "S2"}, "scope"), ({"subject": "../etc"}, "Invalid subject"),
    ({"start_s": "abc"}, "numbers"), ({"start_s": float("nan")}, "finite")])
def test_validation_errors_are_structured(rec, cfg, kw, frag):
    r = run(rec, cfg, **kw)
    assert r["status"] == "error" and frag in r["error_message"]


def test_missing_dataset_is_structured_error(tmp_path, cfg):
    r = analyze_hr_segment(start_s=0, duration_s=30, data_dir=str(tmp_path), config=cfg)
    assert r["status"] == "error" and "PPG-DaLiA" in r["error_message"]


def test_all_methods(rec, cfg):
    for m in ("peaks", "spectral", "spectral_motion", "fusion"):
        assert run(rec, cfg, method=m)["status"] == "ok"


def test_schema_matches_function():
    import inspect
    params = set(inspect.signature(analyze_hr_segment).parameters)
    assert set(TOOL_SCHEMA["input_schema"]["properties"]) <= params
    assert TOOL_SCHEMA["input_schema"]["required"] == ["start_s", "duration_s"]


def test_unstable_window_estimates_cannot_be_high_or_medium_confidence(rec, cfg):
    """Regression: a segment whose window HRs scatter widely (SD > 10 bpm) must be rated low confidence."""
    r = run(rec, cfg, start_s=130, duration_s=30, method="fusion")   # synthetic walking: motion lock-in
    sd = r["estimate"]["window_to_window_std_bpm"]
    assert sd is not None and sd > 10 and r["estimate"]["confidence"] == "low"
