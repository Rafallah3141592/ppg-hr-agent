"""System prompt, guardrails and deterministic explanation templates for the agent."""
from __future__ import annotations

import re

SYSTEM_PROMPT = """You are a careful assistant that explains wrist-PPG heart-rate analyses for the PPG-DaLiA dataset (participant S1).

Rules:
1. For ANY question about a heart-rate value you must call the tool `analyze_hr_segment`; never guess or recall HR values.
2. Use ONLY numbers that appear in the tool result. Quote them with units. Do not invent additional figures.
3. Always communicate uncertainty: state the confidence level, signal quality, movement level, and any warnings or caveats from the tool result.
4. The reference HR (under `evaluation`) was NOT used for the estimate. Describe it as a comparison, not as an input.
5. If `is_synthetic` is true, say clearly the data are synthetic test data, not PPG-DaLiA.
6. You are NOT a medical device. Never diagnose, rule out, or label any condition (e.g. arrhythmia, tachycardia, heart disease), never give treatment or health advice, and never say a heart rate is "normal"/"abnormal" for the person. If asked, decline briefly and offer to explain the signal analysis instead.
7. If the tool returns status="error", relay the problem and ask for a corrected request.
Keep answers short and structured."""

MEDICAL_PATTERNS = re.compile(
    r"diagnos|arrhythm|afib|a-fib|atrial|fibrillat|tachycard|bradycard|heart (attack|disease|failure|condition)|"
    r"cardiac|cardio\w*|medical|medicat|treat(ment)?|disease|is (it|my heart|this) (normal|healthy|dangerous|safe)|"
    r"am i (ok|okay|healthy|sick)|should i (see|worry|go)|doctor|symptom|stroke|\bhealthy\b|\bnormal\b|\babnormal\b|\bdangerous\b",
    re.I)

MEDICAL_REFUSAL = (
    "I can't interpret this as a medical or health question. This tool estimates heart rate from a wrist "
    "PPG signal with classical signal processing for a research assignment; it is not a medical device and "
    "I won't diagnose, rule out, or advise on any health condition. If you have health concerns, please "
    "talk to a qualified clinician. I can explain how the estimate was produced, how reliable the signal was, "
    "or how it compares to the dataset's reference HR.")

DISCLAIMER = "Research analysis only - not a medical device and not a diagnosis."

# phrases an LLM answer must not contain (diagnostic claims); a note is appended if detected
FORBIDDEN_CLAIMS = re.compile(
    r"\byou (have|are suffering)\b|\bdiagnos(is|ed|e)\b(?!.{0,25}\b(not|cannot|can't|won't|no)\b)|"
    r"\b(indicates|suggests|consistent with|sign of) (an? )?(arrhythmia|tachycardia|bradycardia|atrial|heart (disease|condition))",
    re.I)


def _pct(x) -> str:
    return "n/a" if x is None else f"{100 * x:.0f}%"


def explain_result(res: dict) -> str:
    """Deterministic, evidence-grounded explanation: every figure comes from ``res``."""
    if res.get("status") != "ok":
        return (f"I couldn't run the analysis: {res.get('error_message', 'unknown error')}\n"
                "Please adjust the request (subject S1, start >= 0, duration 8-300 s, within the recording).")
    e, seg, q, m = res["estimate"], res["segment"], res["signal_quality"], res["motion"]
    pl = res["plausibility"]
    L = []
    if res["is_synthetic"]:
        L.append("** SYNTHETIC DATA - generated test signal, NOT PPG-DaLiA. Figures below do not describe real performance. **")
    L.append(f"Segment {seg['start_s']:.0f}-{seg['end_s']:.0f} s of {res['request']['subject']} "
             f"({seg['n_windows']} windows of {seg['window_s']:.0f} s, step {seg['step_s']:.0f} s), data source: {res['data_source']}.")
    if e["hr_bpm"] is None:
        L.append("No valid HR estimate could be produced for this segment "
                 f"(valid windows: {_pct(pl['valid_window_fraction'])}).")
    else:
        sd = e["window_to_window_std_bpm"]
        L.append(f"Estimated HR: {e['hr_bpm']:.1f} bpm (method '{e['method']}', median over windows"
                 + (f"; window-to-window SD {sd:.1f} bpm, range {e['window_hr_min_bpm']:.1f}-{e['window_hr_max_bpm']:.1f} bpm)." if sd is not None else ")."))
    L.append(f"Confidence: {e['confidence']}.")
    alts = {k: v for k, v in res["alternatives_bpm"].items() if v is not None and k != e["method"]}
    if alts:
        L.append("Other methods on the same segment: " + ", ".join(f"{k} {v:.1f} bpm" for k, v in alts.items()) + ".")
    L.append(f"Signal quality: {q['label']} (mean SQI {q['mean_sqi']}; {_pct(q['fraction_low_quality_windows'])} of windows low quality)."
             if q["mean_sqi"] is not None else "Signal quality: unavailable.")
    L.append(f"Movement: {m['label']} (mean dynamic-acceleration SD {m['mean_level_g']} g; "
             f"{_pct(m['fraction_high_motion_windows'])} of windows high-motion).")
    ctx = res.get("context", {}).get("annotated_activity_fractions")
    if ctx:
        L.append("Annotated activity in this segment (context only, not used by the estimator): "
                 + ", ".join(f"{k} {v:.0%}" for k, v in ctx.items()) + ".")
    if pl["mean_method_disagreement_bpm"] is not None:
        L.append(f"Mean disagreement between peak-based and spectral estimates: {pl['mean_method_disagreement_bpm']:.1f} bpm.")
    if pl["warnings"]:
        L.append("Warnings: " + "; ".join(f"{k} (x{v})" for k, v in pl["warnings"].items()) + ".")
    ev = res.get("evaluation")
    if ev:
        s = (f"Reference comparison (evaluation only, not used for the estimate): dataset label HR {ev['dataset_label_hr_bpm']} bpm"
             + (" (approximate window alignment)" if ev["label_alignment_approximate"] else "")
             + f", ECG R-peak HR {ev['ecg_rpeak_hr_bpm']} bpm.")
        if "abs_error_bpm" in ev:
            s += f" Absolute error vs label: {ev['abs_error_bpm']} bpm (signed {ev['signed_error_bpm']:+} bpm)."
        L.append(s)
    L.append("Uncertainty: " + " ".join(res["caveats"]))
    L.append(DISCLAIMER)
    return "\n".join(L)


def explain_followup(res: dict, question: str) -> str:
    """Answer a follow-up using only the stored tool result."""
    if res.get("status") != "ok":
        return explain_result(res)
    ql = question.lower()
    e, q, m, pl = res["estimate"], res["signal_quality"], res["motion"], res["plausibility"]
    parts = []
    if any(k in ql for k in ("why", "confiden", "reliab", "trust", "sure")):
        parts.append(f"Confidence is '{e['confidence']}' because: signal quality is {q['label']} "
                     f"(mean SQI {q['mean_sqi']}), movement is {m['label']} ({m['mean_level_g']} g), "
                     f"{_pct(pl['valid_window_fraction'])} of windows gave a valid estimate, and the window-to-window SD is "
                     f"{e['window_to_window_std_bpm']} bpm.")
    if any(k in ql for k in ("quality", "sqi", "signal")):
        parts.append(f"SQI flags seen: {q['flag_counts'] or 'none'}. SQI combines spectral concentration, beat-template "
                     "correlation and IBI regularity; its thresholds are heuristic.")
    if any(k in ql for k in ("motion", "movement", "accel", "walk", "activity")):
        parts.append(f"Movement level {m['mean_level_g']} g ({m['label']}); wrist PPG degrades with movement because arm "
                     "motion adds artefacts at similar frequencies to the pulse.")
    if any(k in ql for k in ("reference", "error", "accura", "compare", "truth", "ecg")):
        ev = res.get("evaluation")
        parts.append("No reference comparison was requested for this analysis." if not ev else
                     f"Reference (not an input to the estimate): label HR {ev['dataset_label_hr_bpm']} bpm, ECG R-peak HR "
                     f"{ev['ecg_rpeak_hr_bpm']} bpm; absolute error vs label {ev.get('abs_error_bpm', 'n/a')} bpm. "
                     "This is a single segment, so it says little about overall accuracy.")
    if any(k in ql for k in ("method", "how", "alternative", "other")):
        parts.append(f"Method '{e['method']}'. Per-method medians on this segment: {res['alternatives_bpm']}.")
    if not parts:
        return explain_result(res)
    parts.append(DISCLAIMER)
    return "\n".join(parts)
