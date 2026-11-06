# Architecture

```
            ┌────────────────────────── signals only ──────────────────────────┐
 S1.pkl ──► data.loader ──► WristSignals ──► SignalSegment ──► estimation.heart_rate ──► HREstimate
 (BVP,ACC)   validate,       (BVP, ACC,        (no reference     preprocessing.filters      (HR, candidates,
             clean NaNs,      activity)         HR inside)       signal_processing.*         SQI, motion,
             sync, units                                                                      warnings)
            └───────────────────────────────────────────────────────────────────┘
                                                              │ estimate is final
 S1.pkl ──► ReferenceData (labels, R-peaks) ──► evaluation.metrics ◄─────────────┘
                                                      │
                                    evaluation.experiments ──► tables, visualization.plots
                                                      │
                tools.analysis_tool.analyze_hr_segment  (structured JSON; separate `evaluation` key)
                                                      │
                          agent.agent.HRAgent ── backend "rule" (offline, deterministic)
                                              └─ backend "llm"  (Anthropic tool use, optional)
                                                      │
                                              app/cli.py  (analyze | ask | chat | explore | evaluate | info)
```

## Key design decisions

* **Reference isolation.** `data.loader` returns two containers: `WristSignals` (what the estimator may see) and `ReferenceData`.
  `estimate_hr` takes plain arrays `(ppg, fs, acc, fs_acc, method, config)`. `tests/test_estimation.py::test_no_reference_leakage`
  fails if any estimation-side module mentions reference identifiers or imports `evaluation`/`data`, and
  `test_estimate_independent_of_reference` proves the estimate is unchanged when the reference is altered.
  In `evaluation.experiments`, estimation (`run_estimation`) and reference attachment (`attach_reference`) are separate steps.
* **Tool results are data, errors are data.** `analyze_hr_segment` never raises to the caller; invalid input returns `status="error"`
  with a message an LLM (or the CLI) can relay.
* **Two agent backends, one tool.** The rule backend and the LLM backend call the same function with the same validation. The LLM
  answer is post-checked (diagnostic wording, bpm figures not present in the tool result).
* **Synthetic data is fenced off.** `Recording.source` is `"synthetic"` or `"ppg-dalia"`; plots, tool results and explanations
  carry the tag. Synthetic data exists for tests and `--synthetic` smoke runs only.
* **Configuration.** All thresholds live in `configs/default.yaml`; relative paths resolve against the project root.

## Module map

| Module | Responsibility |
|---|---|
| `config` | YAML + env loading (`PASSOS_CONFIG`, `PASSOS_DATA_DIR`), `.env` reader |
| `data.loader` | load pickle, validation, NaN interpolation, stream sync/trim, ACC unit check, segment selection, dataset summary |
| `data.synthetic` | generated PPG/ACC with known HR (tests/smoke only) |
| `preprocessing.filters` | Butterworth band-pass, ACC magnitude/dynamic component, motion level & label |
| `signal_processing.peak_detection` | peak finding, parabolic refinement, IBI, IBI plausibility filter |
| `signal_processing.frequency_analysis` | zero-padded periodogram, spectral HR, ACC motion frequencies, motion-suppressed spectral HR |
| `signal_processing.signal_quality` | SQI = 0.4 spectral concentration + 0.3 beat-template correlation + 0.3 IBI regularity |
| `estimation.heart_rate` | `estimate_hr`, `estimate_hr_series`, fusion, confidence, temporal continuity |
| `evaluation.metrics` | reference extraction (labels, R-peaks), MAE/RMSE/bias/MedAE/r/within-tolerance, Bland-Altman |
| `evaluation.experiments` | sliding-window experiments, per-activity/motion tables, SQI gating, worst windows |
| `visualization.plots` | all figures |
| `tools.analysis_tool` | `analyze_hr_segment` + `TOOL_SCHEMA` |
| `agent.agent`, `agent.prompts` | backends, request parsing, guardrails, explanation templates |
| `app/cli.py` | terminal interface |
