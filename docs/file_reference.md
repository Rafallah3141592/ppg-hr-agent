# File-by-file reference

Every file in the repository, what it is for, and its main contents. Pipeline narrative: `docs/system_guide.md`. Design overview: `docs/architecture.md`.

## Root

| File | Purpose |
|---|---|
| `README.md` | Entry point: status, quick start, commands, agent usage, layout, resources. |
| `pyproject.toml` | Package metadata (`passos-wearable`), dependencies, optional extras `llm` (anthropic SDK) and `dev` (pytest, notebook tools), src-layout discovery, pytest config (`pythonpath = src`). |
| `requirements.txt` | Core runtime dependencies: numpy, scipy, pandas, matplotlib, pyyaml. |
| `requirements-dev.txt` | Core + pytest, nbformat, nbclient, ipykernel (tests and notebook execution). |
| `.gitignore` | Excludes caches, virtualenvs, `.env`, and the dataset (`data/raw/`, `*.pkl`, `*.h5`, zips) so it can never be committed. |
| `.env.example` | Template for `ANTHROPIC_API_KEY` and optional path overrides. Copy to `.env`; never commit the real one. |

## `configs/`

| File | Purpose |
|---|---|
| `default.yaml` | Every tunable value (paths, sampling rates, filters, thresholds, agent settings), commented; heuristic values are marked. See the table in `system_guide.md` §12. |

## `data/`

| File | Purpose |
|---|---|
| `README.md` | How to obtain PPG-DaLiA, the expected folder layout, which pickle keys are read, and the security note on pickles. No data is stored here. |

## `src/passos_wearable/` (the package)

| File | Purpose and main contents |
|---|---|
| `__init__.py` | Package marker and `__version__`. (All sub-package `__init__.py` files are intentionally empty markers.) |
| `config.py` | `load_config(path, overrides)` reads YAML (precedence: overrides > `PASSOS_DATA_DIR` env > file; file path from argument, `PASSOS_CONFIG`, or default). `resolve_path` makes relative paths relative to the project root, so commands work from any directory. `load_dotenv` is a tiny `.env` reader. |
| `data/loader.py` | Dataclasses `SignalSegment`, `WristSignals` (with `.segment()`), `ReferenceData`, `Recording`; `ACTIVITY_NAMES` (0 transient … 8 working); exceptions `DataError`, `DatasetNotFoundError`; `validate_subject`, `find_subject_file`, `dataset_available`, `clean_missing`, `build_recording` (validation + cleaning + sync + unit check), `load_subject` (real pickle), `describe_recording` (dataset investigation). |
| `data/synthetic.py` | `make_synthetic_recording`: schedule of activities with smooth HR, pulse waveform with harmonics, movement artefact correlated with ACC, noise, matching reference labels and R-peaks. **Test/smoke use only**; recordings are tagged `source="synthetic"`. |
| `preprocessing/filters.py` | `bandpass`, `preprocess_ppg`, `acc_magnitude`, `dynamic_acc_magnitude`, `motion_level`, `motion_label`. |
| `signal_processing/peak_detection.py` | `detect_ppg_peaks`, parabolic refinement, `peaks_to_ibi`, `clean_ibi`, `hr_from_ibi`. |
| `signal_processing/frequency_analysis.py` | `power_spectrum`, `spectral_peaks`, `spectral_hr`, `motion_frequencies`, `motion_suppressed_hr`. |
| `signal_processing/signal_quality.py` | `SignalQuality` dataclass, `spectral_concentration`, `template_correlation`, `compute_sqi`. |
| `estimation/heart_rate.py` | `METHODS`, `HREstimate` (HR, confidence, candidates, SQI, motion, warnings, `to_dict`), `estimate_hr` (one window), `estimate_hr_series` (sliding windows), `apply_temporal_continuity`. Takes arrays only - no reference access. |
| `evaluation/metrics.py` | Reference extraction (`reference_hr_from_labels`, `reference_hr_from_rpeaks`, `reference_for_windows`), `error_metrics`, `group_metrics`, `bland_altman`. |
| `evaluation/experiments.py` | `run_estimation`, `attach_reference`, `summarize`, `summarize_by`, `sqi_gating`, `worst_windows`; method lists `BASE_METHODS`, `ALL_METHODS`. |
| `visualization/plots.py` | `plot_overview`, `plot_segment`, `plot_hr_comparison`, `plot_scatter_bland`, `plot_error_by_activity`, `plot_activity_comparison`, `plot_sqi_vs_error`, `plot_motion_vs_error`. Titles carry the data-source tag. Uses the non-interactive Agg backend. |
| `tools/analysis_tool.py` | `analyze_hr_segment` (validated, two-stage, structured result) and `TOOL_SCHEMA` for LLM function calling. |
| `agent/prompts.py` | `SYSTEM_PROMPT`, `MEDICAL_PATTERNS`, `MEDICAL_REFUSAL`, `DISCLAIMER`, `FORBIDDEN_CLAIMS`, `explain_result`, `explain_followup`. |
| `agent/agent.py` | `HRAgent` (backend selection, `ask`, `run_tool`), `parse_request`, rule backend, LLM tool-use loop, `_guard` output check, `AgentReply`. |

## `app/`

| File | Purpose |
|---|---|
| `cli.py` | Terminal interface (`info`, `analyze`, `ask`, `chat`, `explore`, `evaluate`); adds `src/` to `sys.path` so it runs without installation; maps errors to exit codes 2/3. |

## `notebooks/`

| File | Purpose |
|---|---|
| `exploration_and_experiments.ipynb` | 17-section executed walkthrough: intro, setup, dataset access, investigation, S1 exploration, visualisation, preprocessing, HR estimation, alternative method, quality, motion, reference evaluation, experimental comparison, error analysis, tool demo, agent demo, conclusions. Uses real S1 when available, otherwise a tagged synthetic fallback; saved with outputs from the shipped (synthetic) run. |

## `tests/` (73 tests)

| File | Verifies |
|---|---|
| `conftest.py` | Shared fixtures: config and a 300 s synthetic recording. |
| `test_data_loader.py` | Pickle layout parsing (fake file), ACC unit conversion, missing dataset error, subject validation, segment bounds, NaN handling, stream sync, corrupt files, segment holds no reference. |
| `test_estimation.py` | Known HR recovered across rates/methods, ACC methods, motion suppression, noise/flat/invalid input, IBI cleaning, series windows, continuity, **no reference leakage** (source scan + signature), **estimate independent of reference**. |
| `test_evaluation.py` | Metric values, NaN/coverage handling, groups, Bland-Altman, reference extraction, experiment tables, stage separation. |
| `test_analysis_tool.py` | Result structure/JSON, optional separate reference, 9 validation errors, missing dataset, all methods, schema matches function, low-confidence regression. |
| `test_agent.py` | Request parsing, rule flow and follow-ups, medical refusal (no tool call), error relay, input limits, mocked LLM tool loop, guardrail notes, auto-backend fallback. |
| `test_visualization.py` | All 8 figures are written. |
| `test_config_and_cli.py` | Config/env overrides, CLI synthetic run, CLI missing-dataset exit code. |

## `reports/`

| File | Purpose |
|---|---|
| `technical_report.md` | Method and implementation report (no invented real-data results). |
| `experiments.md` | Execution log: blocker, checklist, fixed defects, labelled synthetic experiment tables, test outcomes, notebook and agent status. |
| `limitations.md` | Requirement-by-requirement status (verified / unverified / incomplete), failure cases, limitations, future work. |

## `docs/`

| File | Purpose |
|---|---|
| `system_guide.md` | Step-by-step explanation of the whole system, configuration table, troubleshooting. |
| `file_reference.md` | This file. |
| `architecture.md` | Diagram, design decisions, module map. |

## `outputs/`

| Path | Purpose |
|---|---|
| `README.md` | Explains the tagging of artifact folders. |
| `synthetic_smoke_test/` | CLI `evaluate`/`explore` artifacts on **synthetic** data: `window_estimates.csv`, `metrics_overall.csv`, `metrics_overall_vs_ecg.csv`, `metrics_by_activity.csv`, `metrics_by_motion.csv`, `sqi_gating.csv`, `worst_windows.csv`, 8 PNG plots, `SOURCE.txt`. |
| `notebook_run/` | Artifacts from the shipped notebook execution (synthetic), with `SOURCE.txt`. |
