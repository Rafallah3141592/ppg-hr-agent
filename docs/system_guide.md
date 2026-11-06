# System guide: how everything works, step by step

> Data-status reminder: this project was built without access to the real PPG-DaLiA files. Everything below describes the implemented method. Any numbers quoted in this guide's examples are illustrative; real results appear only after you run the commands on real S1 data.

## 1. What the system does

Given a recording of participant **S1** from PPG-DaLiA, the system:

1. loads the wrist **PPG** (BVP, 64 Hz) and **accelerometer** (ACC, 32 Hz), plus - kept separate - the **reference heart rate** (HR) derived from ECG;
2. estimates HR in sliding 8-second windows using the PPG (and ACC for movement handling) only;
3. rates how trustworthy each estimate is (signal quality, movement, agreement between methods);
4. compares estimates to the reference **afterwards** to measure error;
5. exposes one function (`analyze_hr_segment`) that an AI agent can call to analyse any time segment, and explains the structured result in plain language with uncertainty and without medical claims.

Why PPG on the wrist is hard: the pulse signal is tiny compared with the signal caused by arm movement. During walking or cycling, movement creates strong periodic components in the PPG at similar frequencies to the heart rate, so simple methods can lock onto the movement instead of the heartbeat. Most of the design (accelerometer analysis, quality scores, confidence levels) exists because of this.

## 2. The pipeline at a glance

```
S1.pkl
  ├─ signal.wrist.BVP, ACC, activity ──► [Step 1 load/validate] ──► WristSignals
  │                                                                  │ segment(start, duration)
  │                                                                  ▼
  │                                     [Step 2 filter] → [Step 3 peaks/IBI] → [Step 4 spectrum]
  │                                     [Step 5 motion]  → [Step 6 fusion]   → [Step 7 quality/confidence]
  │                                                                  ▼
  │                                                              HREstimate  (final, immutable w.r.t. reference)
  └─ label, rpeaks ──► ReferenceData ──► [Step 8 evaluation: errors, tables, plots]
                                                                  ▼
                                   [Step 9 analysis tool] ──► [Step 10 agent] ──► [Step 11 CLI / notebook]
```

The arrow from `ReferenceData` only meets the estimate **after** the estimate is complete. That is the central integrity rule of the project.

---

## Step 1 - Load, validate, clean, synchronise (`data/loader.py`)

**Input:** `S1/S1.pkl` (a Python pickle). **Output:** a `Recording` object holding `signals` (`WristSignals`) and `reference` (`ReferenceData`), plus a list of warnings.

What happens, in order:

1. **Subject check** (`validate_subject`): the id must match `S<digits>` (this blocks path tricks like `../`) and be in `data.allowed_subjects` (default `[S1]`, the assignment scope).
2. **Find the file** (`find_subject_file`): looks for `<data.dir>/S1/S1.pkl`, then `<data.dir>/PPG_FieldStudy/S1/S1.pkl`, then `<data.dir>/S1.pkl`. If none exists a `DatasetNotFoundError` is raised with download instructions.
3. **Unpickle** with `encoding="latin1"` (the dataset was written by Python 2). A missing key or malformed file becomes a `DataError` naming the problem.
4. **Missing-value handling** (`clean_missing`): non-finite samples are found. Runs of up to `max_interp_gap_s` (2 s) with valid samples on both sides are linearly interpolated. Longer runs are left as `NaN` on purpose: inventing 10 seconds of signal would produce fake estimates, so windows touching such gaps are rejected later. If more than `max_nan_fraction` (20 %) of a stream is missing, loading fails.
5. **Unit check** (`_normalize_acc_units`): gravity is about 1 g, so the median ACC magnitude should be about 1. If it is above 10 the data are assumed to be raw Empatica counts (64 counts = 1 g) and are divided by 64. This keeps the movement thresholds (in g) meaningful. It is a heuristic and is reported as a warning when applied.
6. **Synchronisation:** BVP and ACC come from the same device and start at t = 0, but their sample counts may not match perfectly. Durations are computed from sample counts and rates. A difference above `max_stream_duration_mismatch_s` (5 s) is an error; a smaller one is handled by trimming both streams to the common duration (reported as a warning). The activity stream is trimmed too.
7. **Reference sanity:** reference HR labels outside 20-240 bpm become `NaN`; the label count is compared with the expected count for the duration and a warning is raised if they differ by more than 2.

**Segment selection:** `WristSignals.segment(start_s, duration_s)` converts seconds to sample indices separately for each stream (`start × fs`), returning a `SignalSegment`. Negative starts, zero/negative durations, NaN values, or segments beyond the recording length raise `ValueError`. A `SignalSegment` has **no reference field** by design.

**`describe_recording`** summarises durations, rates, reference HR range and seconds per activity (used by `cli.py info` and notebook section 4).

## Step 2 - Preprocessing (`preprocessing/filters.py`)

`preprocess_ppg(ppg, fs, config)`:
- **Band-pass 0.5-4 Hz**, 4th-order Butterworth applied forward and backward (`sosfiltfilt`, zero phase so peaks do not shift in time). 0.5 Hz = 30 bpm, 4 Hz = 240 bpm, so this keeps every plausible heart rate and removes slow baseline drift (breathing, sensor pressure changes) and high-frequency noise.
- **Standardise** to unit standard deviation so thresholds do not depend on the sensor's arbitrary amplitude.
- Rejects non-finite input and windows too short for the filter.

Movement helpers: `acc_magnitude` (√(x²+y²+z²), independent of device orientation), `dynamic_acc_magnitude` (0.3-8 Hz band-pass, which removes gravity/posture and leaves movement), `motion_level` (standard deviation of that dynamic magnitude, in g), `motion_label` (`low` < 0.05 g, `high` ≥ 0.2 g, else `medium`; heuristic thresholds in config).

## Step 3 - Peak detection and inter-beat intervals (`signal_processing/peak_detection.py`)

This is the "time-domain" method.

1. `detect_ppg_peaks`: SciPy `find_peaks` on the filtered window with a minimum distance of `fs × 60 / 200` samples (no two beats closer than the 200 bpm limit) and a minimum prominence of 0.3 × the window's standard deviation (ignores tiny ripples). Each peak is refined by **parabolic interpolation** using its two neighbours, improving timing beyond the 15.6 ms sampling grid.
2. `peaks_to_ibi`: differences between consecutive peak times, in milliseconds. This is the **inter-beat interval (IBI)**.
3. `clean_ibi` (plausibility): keep intervals between 300 and 1500 ms, then keep those within 25 % of the median. A *missed* beat shows up as an interval about twice the median; a *spurious* extra peak as a short one - both are removed.
4. `hr_from_ibi`: `60000 / median(IBI)` in bpm. The median resists remaining outliers. If fewer than `min_valid_ibis` (3) intervals survive, the peak method returns no value instead of guessing.

## Step 4 - Spectral estimation (`signal_processing/frequency_analysis.py`)

The "frequency-domain" method: a regular pulse is a peak in the spectrum.

- `power_spectrum`: Hann-windowed periodogram, zero-padded to 8192 points. Honest note: zero-padding smooths the curve but the true resolution is 1/window length = 0.125 Hz = 7.5 bpm for 8 s windows.
- `spectral_hr`: finds up to six spectral peaks between 40 and 200 bpm; the strongest gives HR = frequency × 60. All candidate peaks are kept for the motion step.

## Step 5 - Movement analysis and suppression

In `heart_rate.estimate_hr` and `frequency_analysis.py`:

1. Compute `motion_level` and its label from the ACC.
2. `motion_frequencies`: sum the power spectra of the three ACC axes and find the strongest peaks (0.5-4 Hz). A peak counts only if it holds at least 15 % of in-band power. **This step is skipped when motion is `low`**; otherwise sensor noise would be mistaken for movement (a bug found and fixed during testing).
3. `motion_suppressed_hr`: walk through PPG spectral peaks from strongest to weakest and take the first one that is **not** within ±0.12 Hz of a motion frequency or its second harmonic. If everything is excluded, fall back to the strongest peak and add a warning.

Known weakness: if the true heart rate happens to equal the movement rhythm (common when brisk walking), the correct peak is excluded. This is observed on synthetic data and documented in `reports/limitations.md`.

## Step 6 - Methods and fusion (`estimation/heart_rate.py`)

`estimate_hr(ppg, fs_ppg, acc, fs_acc, method, config)` processes **one window** and computes all candidates:

| Method | How |
|---|---|
| `peaks` | Step 3 |
| `spectral` | Step 4 |
| `spectral_motion` | Step 5 (requires ACC; raises `ValueError` without it) |
| `fusion` | If peaks and spectral are both available and agree within 5 bpm **and** movement is not `high`: their mean. Otherwise the motion-suppressed value (or plain spectral if no ACC). If only one source exists, use it. |

It also performs the **plausibility checks**: result within 40-200 bpm, flat-signal detection, peak-vs-spectral disagreement warning, rejection of non-finite or too-short input. `estimate_hr_series` slides this over a longer signal (8 s windows, 2 s step - the same grid as the dataset's reference labels, so estimates and labels line up one-to-one). A window that raises (for example it overlaps an unrepaired NaN gap) becomes an *invalid* estimate rather than stopping the run.

`apply_temporal_continuity` (optional post-processing, the `fusion_continuity` variant): a jump of more than 12 bpm from the last accepted value in 2 s is physiologically unlikely, so the previous value is held for at most 4 windows; after that the new level is accepted.

## Step 7 - Signal quality and confidence (`signal_processing/signal_quality.py`)

Three indicators, combined into a score from 0 to 1:

| Indicator | Meaning | Weight |
|---|---|---|
| Spectral concentration | Fraction of in-band power within ±0.2 Hz of the estimated HR (and of its 2nd harmonic). A clean pulse concentrates power there. | 0.4 |
| Beat-template correlation | Each beat (from 0.35 × IBI before to 0.65 × IBI after the peak) is correlated with the average beat. Clean signals have repeatable beat shapes. | 0.3 |
| IBI regularity | 1 − CV/0.25 where CV = standard deviation / mean of valid IBIs. | 0.3 |

Labels: `high` ≥ 0.6, `medium` ≥ 0.4, else `low`. Flags explain problems (`low_spectral_concentration`, `insufficient_valid_beats`, `irregular_ibi`, `many_rejected_ibis`). **Thresholds are heuristic and not tuned on real data.**

**Per-window confidence** (`HREstimate.confidence`): `none` if invalid; `low` if the quality label is low, or methods disagree during high movement; `high` if quality is high, methods agree and movement is not high; otherwise `medium`.

## Step 8 - Evaluation against the reference (`evaluation/`)

This step is the **only** place reference HR is turned into numbers.

**Reference extraction** (`metrics.py`):
- `reference_hr_from_labels`: the dataset's label *i* describes the window [2i, 2i+8] s. For a segment, the mean of all label windows fully inside it is used. If none fit, the label nearest the segment centre is used and flagged `approximate`.
- `reference_hr_from_rpeaks`: ECG R-peak sample indices ÷ 700 Hz give beat times; the median IBI (after plausibility filtering) inside the segment gives an independent ECG-based HR.

**Metrics** (`error_metrics`), with error = estimate − reference, NaN pairs dropped:
- **MAE** mean |error| · **RMSE** √mean(error²) (penalises big misses) · **bias** mean error (systematic over/under-estimation) · **median absolute error** (robust to outliers) · error SD · Pearson r · share within 5 bpm · **coverage** (valid estimates / windows with a reference). Coverage matters: a method that refuses hard windows can look accurate.
- `group_metrics` / `bland_altman` for per-group tables and agreement limits (bias ± 1.96 SD).

**Experiments** (`experiments.py`):
1. `run_estimation` - sliding windows, signals only → table of all method outputs, SQI, motion, confidence.
2. `attach_reference` - adds reference HR (labels and ECG) and the activity label (most frequent annotation in each window). It does not alter any estimate (tested).
3. `summarize`, `summarize_by(activity | motion_label)`, `sqi_gating` (accuracy vs coverage if low-confidence windows are discarded), `worst_windows` (failure cases to inspect).

## Step 9 - The analysis tool (`tools/analysis_tool.py`)

`analyze_hr_segment(subject, start_s, duration_s, method, include_reference, ...)`:

1. **Validates** every argument (method name, numeric and finite times, start ≥ 0, duration 8-300 s, allowed subject, segment inside the recording). Problems return `{"status": "error", "error_message": ...}` instead of raising, so an LLM can read and relay them.
2. **Stage 1 - estimate:** cuts the segment, runs `estimate_hr_series`, and aggregates: the reported HR is the **median** of valid window estimates, with the window-to-window SD and range.
3. **Overall confidence:** `high` if ≥ 60 % of windows are high-confidence and SD ≤ 5 bpm; `low` if fewer than 40 % are medium/high, fewer than 50 % of windows are valid, or SD > 10 bpm; else `medium`; `none` if nothing valid.
4. Adds alternatives (median per method), signal-quality summary, movement summary, plausibility information (valid fraction, mean disagreement, aggregated warnings), annotated-activity fractions (context only) and **caveats** (always: not clinically validated; conditionally: synthetic data, high movement, low quality, no estimate).
5. **Stage 2 - evaluation (optional):** only now, if requested, reference values and the error are added under `evaluation`, with a note that the reference was not an input.

`TOOL_SCHEMA` is the machine-readable description handed to the LLM.

## Step 10 - The agent (`agent/agent.py`, `agent/prompts.py`)

`HRAgent.ask(message)` always starts with the same gates: empty/too-long message handling, then the **medical filter** (`MEDICAL_PATTERNS`): a diagnostic or health-advice question gets a fixed refusal and **no tool or model call**.

**Rule backend** (offline, deterministic - default without API key):
1. `parse_request` extracts start and duration from phrases like "from 120 s for 30 s", "start at 2 minutes for 45 seconds", or "100-130 s", and the method name if mentioned.
2. Calls the tool; `explain_result` writes the answer from the result fields only (segment, HR, confidence, other methods, quality, movement, warnings, reference comparison, caveats, disclaimer). Synthetic data produce a bold warning line.
3. Follow-up questions without a new segment ("why this confidence?", "how does it compare to the reference?") use `explain_followup` on the stored result.

**LLM backend** (needs `pip install anthropic` and `ANTHROPIC_API_KEY`): sends the system prompt and `TOOL_SCHEMA` to the Anthropic Messages API; when the model requests `analyze_hr_segment`, the agent runs it and returns the JSON as a `tool_result`; this repeats up to `max_tool_rounds`. The system prompt requires tool use for any HR value, numbers only from the tool result, uncertainty statements, no diagnosis. A post-check (`_guard`) appends a note if the answer contains diagnostic wording or a "bpm" figure not present in the tool result, and always ends with the disclaimer. Tested with a mocked client only.

`backend: auto` picks `llm` if both the SDK and key exist, else `rule`.

## Step 11 - Interfaces

**CLI (`app/cli.py`)**

| Command | What it does |
|---|---|
| `info` | dataset summary and loader warnings |
| `analyze --start S --duration D --method M [--json] [--no-reference]` | one segment through the tool, printed as explanation or JSON |
| `ask "text"` | one natural-language request to the agent |
| `chat` | interactive: asks participant, start, duration, method → prints the analysis → free questions to the agent (`new`, `quit`) |
| `explore --out DIR` | overview, example segment and activity-comparison plots |
| `evaluate --out DIR` | all windows, all methods: CSV tables + 7 plots |
| any command + `--synthetic` | generated TEST data (tagged everywhere) |

Exit codes: 0 success, 2 invalid request/data, 3 dataset not found.

**Notebook (`notebooks/exploration_and_experiments.ipynb`)** - the 17 required sections in order; uses real S1 if present, otherwise the tagged synthetic fallback.

## 12. Configuration reference (`configs/default.yaml`)

| Section.key | Default | Effect |
|---|---|---|
| `data.dir` | `data/raw/PPG_FieldStudy` | dataset folder (env `PASSOS_DATA_DIR` overrides) |
| `data.allowed_subjects` | `[S1]` | permitted subjects |
| `data.fs_ppg / fs_acc / fs_activity / fs_ecg` | 64 / 32 / 4 / 700 | sampling rates (per dataset documentation) |
| `data.max_nan_fraction`, `max_interp_gap_s` | 0.2, 2.0 | missing-data limits |
| `data.max_stream_duration_mismatch_s` | 5.0 | BVP/ACC sync tolerance |
| `preprocessing.ppg_bandpass_hz`, `filter_order` | [0.5, 4.0], 4 | filter |
| `estimation.window_s / step_s / min_window_s` | 8 / 2 / 6 | windowing |
| `estimation.hr_range_bpm` | [40, 200] | plausible HR |
| `estimation.peak_prominence_std` | 0.3 | peak sensitivity |
| `estimation.ibi_range_ms`, `ibi_max_rel_dev`, `min_valid_ibis` | [300,1500], 0.25, 3 | IBI plausibility |
| `estimation.spectral_nfft` | 8192 | zero padding |
| `estimation.motion_exclusion_tol_hz`, `motion_min_peak_share` | 0.12, 0.15 | motion suppression |
| `estimation.agreement_tol_bpm` | 5 | peaks-vs-spectral agreement |
| `estimation.continuity_max_jump_bpm`, `continuity_max_hold` | 12, 4 | temporal continuity |
| `quality.sqi_high / sqi_medium / spectral_halfwidth_hz` | 0.6 / 0.4 / 0.2 | SQI labels |
| `motion.low_std_g / high_std_g` | 0.05 / 0.2 | motion labels |
| `analysis.min_duration_s / max_duration_s` | 8 / 300 | tool limits |
| `agent.backend / model / max_tokens / max_tool_rounds` | auto / claude-sonnet-5-5 / 1024 / 4 | agent |

Use another config with `--config path.yaml` or `PASSOS_CONFIG`.

## 13. Reading a result (worked template)

```
Estimated HR: <median> bpm (method 'fusion', median over windows; window-to-window SD <x> bpm, range <a>-<b> bpm)
Confidence: high | medium | low | none
Signal quality: <label> (mean SQI <score>; <p>% of windows low quality)
Movement: <label> (mean dynamic-acceleration SD <g> g; <p>% of windows high-motion)
Other methods on the same segment: peaks ..., spectral ..., spectral_motion ...
Reference comparison (evaluation only, not used for the estimate): label HR ..., ECG R-peak HR ..., absolute error ...
```
How to interpret: trust the number most when confidence is `high`, SD is small and the other methods agree. A large SD, disagreement between methods, or high movement means the value may follow arm movement rather than the heartbeat. A single segment's error says little about overall accuracy; use `evaluate` for that.

## 14. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `PPG-DaLiA was not found` (exit 3) | place `S1/S1.pkl` under `data/raw/PPG_FieldStudy/` or set `PASSOS_DATA_DIR`; `--synthetic` for a no-data smoke test |
| `DataError ... does not look like a PPG-DaLiA pickle` | file is not the official `S1.pkl`, or its keys differ from `data/README.md` |
| `Segment ends at ... but the recording is only ...` | choose start + duration inside the recording (see `info`) |
| Many "window rejected" warnings | NaN gaps longer than 2 s in that region |
| Agent says "Tell me which segment" | give start and duration, e.g. "from 120 s for 30 s" |
| `backend llm` error | install `anthropic`, set `ANTHROPIC_API_KEY`, or use `--backend rule` |
| Notebook shows SYNTHETIC | dataset not found; add it and re-execute |
