# Technical report: wrist-PPG HR estimation on PPG-DaLiA S1

**Read first:** no part of this report is based on real PPG-DaLiA data because the dataset could not be downloaded in the build environment (see `experiments.md` §0). This report documents the *method and implementation*; real-data results and observations are pending and are deliberately not invented here.

## 1. Problem and scope
Estimate HR (bpm) from the Empatica E4 wrist BVP (64 Hz) of participant S1, using the E4 accelerometer (32 Hz) for movement analysis; compare with the ECG-derived reference; expose the analysis as a tool for an agent that explains results with uncertainty and without medical claims.

## 2. Data handling (`data.loader`)
* Reads `S1/S1.pkl` (BVP, ACC, activity, label, rpeaks). Sampling rates come from config and are checked via durations (BVP vs ACC length must agree within 5 s; otherwise `DataError`; small mismatches are trimmed to the common duration).
* Missing values: non-finite gaps ≤ 2 s interpolated linearly; longer gaps stay NaN and any window touching them is rejected; > 20 % missing → error.
* ACC unit heuristic: median |a| > 10 ⇒ raw E4 counts ⇒ divide by 64 (so motion thresholds are in g).
* Reference HR (`label`, `rpeaks`) lives in `ReferenceData`, never in a `SignalSegment`.
* Time-based segments: `segment(start_s, duration_s)` with bounds checks.

## 3. Estimation (`preprocessing`, `signal_processing`, `estimation`)
Window = 8 s (matches the dataset label window), step 2 s.
1. **Band-pass** 0.5-4 Hz (30-240 bpm), 4th-order Butterworth, zero phase; standardise.
2. **Peaks → IBI:** `find_peaks` with minimum distance from the max HR and prominence 0.3·SD; parabolic sub-sample refinement; IBI filter: 300-1500 ms and within 25 % of the median (rejects missed/spurious beats); HR = 60000/median IBI; needs ≥ 3 valid IBIs.
3. **Spectral:** Hann-windowed zero-padded periodogram; strongest peak in 40-200 bpm.
4. **Motion-suppressed spectral:** dominant ACC frequencies (3-axis summed spectrum, each holding ≥ 15 % of in-band power, only when the window's motion level is not "low") and their 2nd harmonics are excluded (±0.12 Hz) from the PPG peak candidates; fallback to the plain peak if all are excluded.
5. **Fusion:** if peaks and spectral agree within 5 bpm and motion is not high → mean; else motion-suppressed spectrum (or spectral without ACC).
6. **Temporal continuity (post-processing):** jumps > 12 bpm from the last accepted value are held for ≤ 4 windows.
7. **Plausibility:** 40-200 bpm range, peak/spectral agreement, IBI filter, flat/NaN input rejection.

## 4. Quality and movement
* **SQI** (0-1) = 0.4·spectral concentration (power within ±0.2 Hz of HR and 2×HR / in-band power) + 0.3·beat-template correlation + 0.3·IBI regularity (1 − CV/0.25). Labels: ≥ 0.6 high, ≥ 0.4 medium.
* **Movement:** SD of 0.3-8 Hz band-passed ACC magnitude (g); < 0.05 low, ≥ 0.2 high.
* **Confidence:** none (invalid) / low (low SQI, or disagreement under high motion) / high (high SQI, agreement, motion not high) / medium otherwise. For segments, window medians are aggregated and confidence also depends on valid fraction and window-to-window SD.
All thresholds are heuristic and untuned (config).

## 5. Evaluation (`evaluation`)
Reference per window: mean of dataset labels for label windows inside the segment (nearest label, flagged *approximate*, otherwise); ECG R-peak HR from median IBI inside the window. Metrics: MAE, RMSE, bias (est − ref), median absolute error, SD of error, Pearson r, share within 5 bpm, coverage (valid / available). Groupings: activity, movement label, SQI/confidence gating, worst windows, Bland-Altman. Estimation and reference attachment are separate stages; leakage is test-enforced.

## 6. Agent
Tool `analyze_hr_segment` → structured JSON. Rule backend (offline) and LLM backend (Anthropic tool use) share it. Guardrails: medical-question refusal before any model call; system prompt; post-check for diagnostic wording and ungrounded bpm figures; disclaimer on every answer; synthetic data announced.

## 7. Results
**Real S1 results: not available (blocked).** A synthetic smoke test and its interpretation are in `experiments.md` §3; they demonstrate that the code runs and reveal a motion-lock-in failure mode of hard frequency exclusion, not any property of PPG-DaLiA.

## 8. Unsuccessful approaches (so far, synthetic only)
* Unconditional ACC-peak suppression: noise treated as motion at rest (fixed by gating).
* Hard motion suppression at HR ≈ cadence: removes the true peak (open problem; see future work).
* Mean-confidence aggregation ignoring window scatter (fixed).

## 9. Reproducibility
`pip install -r requirements-dev.txt`, `pytest`, `python app/cli.py evaluate --out outputs/real`; deterministic (no randomness outside the seeded synthetic generator).
