# Limitations, failure cases and future work

## Requirement status (assignment Part 14)

Legend: **V** implemented and verified (unit tests / synthetic run) · **U** implemented, *not* verified on real data · **X** incomplete.

| Requirement | Status | Notes |
|---|---|---|
| Obtain PPG-DaLiA | **U / X** | Instructions in `data/README.md`. Automatic download not implemented; download was blocked here (HTTP 403 from sandbox proxy). Next step: download manually. |
| Dataset loading, S1 selection, signal extraction | **U** | Parser tested on a fake pickle in the documented layout only. |
| Sampling-rate handling, time-based segments | **V** | `WristSignals.segment`, bounds-checked, tested. Rates come from config (64/32/4/700 Hz per dataset docs; not read from file). |
| Missing values, validation, synchronisation | **V** (synthetic/unit) | Short NaN gaps interpolated, long gaps kept NaN (windows rejected), >20 % missing → error, stream-length mismatch trimmed/rejected, ACC unit heuristic. Never exercised on real artefacts. |
| EDA plots, activity comparison | **U** | Plots implemented and rendered on synthetic data. **Observations "supported by the actual dataset" cannot be written until real data is run**; `reports/technical_report.md` contains none. |
| Preprocessing, filtering, peaks, IBI, HR, alternative methods | **V** on synthetic, **U** on real | 4 methods + continuity post-processing. |
| Plausibility checks, SQI, movement analysis | **V**/**U** | Thresholds are heuristic, untuned. |
| Reference extraction, isolation from estimation | **V** | Enforced by structure and tests. |
| MAE/RMSE/bias/MedAE, segment evaluation, plots, error analysis | **V** (code), **U** (real numbers) | |
| Failure cases documented | partially | Only failure cases seen on synthetic data (below). Real-data failure analysis pending. |
| Analysis tool, structured results, text interface | **V** | |
| LLM function calling | **U** | Mocked-client test only. |
| Deterministic agent fallback | **V** | |
| Uncertainty communication, no medical conclusions | **V** | Rule-based refusal + prompt + post-check; the regex guard is a safety net, not a guarantee. |
| Tests, config, logging, validation, dependency mgmt | **V** | Logging is minimal (loader warnings, tool failures). |
| Verification from a clean virtualenv | see `reports/experiments.md` | |
| GitHub publication | **X** | Repository is GitHub-ready; not published (needs your authorisation). |

## Known failure cases (observed on synthetic data; real behaviour unknown)

* **Motion lock-in:** when true HR is close to the movement cadence (or its 2nd harmonic), the hard exclusion in `spectral_motion`/`fusion` removes the *correct* peak.
  On the synthetic walking/stairs blocks (HR 105/128 bpm vs cadence 108/120 bpm) `fusion` MAE is 23-26 bpm while plain `spectral` is 5-9 bpm. Real wrist data may behave differently; this is exactly what real evaluation must settle before trusting the default `fusion`.
* **Unstable fusion output** (large window-to-window jumps) in high-motion segments; temporal continuity reduces MAE on synthetic data but only by holding earlier values.
* **SQI is not a reliable error predictor in all conditions:** on synthetic data "confidence ≥ medium" gating did not reduce MAE; only "high" did (at 40 % coverage).
* Windows with NaN gaps are rejected, not repaired.

## General limitations

* No real-data validation; all thresholds (`quality.*`, `motion.*`, `ibi_*`, `continuity_*`) are heuristic defaults. Tuning on S1 and evaluating on S1 would be optimistic; use other subjects for validation if scope allows.
* Single participant: no generalisation claim of any kind.
* 8 s windows smooth fast HR changes and limit spectral resolution to ~7.5 bpm (zero padding interpolates only).
* "Reference" is the dataset's ECG-derived HR, itself an estimate. Label window alignment is approximate for segments not starting on a 2 s boundary (flagged).
* Activity annotations are used only for grouping/context.
* Pickle loading executes code if the file is malicious - only use the official dataset.
* LLM backend can still produce wording the regex guard misses.
* Not a medical device; no clinical claims.

## Future work

Run on real S1 and tune/validate thresholds on other subjects; adaptive filtering (LMS/NLMS with ACC reference) and Viterbi-style HR tracking; replace hard motion exclusion by soft weighting; exercise LLM backend against the live API; automatic dataset download; richer logging.
