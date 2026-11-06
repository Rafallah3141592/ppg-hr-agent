# Experiments and execution log

Every entry says what was **actually run** and its status. Date of runs: 2026-11-05/06, Python 3.13, NumPy 2.5, SciPy 1.18, pandas 3.1, Matplotlib 3.11.

## 0. Blocker: real data

| Item | Status |
|---|---|
| Download PPG-DaLiA (UCI) from the build sandbox | **BLOCKED** - outbound HTTPS to `archive.ics.uci.edu` and `zenodo.org` returned HTTP 403 from the sandbox proxy. No alternative route was attempted. |
| Everything that needs the real S1 file | **NOT EXECUTED**: real loader run, real EDA observations, real metrics, real plots, notebook on real data. |
| Recovery | Put `S1.pkl` in `data/raw/PPG_FieldStudy/S1/` and run the commands in `README.md` ("Reproduce on real data"). The notebook switches to real data automatically. |

## 1. Execution checklist (Part 14.4)

| # | Step | Status |
|---|---|---|
| 1 | Install/verify dependencies (fresh venv from `requirements-dev.txt` + `pip install -e .`) | **Done**, passed |
| 2 | Import every project module (`pkgutil.walk_packages`) | **Done**, all import (fresh venv) |
| 3 | Run loader on real dataset | **Blocked** (see §0). Loader run on a fake pickle in the documented layout (unit tests) |
| 4 | Analysis on real S1 segments | **Blocked** |
| 5 | HR pipeline | **Done on synthetic** (unit tests, CLI, notebook) |
| 6 | Evaluation metrics | **Done on synthetic**, §3 |
| 7 | Visualisations | **Done on synthetic**, `outputs/synthetic_smoke_test/`, `outputs/notebook_run/` |
| 8 | Analysis tool | **Done** (unit tests, CLI, notebook; synthetic data) |
| 9 | Agent demonstration | **Done**, `rule` backend. `llm` backend: mocked-client test only, **not run against the live API** |
| 10 | Automated tests | **73 passed, 0 failed** (original tree and fresh venv) |
| 11-12 | Fix errors, re-test | Done, see §2 |

## 2. Defects found during execution and fixed

1. Motion suppression treated accelerometer *noise* peaks as movement in resting windows (synthetic sitting MAE 20.9 bpm for `spectral_motion`). Fixed with an absolute motion-level gate; sitting MAE dropped to 0.3 bpm.
2. `Axes.boxplot(labels=...)` raised on Matplotlib 3.11 (parameter renamed). Fixed in a version-independent way; a plot test now covers all 8 figures.
3. Tool rated a segment with 36 bpm window-to-window SD as "medium" confidence. Added an SD > 10 bpm → "low" rule and a regression test.
4. A test assumed a 1.56 s NaN gap would stay NaN while the config interpolates gaps ≤ 2 s; the test, not the code, was wrong.

## 3. Synthetic smoke experiment (NOT PPG-DaLiA)

**Data:** `make_synthetic_recording(900 s, seed=0)`: 5 activity blocks (sitting/walking/working/cycling/stairs), smooth HR 68-128 bpm, movement artefact in PPG correlated with ACC, Gaussian noise.
**Purpose:** show that the pipeline runs end-to-end and exposes the known failure mode. **These numbers say nothing about real accuracy**, and the synthetic walking/stairs blocks deliberately/accidentally place HR near the movement cadence (the hardest case for motion suppression).
Command: `python app/cli.py evaluate --synthetic --synthetic-seconds 900 --out outputs/synthetic_smoke_test` · 447 windows (8 s, step 2 s) · status: **executed**.

Overall vs. (synthetic) reference labels, tolerance 5 bpm:

| method | MAE | RMSE | bias | MedAE | r | within 5 bpm |
|:--|--:|--:|--:|--:|--:|--:|
| peaks | 3.26 | 7.65 | 1.65 | 0.92 | 0.92 | 0.82 |
| spectral | 2.64 | 6.28 | 0.94 | 0.23 | 0.95 | 0.85 |
| spectral_motion | 10.41 | 23.43 | 0.04 | 0.22 | 0.54 | 0.72 |
| fusion (default) | 9.77 | 22.35 | -0.62 | 0.20 | 0.57 | 0.73 |
| fusion + continuity | 6.50 | 12.71 | -2.75 | 0.20 | 0.78 | 0.73 |

MAE by activity (bpm): sitting 0.17-0.66 for all methods; working 0.18-1.74; cycling 0.34-3.7; **walking** spectral 5.17 / peaks 6.02 / fusion 23.75; **stairs** spectral 8.80 / peaks 9.04 / fusion 26.41.

Confidence gating, `fusion`: all windows MAE 9.77 · confidence ≥ medium MAE 10.17 (93 % kept) · confidence = high MAE 0.17 (40 % kept, 100 % within 5 bpm).

What this shows (about the code, not about wearables): (a) all 4 methods + continuity run and are evaluated consistently; (b) hard motion-frequency exclusion fails on synthetic data when HR ≈ cadence, so the default `fusion` is **not** shown to be best — on real data run `evaluate` and choose the method from the real table before using the default; (c) "high" confidence was reliable here, "medium" was not.

## 4. Test outcomes (`python -m pytest -q`) - 73 passed

| File | Tests | Covers |
|---|--:|---|
| test_data_loader.py | 12 | DaLiA-layout pickle parse, ACC unit fix, missing dataset error, subject validation, segment bounds, NaN gaps, sync trim, corrupt pickle |
| test_estimation.py | 25 | recovery of known HR (5 rates × 3 methods), ACC methods, motion suppression, noise/flat/invalid inputs, IBI cleaning, series, continuity, **no reference leakage**, **estimate independent of reference** |
| test_evaluation.py | 8 | metric values, NaN/coverage, groups, Bland-Altman, label/R-peak reference extraction, experiment tables, stage separation |
| test_analysis_tool.py | 15 | structure/JSON, reference optional & separate, 9 validation errors, missing dataset, all methods, schema, low-confidence regression |
| test_agent.py | 9 | request parsing, rule flow, follow-ups, medical refusal, errors, input limits, mocked LLM tool loop, guardrail |
| test_visualization.py | 1 | all 8 figures written |
| test_config_and_cli.py | 3 | config/env override, CLI synthetic run, CLI missing-dataset exit code |

## 5. Notebook

`notebooks/exploration_and_experiments.ipynb`: 17 titled sections, 16 code cells, **executed top-to-bottom with 0 errors** via nbclient.
Run mode in the shipped copy: **SYNTHETIC fallback** (sections 3-17 ran on generated data and are tagged as such; section 3 prints the BLOCKED message). Re-execute with the dataset present for real results.

## 6. Agent demonstration

Executed in notebook §16 and via `python app/cli.py ask ...` with the `rule` backend: segment request → tool call → grounded explanation; follow-up "why this confidence?" answered from the stored result; "do I have an arrhythmia?" refused with no tool call. LLM backend: see README (not run live).
