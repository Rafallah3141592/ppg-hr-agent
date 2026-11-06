# PPG Heart Rate Agent and Function of Analysis 

Heart-rate (HR) estimation from the **wrist PPG** of PPG-DaLiA participant **S1**, with accelerometer-based movement analysis,
reference-based evaluation, and an evidence-grounded AI analysis agent. Classical signal processing (NumPy/SciPy); no deep learning.

> ## Status - read this first
> * The PPG-DaLiA dataset **could not be downloaded** in the environment where this project was built (network allowlist returned HTTP 403).
>   **No result in this repository comes from real PPG-DaLiA data.** Everything was verified with unit tests and with **synthetic** signals,
>   and all synthetic outputs are labelled `SYNTHETIC (not PPG-DaLiA)`.
> * Code paths that need the real file (`load_subject` on the real pickle, real-S1 metrics, real plots) are implemented but **unverified on real data**.
>   Download the dataset (see `data/README.md`) and run the two commands in *Reproduce on real data* to get real results.
> * Not a medical device. The agent refuses diagnostic or health-advice questions.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt        # core + pytest + notebook execution
pip install -e .                           # optional (the CLI also works without installing)
python -m pytest -q                        # 73 tests
python app/cli.py analyze --synthetic --start 100 --duration 24    # works with no dataset (TEST data)
```

### Reproduce on real data

```bash
# 1) put S1 at data/raw/PPG_FieldStudy/S1/S1.pkl  (see data/README.md), then:
python app/cli.py info                                        # dataset summary, validation warnings
python app/cli.py evaluate --out outputs/real                 # all methods on all S1 windows: tables + plots
python app/cli.py explore  --out outputs/real                 # EDA plots
jupyter nbconvert --to notebook --execute --inplace notebooks/exploration_and_experiments.ipynb   # real S1 results automatically
```

## Documentation map

| Read | For |
|---|---|
| `docs/system_guide.md` | **Full step-by-step explanation** of the system: data loading → filtering → peaks/IBI → spectrum → motion handling → fusion → quality/confidence → evaluation → tool → agent → CLI; config table; troubleshooting |
| `docs/file_reference.md` | **Every file** in the repository and what it contains |
| `docs/architecture.md` | Diagram, design decisions, module map |
| `reports/technical_report.md`, `experiments.md`, `limitations.md` | Method report, execution log with real outcomes, requirement status and failure cases |
| `data/README.md`, `outputs/README.md` | Dataset instructions; how artifact folders are tagged |

## Interfaces

| Goal | Command |
|---|---|
| Analyse a segment (participant, start, duration, method) | `python app/cli.py analyze --subject S1 --start 120 --duration 30 --method fusion` |
| Machine-readable result | add `--json`; hide the reference with `--no-reference` |
| Ask the agent | `python app/cli.py ask "analyse S1 from 2 min for 45 s using spectral"` |
| Interactive session (prompts for S1, start, duration, method; then Q&A with the agent) | `python app/cli.py chat` |
| Full evaluation | `python app/cli.py evaluate --out outputs/real` |
| EDA plots | `python app/cli.py explore --out outputs/real` |
| Smoke test without data | add `--synthetic` to any command (TEST data only) |

Methods: `peaks` (band-pass → peaks → IBI → median), `spectral` (dominant spectral peak), `spectral_motion` (spectral peak after removing ACC movement frequencies),
`fusion` (default; agreement → mean, otherwise motion-suppressed spectrum). Exit codes: 0 ok, 2 invalid request, 3 dataset not found.

### Example (synthetic data, so the numbers are NOT dataset results)

```
$ python app/cli.py analyze --synthetic --start 100 --duration 24
** SYNTHETIC DATA - generated test signal, NOT PPG-DaLiA. ... **
Estimated HR: ... bpm (method 'fusion', median over windows; window-to-window SD ...)
Confidence: ...
Signal quality / Movement / Reference comparison (evaluation only) / Uncertainty / disclaimer
```

## AI agent

`HRAgent` wraps one tool, `analyze_hr_segment(subject, start_s, duration_s, method, include_reference)`, which returns structured JSON
(estimate, alternatives, signal quality, motion, plausibility, evidence-bearing caveats, and a separate `evaluation` block).

* **`rule` backend (default without API key):** deterministic parser → tool → templated, evidence-grounded explanation. Follow-up questions
  ("why this confidence?", "how does it compare to the reference?") are answered from the stored tool result only. **Executed and tested.**
* **`llm` backend:** real LLM function calling through the Anthropic Messages API. Install `pip install anthropic`, set `ANTHROPIC_API_KEY`
  (see `.env.example`), model in `configs/default.yaml → agent.model`. `backend: auto` switches to it when both are present.
  **Implemented; verified only with a mocked client in `tests/test_agent.py` — never run against the live API** (no key/network in the build environment).
* **Safety:** medical/diagnostic questions are refused before any model call; the LLM system prompt forbids diagnoses; LLM answers are
  post-checked for diagnostic wording and for bpm figures that are not in the tool result (a guardrail note is appended); every answer ends with a disclaimer.

## Repository layout

```
configs/default.yaml        all thresholds and paths      src/passos_wearable/   package (see docs/architecture.md)
app/cli.py                  terminal interface            tests/                 73 pytest tests
notebooks/                  executed walkthrough (17 sections)    docs/architecture.md
reports/                    technical_report, experiments, limitations          outputs/   generated artifacts (README explains tagging)
data/README.md              dataset instructions (no data shipped)
```

## Test and execution status

See `reports/experiments.md` for the exact outcomes. Summary: 73/73 tests pass; notebook executes end-to-end (on synthetic fallback); CLI,
analysis tool and rule-agent executed; **nothing executed on real PPG-DaLiA**.

## Resources and AI tools used

* Dataset: PPG-DaLiA, Reiss et al., *Sensors* 19(14):3079 (2019), UCI ML Repository #495. Not redistributed.
* Libraries: NumPy, SciPy, pandas, Matplotlib, PyYAML, pytest, nbformat/nbclient (optional: anthropic SDK).
* AI tooling: this repository was written with Claude (Anthropic) as a coding assistant. The optional runtime agent can call Claude via the Anthropic API.
  Algorithms follow standard PPG practice (band-pass + peak/spectral HR, ACC-guided motion-frequency exclusion); no third-party HR code was copied.

## Limitations

See `reports/limitations.md`. Key ones: nothing validated on real data yet; thresholds are heuristic; one participant; motion suppression fails when HR ≈ movement cadence.
