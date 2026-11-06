# Data

**The PPG-DaLiA dataset is not included in this repository and must not be committed or redistributed.**
`.gitignore` excludes `data/raw/`, `*.pkl`, `*.h5` and `data/*.zip`.

## Getting PPG-DaLiA

1. Open the UCI Machine Learning Repository page: <https://archive.ics.uci.edu/dataset/495/ppg+dalia>
   (Reiss, Indlekofer, Schell, Laerhoven: *Deep PPG: Large-Scale Heart Rate Estimation with Convolutional Neural Networks*, Sensors 2019, 19(14):3079). Check the dataset's licence/terms there.
2. Download and unzip it (about 3 GB).
3. Make the folder structure look like this (only `S1` is needed for this project):

```
data/raw/PPG_FieldStudy/
└── S1/
    └── S1.pkl
```

Any other location works too: `export PASSOS_DATA_DIR=/path/to/PPG_FieldStudy` or edit `data.dir` in `configs/default.yaml`.
The loader also accepts `<dir>/PPG_FieldStudy/S1/S1.pkl` and `<dir>/S1.pkl`.

## What the loader reads from `S1.pkl`

| Key | Used for | Rate |
|---|---|---|
| `signal.wrist.BVP` | PPG (estimation input) | 64 Hz |
| `signal.wrist.ACC` | movement analysis (estimation input) | 32 Hz |
| `activity` | context / grouping only | 4 Hz |
| `label` | reference HR, **evaluation only** | one value per 8 s window, 2 s step |
| `rpeaks` | ECG-derived reference HR, **evaluation only** | indices at 700 Hz |

The key layout follows the dataset documentation; it has **not** been checked against the real file in the environment where this
project was built (the download was blocked). The parser is unit-tested against a fake pickle in this layout. If your file differs, `DataError` will say which key is missing.

Security: `.pkl` files are Python pickles. Only load the official file you downloaded yourself.

## Checking your setup

```bash
python app/cli.py info
```
