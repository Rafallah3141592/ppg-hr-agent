# Outputs

Generated artifacts (plots, CSV metrics). Subfolders are tagged by data source:

| Folder | Contents | Data |
|---|---|---|
| `synthetic_smoke_test/` | plots + metrics from `app/cli.py evaluate/explore --synthetic` | **SYNTHETIC generated signals. NOT PPG-DaLiA. Proves the pipeline runs; says nothing about real accuracy.** |
| `notebook_run/` | artifacts of the last notebook execution | Tagged in `SOURCE.txt`; the shipped run used SYNTHETIC data |
| `real/` (created by you) | `python app/cli.py evaluate --out outputs/real` after you download PPG-DaLiA | real S1 |

Every figure title contains the data-source tag. `SOURCE.txt` in each folder records it too.
