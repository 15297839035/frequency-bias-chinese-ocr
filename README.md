# Frequency, not shape, decides how Chinese text recognizers misread rare characters

Code, derived data and experiment outputs for the paper

> **Heyu Zhang, Jin Liu** — School of Software, Jiangxi Normal University
> Prepared for *Pattern Recognition Letters* (2026)

This repository is a research artifact, not a packaged library. It contains the analysis and
training code, the HWDB split indices, and the aggregate results behind every experiment in the
paper. It deliberately does **not** contain the raw images of the benchmark datasets, model
weights, or per-sample prediction files (see [What is not included](#what-is-not-included)).

The `ly/` prefix is kept from the authors' working tree because the manuscript and the
supplementary material refer to paths such as `ly/scripts/v4_c16.sh` and
`ly/exp/C-16_external/...`.

## Layout

| Path | Contents |
|---|---|
| `ly/scripts/` | Research code for data preparation, training, evaluation and analysis |
| `ly/data/hwdb/` | Indices of the HWDB-LT / HWDB-NAT / flip / dose training splits, plus their summaries |
| `ly/data/bctr/` | BCTR training character-frequency table used to build the splits |
| `ly/exp/` | Per-experiment results: JSON metrics, reports, model SHA-256, small derived features |
| `docs/experiments.md` | Map from experiment IDs to scripts, results and paper sections |
| `docs/data.md` | Split construction, dataset sources and access terms |
| `requirements.txt` | Python packages used by the analysis and training scripts |

## What is not included

| Omitted | Why | Where to get it |
|---|---|---|
| Raw BCTR images | Third-party benchmark, large | [FudanVI/BCTR](https://github.com/FudanVI/benchmarking-chinese-text-recognition) |
| Raw CASIA-HWDB images | Third-party database distributed under research-use terms; not redistributable here | [CASIA-HWDB](https://nlpr.ia.ac.cn/databases/handwriting/home.html) |
| ICDAR 2013 handwritten-text test images | Competition data, third party | ICDAR 2013 Robust Reading Competition (Yin et al., 2013) |
| Model checkpoints and logits | Hundreds of MB to several GB | Available from the authors on request |
| Per-sample predictions | Several hundred MB; not included in this Git repository | Available from the corresponding author on request |
| ASTER, TransOCR, OpenOCR source code | Third-party code with its own licence | [OpenOCR](https://github.com/Topdu/OpenOCR), [BCTR baselines](https://github.com/FudanVI/benchmarking-chinese-text-recognition) |
| Manuscript LaTeX source | Submitted separately to the journal | — |

The split indices in `ly/data/hwdb/` are derived data: they contain positions into the
CASIA-HWDB training set, not images or labels. They are released under CC BY 4.0
(see `LICENSE-DATA`).

## Dataset access

None of the raw datasets is redistributed in this repository.

* **BCTR** (scene, web and document Chinese text lines): public; data and baselines at
  <https://github.com/FudanVI/benchmarking-chinese-text-recognition> (MIT licence).
* **CASIA-HWDB 1.0/1.1** (isolated handwritten characters; HWDB 2.x lines):
  distributed by NLPR/CASIA for research use; see
  <https://nlpr.ia.ac.cn/databases/handwriting/home.html>. Please obtain it from the provider.
* **ICDAR 2013 Robust Reading Competition** handwritten-text test set: competition data;
  see Yin et al., *Text Localization, Detection and Recognition*, ICDAR 2013.

## Environment

The scripts were run with two environments:

* analysis and HWDB training: Python 3.12, PyTorch 2.8, NumPy 1.26 (`numpy<2`);
* OpenOCR-based text-recognition experiments: the OpenOCR environment
  (`openrec`, `tools`), with `OpenOCR-main/` on the import path.

```bash
pip install -r requirements.txt
# optional: clone OpenOCR next to this repository
git clone https://github.com/Topdu/OpenOCR.git OpenOCR-main
```

Several shell scripts contain absolute paths from the authors' server
(`/root/autodl-tmp/...`); edit the path variables at the top of a script before running it.
Python scripts resolve paths relative to the repository root (`Path(__file__).parents[2]`),
so keep the `ly/scripts/` layout.

## Reproducing the splits

Download CASIA-HWDB 1.0/1.1 and the ICDAR 2013 competition `.gnt` files, then:

```bash
python ly/scripts/a1_hwdb_prepare.py convert --name train --gnt-dirs <HWDB1.0trn> <HWDB1.1trn> --out ly/data/hwdb
python ly/scripts/a1_hwdb_prepare.py convert --name val   --gnt-dirs <HWDB1.0tst> <HWDB1.1tst> --out ly/data/hwdb
python ly/scripts/a1_hwdb_prepare.py convert --name test  --gnt-dirs <competition_gnt>        --out ly/data/hwdb
python ly/scripts/a1_hwdb_prepare.py classes --out ly/data/hwdb
python ly/scripts/a1_hwdb_prepare.py longtail --out ly/data/hwdb --imbalance 100 --nmax 500
python ly/scripts/a1_hwdb_prepare.py flip --out ly/data/hwdb --order rev
python ly/scripts/a1_hwdb_prepare.py flip --out ly/data/hwdb --order perm
python ly/scripts/a1_hwdb_prepare.py dose --out ly/data/hwdb --alpha 0.25   # also 0.5, 0.75
python ly/scripts/a1_hwdb_prepare.py natural --out ly/data/hwdb
```

The commands write the same `train_*_indices.npy` and `train_*_summary.json` files that are
included here. The split profile, seed and counts are recorded in each summary JSON.

## Reproducing the experiments

`docs/experiments.md` lists each experiment (D-03, A0/A1, C-01 … C-17, V4A) with its scripts,
result directory and the section of the paper it supports. The experiments require the
datasets above and, for the text-recognition experiments, OpenOCR and the corresponding model
checkpoints; the scripts are the original research code and some paths must be adapted.

## Results

`ly/exp/` contains the aggregate outputs only: metric JSON files, Markdown reports, model
SHA-256 records, and small derived arrays (glyph-neighbour lists, split weights). The
per-sample predictions used for the bootstrap intervals and McNemar tests are not included.

## Data and code availability

Code and derived data (split indices, aggregate results) are in this repository. Per-sample
predictions and trained weights are not included because of their size; they are available
from the corresponding author on reasonable request. Raw benchmark images remain under the
terms of their providers and are not redistributed.

## Licence

* Code: MIT (`LICENSE`).
* Derived data (split indices, summaries, aggregate results): CC BY 4.0 (`LICENSE-DATA`).
* Third-party datasets and code: original terms; see `docs/data.md`.

## Citation

```bibtex
@article{zhang2026frequency,
  title   = {Frequency, not shape, decides how Chinese text recognizers misread rare characters},
  author  = {Zhang, Heyu and Liu, Jin},
  journal = {Pattern Recognition Letters},
  year    = {2026},
  note    = {Manuscript prepared for submission}
}
```

If you use the split indices or code, please cite the paper and this repository.
