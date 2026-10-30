# Frequency, not shape, decides how Chinese text recognizers misread rare characters

Code, derived data and experiment outputs for the paper

> **Heyu Zhang, Jin Liu** — School of Software, Jiangxi Normal University
> Prepared for *Pattern Recognition Letters* (2026)

**Artifact version:** v1.0.0

This repository is a research artifact, not a packaged library. It contains the analysis and
training code, the HWDB split indices, and the aggregate results behind every experiment in
the paper. It deliberately does **not** contain the raw images of the benchmark datasets,
model weights, or per-sample prediction files (see [What is not included](#what-is-not-included)).

## Layout

| Path | Contents |
|---|---|
| `src/core/` | Shared correction and selection code (`prior_correction.py`, `selftest.py`) |
| `src/data/` | HWDB conversion and split construction, line-swap data preparation |
| `src/inference/` | Posterior dumps, external/industrial model inference, SMTR decoding |
| `src/training/` | HWDB classifiers, BCTR fine-tuning, CRNN training (plain, LA, line-swap) |
| `src/evaluation/` | Re-evaluation and metric aggregation for HWDB and BCTR |
| `src/analysis/` | Confusion analysis, interventions, ablations, τ curves, reports |
| `src/figures/` | Paper and revision figures |
| `src/pipelines/` | Original shell pipelines used on the GPU server |
| `data/hwdb/` | Indices of the HWDB-LT / NAT / flip / dose training splits, plus summaries |
| `data/bctr/` | BCTR training character-frequency table used to build the splits |
| `results/` | Per-experiment results: JSON metrics, reports, model SHA-256, derived features |
| `predictions/` | Per-sample prediction archive (Git LFS): 245 JSONL files, 2.41 GB uncompressed |
| `docs/experiments.md` | Experiment ID → code → results → paper section |
| `docs/data.md` | Split construction, dataset sources and access terms |

The manuscript and the supplementary material refer to the authors' original working paths
(`ly/scripts/...`, `ly/exp/...`). This repository uses the conventional layout above;
`docs/experiments.md` maps each experiment to its new location. The original `ly/` prefix was
only a working-directory name and is not part of the artifact.

## What is not included

| Omitted | Why | Where to get it |
|---|---|---|
| Raw BCTR images | Third-party benchmark, large | [FudanVI/BCTR](https://github.com/FudanVI/benchmarking-chinese-text-recognition) |
| Raw CASIA-HWDB images | Third-party database distributed under research-use terms; not redistributable here | [CASIA-HWDB](https://nlpr.ia.ac.cn/databases/handwriting/home.html) |
| ICDAR 2013 handwritten-text test images | Competition data, third party | ICDAR 2013 Robust Reading Competition (Yin et al., 2013) |
| Model checkpoints and logits | Hundreds of MB to several GB | Available from the authors on request |
| ASTER, TransOCR, OpenOCR source code | Third-party code with its own licence | [OpenOCR](https://github.com/Topdu/OpenOCR), [BCTR baselines](https://github.com/FudanVI/benchmarking-chinese-text-recognition) |
| Manuscript LaTeX source | Submitted separately to the journal | — |

The split indices in `data/hwdb/` are derived data: they contain positions into the
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

The code is organised as small namespace packages. Put `src` on the import path when
running a script directly:

```bash
export PYTHONPATH="$PWD/src"
python src/data/prepare_hwdb.py --help
```

Several shell pipelines contain absolute paths from the authors' GPU server
(`/root/autodl-tmp/...`); edit the variables at the top of a pipeline before running it.
The Python scripts are the original research code and are not packaged for turnkey
reproduction.

## Reproducing the splits

Download CASIA-HWDB 1.0/1.1 and the ICDAR 2013 competition `.gnt` files, then:

```bash
python src/data/prepare_hwdb.py convert --name train --gnt-dirs <HWDB1.0trn> <HWDB1.1trn> --out data/hwdb
python src/data/prepare_hwdb.py convert --name val   --gnt-dirs <HWDB1.0tst> <HWDB1.1tst> --out data/hwdb
python src/data/prepare_hwdb.py convert --name test  --gnt-dirs <competition_gnt>        --out data/hwdb
python src/data/prepare_hwdb.py classes --out data/hwdb
python src/data/prepare_hwdb.py longtail --out data/hwdb --imbalance 100 --nmax 500
python src/data/prepare_hwdb.py flip --out data/hwdb --order rev
python src/data/prepare_hwdb.py flip --out data/hwdb --order perm
python src/data/prepare_hwdb.py dose --out data/hwdb --alpha 0.25   # also 0.5, 0.75
python src/data/prepare_hwdb.py natural --out data/hwdb
```

The commands write the same `train_*_indices.npy` and `train_*_summary.json` files that are
included here. The split profile, seed and counts are recorded in each summary JSON.

## Reproducing the experiments

`docs/experiments.md` lists each experiment (D-03, A0/A1, C-01 … C-17, V4A) with its code,
result directory and the section of the paper it supports. The experiments require the
datasets above and, for the text-recognition experiments, OpenOCR and the corresponding
model checkpoints.

## Predictions

`predictions/per_sample_predictions.tar.gz` contains the per-sample predictions of all
experiments: 245 JSONL files, 2,410,320,768 bytes uncompressed. The archive preserves the
repository-relative paths and restores into `results/<experiment>/dumps/`:

```bash
git lfs pull                       # if the archive is not present after cloning
tar -xzf predictions/per_sample_predictions.tar.gz -C .
```

The archive contains no images, model weights or training logs. Its size and SHA-256 are
recorded in `predictions/README.md`. Direct download without a Git LFS client:
<https://media.githubusercontent.com/media/15297839035/frequency-bias-chinese-ocr/main/predictions/per_sample_predictions.tar.gz>.

## Results

`results/` contains the aggregate outputs only: metric JSON files, Markdown reports, model
SHA-256 records, and small derived arrays (glyph-neighbour lists, split weights). The
per-sample predictions used for the bootstrap intervals and McNemar tests are stored in
`predictions/` (see [Predictions](#predictions)); raw model checkpoints and logits are
available from the corresponding author on request.

## Data and code availability

Code, derived data (split indices, aggregate results) and per-sample predictions are in this
repository; the predictions are stored with Git LFS. Model checkpoints are not included and
are available from the corresponding author on request. Raw benchmark images remain under
the terms of their providers and are not redistributed.

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
