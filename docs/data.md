# Data: HWDB splits and derived tables

This directory contains **derived data only**. No raw images, labels or competition
annotations are redistributed.

## `data/hwdb/`

| File | Shape / size | Meaning |
|---|---|---|
| `train_nat_indices.npy` | `(403741,)` int64 | HWDB-NAT: class counts proportional to BCTR training frequency, clipped to `[2, 500]` |
| `train_lt100_indices.npy` | `(403741,)` int64 | HWDB-LT: 500 → 5 images per class (exponential, imbalance 100), assigned by BCTR frequency rank |
| `train_lt100rev_indices.npy` | `(401743,)` int64 | HWDB-LT with the rank reversed (12 classes capped by available data) |
| `train_lt100perm_indices.npy` | `(403597,)` int64 | HWDB-LT with a random permutation of the profile (seed 0; 1 class capped) |
| `train_lt100dose025_indices.npy` | `(403741,)` int64 | Dose interpolation, α = 0.25 (natural → reversed) |
| `train_lt100dose050_indices.npy` | `(404402,)` int64 | Dose interpolation, α = 0.50 (every class has 107–108 images) |
| `train_lt100dose075_indices.npy` | `(403741,)` int64 | Dose interpolation, α = 0.75 |
| `train_*_summary.json` | — | Profile, seed, per-class count statistics, Spearman correlation with the natural rank |
| `classes.txt` | 3,755 characters | GB2312 level-1 class list used by the HWDB classifiers and the ICDAR 2013 test set |

Each index array points into the concatenated training-image array produced by
`src/data/prepare_hwdb.py convert` from the CASIA-HWDB 1.0/1.1 `.gnt` files. The script
sorts the `.gnt` files by path and reads their records in order; the same order is used for
images, characters and writer IDs. Run the script on the same dataset to reproduce the array
that the indices refer to.

The summaries are the authoritative description of each split; e.g.
`train_nat_summary.json` records the scale, clipping and per-bucket counts, and
`train_lt100dose050_summary.json` records `alpha`, `scale_s` and the resulting count range.

Example:

```python
import numpy as np
idx = np.load("data/hwdb/train_lt100_indices.npy")
print(idx.shape, idx.dtype, idx[:10], idx.min(), idx.max())
```

## `data/bctr/train_character_frequency.json`

Add-one-free raw counts of characters in the BCTR training labels (6,624 classes,
schema `scheme51-train-frequency-v1`), extracted from the BCTR training text. The file
contains the character list, counts, label hashes and dataset source names; it contains no
text lines. It is used to rank classes for the HWDB-LT profile and to compute the
natural-frequency metrics. The `sources[*].path` entries were rewritten to dataset-relative
names.

Fields:

* `characters`: 6,624 entries, including `blank`;
* `counts`: integer occurrences per entry;
* `counting`: raw CTC class IDs from the encoder, no case folding or metric normalisation;
* `max_text_length`: 25; `use_space_char`: false;
* `labels_hash_encoding`: hash format of the label files used to build the counts.

## What the upstream datasets are

* **BCTR** — 1.02M / 127k / 127k scene, web and document lines; 6,623-character PP-OCR
  dictionary. Public: <https://github.com/FudanVI/benchmarking-chinese-text-recognition>
  (MIT licence for the repository; data hosted by the authors).
* **CASIA-HWDB 1.0/1.1** — isolated handwritten Chinese characters, 3,755 GB2312 level-1
  classes; training and validation writers. Distributed by NLPR/CASIA for research use:
  <https://nlpr.ia.ac.cn/databases/handwriting/home.html>.
* **CASIA-HWDB 2.x and ICDAR 2013** — handwritten text lines used as an external domain
  (validation: HWDB 2.x; test: ICDAR 2013 competition set).
* **ICDAR 2013 Robust Reading Competition** — handwritten-text test images; competition
  data, see Yin et al., ICDAR 2013.

We do not redistribute any of these images. Users must obtain them from the providers and
follow their terms. In particular, CASIA-HWDB is provided for research use and is not
redistributed here.

## What is intentionally absent

* `test_images.npy.gz` (268 MB), the resized ICDAR 2013 test images;
* `test_labels.npy`, `test_writers.npy`, `*_chars.txt`, `*_files.json` from the authors'
  working copy (dataset annotations and absolute server paths).

These are either raw/derived dataset content or machine-specific provenance. The split
indices and summaries above, together with `src/data/prepare_hwdb.py`, are sufficient to rebuild
them from the upstream datasets.
