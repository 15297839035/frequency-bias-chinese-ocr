# Optional Zenodo deposit

The journal accepts a GitHub link for code, but a DOI-issuing repository is the stronger form
of the Elsevier data-availability statement (Option C). This note records the metadata to use
if the prediction archive is also deposited on Zenodo.

## Recommended record

* **Title**: Per-sample predictions for "Frequency, not shape, decides how Chinese text
  recognizers misread rare characters"
* **Authors**: Heyu Zhang, Jin Liu (School of Software, Jiangxi Normal University)
* **Upload type**: Dataset
* **File**: `predictions/per_sample_predictions.tar.gz` (323,090,535 bytes; SHA-256
  `7ccac4675605f20f245057c5273aff4f97066d87901add05ee7d35b97771ab68`)
* **Licence**: CC BY 4.0
* **Keywords**: Chinese text recognition; long-tailed recognition; logit adjustment; label
  prior; CTC; per-sample predictions
* **Related identifiers**:
  - `is supplemented by` / `is part of`: the GitHub repository
    <https://github.com/heyuuuuuuuuu/frequency-bias-chinese-ocr>
  - `is supplement to`: the paper DOI once assigned
* **Description**:

  > Per-sample prediction outputs (245 JSONL files, 2,410,320,768 bytes uncompressed) for the
  > experiments in the paper. The archive restores into `results/<experiment>/dumps/` and
  > contains no images, model weights or training logs.

## Steps

1. Create a Zenodo account and a new upload.
2. Upload `predictions/per_sample_predictions.tar.gz`.
3. Fill the metadata above; reserve a DOI.
4. Publish; copy the DOI.
5. Update the paper's Data availability statement and `cover_letter.tex` to cite the DOI
   alongside the GitHub repository, then rebuild the submission package.
6. Add the DOI to the repository README and to `predictions/README.md`.

A GitHub tag/release is an acceptable interim step; Zenodo adds a citable DOI and long-term
preservation.
