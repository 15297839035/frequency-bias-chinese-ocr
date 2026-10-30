# Experiment map

The paper reports experiments D-03, A0/A1 and C-01 … C-17. The table maps each experiment to
its code, its result directory under `results/`, and the section of the paper it supports.
Section numbers refer to the current manuscript.

| ID | What it measures | Code (`src/…`) | Results (`results/`) | Paper |
|---|---|---|---|---|
| D-03 | BCTR long-tail diagnosis: rare-line accuracy gap, substitution types, look-alike ceiling | `analysis/d03_longtail_quickcheck.py`, `analysis/d03_lookalike_ceiling.py` | `D-03_bctr_longtail_diagnosis/` | §1, §5.1 |
| A0-01 | CTC top-16 posterior dumps, frozen-prediction check | `inference/dump_ctc_posteriors.py`, `inference/check_ctc_dumps.py` | `A0-01_ctc_dumps/` | §5.1 |
| A0-02 | Look-alike sets from IDS/glyph neighbours | derived arrays; used by `core/prior_correction.py` | `A0-02_lookalike/` | §5.1, §5.3 |
| A0-03 | G-LA / L-LA correction on BCTR (shared core used by later analyses) | `core/prior_correction.py` | `A0-03_prior_correction/` | §5.1, §5.5 |
| A0-05 | Glyph-neighbour lists (used by the error analysis) | derived arrays; used by `analysis/c03_confusion_analysis.py` | `A0-05_glyph_neighbors/` | §5.1 |
| A1-00 | HWDB conversion and split preparation | `data/prepare_hwdb.py` | `A1-00_prep/` | §4.1, §5.3 |
| A1-02 | HWDB-LT training (12 losses/seeds) and pre-registered verdict | `training/train_hwdb_classifier.py`, `pipelines/a1_hwdb_queue.sh`, `evaluation/summarize_hwdb_runs.py` | `A1-02_hwdb_lt/` | §5.3 |
| A1-03 | LA + smoothing on HWDB-LT (exploratory) | `training/train_hwdb_classifier.py`, `pipelines/a1_hwdb_la_queue.sh` | `A1-03_la_plus/` | §5.3 |
| C-01 | Balanced vs natural-frequency re-evaluation, τ curves | `evaluation/reevaluate_hwdb.py`, `figures/c01_tau_curves.py` | `C-01_bal_vs_nat/` | §5.3, §5.4 |
| C-01b | Additional seeds for C-01 | `pipelines/c01b_hwdb_seeds.sh` | `C-01b_hwdb_seeds/` | §5.3 |
| C-01n | HWDB-NAT (class counts ∝ real text) | `data/prepare_hwdb.py natural`, `evaluation/reevaluate_hwdb.py` | `C-01n_hwdb_nat/` | §5.3, §5.4 |
| C-02 | BCTR training-side conditional LA | `training/finetune_bctr_la.py`, `evaluation/evaluate_bctr.py`, `evaluation/evaluate_bctr_seeds.py` | `C-02_bctr_la/` | §5.5 |
| C-03 | Error mechanism: direction asymmetry, β, candidate coverage | `analysis/c03_confusion_analysis.py`, `analysis/c03_confusion_asymmetry.py` | `C-03_analysis/` | §5.1 |
| C-04 | External official SVTRv2 T/S/B models | `inference/dump_external_models.py`, `analysis/c04_external_analysis.py` | `C-04_external/` | §5.1, §5.5 |
| C-04b | External industrial OpenOCR models | `inference/dump_industrial_models.py` | `C-04b_industrial/` | §5.5 |
| C-05 | CRNN trained from scratch on BCTR | `training/train_crnn.py` | `C-05_crnn/` | §5.1, §5.5 |
| C-06 | Prior estimated from unlabelled text | `analysis/c06_self_prior.py` | `C-06_selfprior/` | §5.5 |
| C-07 | HWDB frequency flip / random order; direction test | `data/prepare_hwdb.py flip`, `pipelines/c07_frequency_flip.sh`, `analysis/c07_flip_direction.py`, `figures/c07_flip_figure.py` | `C-07_freq_flip/` | §5.2 |
| C-08 | G-LA ablations, sensitivity, annotation audit | `analysis/c08_correction_ablation.py`, `analysis/c08_label_audit.py`, `analysis/c08_export_examples.py`, `pipelines/c08_after_c05.sh` | `C-08_ablation/` | §5.3, §5.5 |
| C-09 | SMTR autoregressive step-wise correction | `inference/decode_smtr.py`, `analysis/c09_smtr_analysis.py`, `pipelines/c09_smtr.sh` | `C-09_smtr/` | §5.4, §5.5 |
| C-10 | HWDB long-tail baselines and τ-normalisation | `training/train_hwdb_classifier.py`, `analysis/c10_tau_normalization.py`, `analysis/c10_longtail_report.py`, `pipelines/c10_longtail.sh` | `C-10_lt_baselines/` | §5.3 |
| C-11 | CRNN trained from scratch with LA in the loss | `training/train_crnn_la.py`, `pipelines/c11_crnn_la.sh`, `evaluation/evaluate_crnn_la.py`, `pipelines/c11_after.sh` | `C-11_crnn_la/` | §5.5 |
| C-13 | Frequency swap of 54 confusable pairs in text lines | `data/line_swap.py`, `training/line_swap_sampler.py`, `training/train_crnn_line_swap.py` | `C-13_crnn_flip/` | §5.2 |
| C-14 | Dose–response of the training-frequency profile | `analysis/c14_dose_response.py`, `pipelines/c14_dose_early.sh` | `C-14_dose/` | §5.2 |
| C-15 | Frequency-matched random-class smoothing control | `training/train_hwdb_classifier.py`, `analysis/c15_random_smoothing_report.py` | `C-15_rand_smooth/` | §5.3 |
| C-16 | External baselines (ASTER, TransOCR, HWDB lines) | `inference/run_bctr_baselines.py`, `inference/run_hwdb_baselines.py`, `pipelines/c16_external_baselines.sh` | `C-16_external/` | §5.1, §5.5 |
| C-17 | Fine τ curves for CTC and SMTR | `analysis/c17_tau_curves.py` | `C-17_tau_curves/` | §5.4 |
| V4A | Descriptive direction / self-prior counts on CPU | `analysis/v4a_direction_counts.py`, `analysis/v4a_self_prior_counts.py`, `pipelines/v4_cpu.sh` | `V4A_cpu/` | §5.1, §5.5 |
| — | Paper figures | `figures/paper_figures.py`, `figures/revision_figures.py` | manuscript figures | — |
| — | Pipeline orchestration and restart | `pipelines/restart_pipeline.sh`, `pipelines/rerun_c05.sh`, `pipelines/v4_pipeline.sh`, `pipelines/c05_crnn_cpu.sh`, `pipelines/c10_longtail_cpu.sh` | logs in `results/` | — |

## Notes

* Results are aggregate: metric JSON, `report.md`, model SHA-256 and small derived arrays.
  Per-sample predictions and checkpoints are not in the repository (see the README).
* The scripts are the original research code. Shell pipelines contain absolute paths from the
  authors' GPU server (`/root/autodl-tmp/...`); edit them before running. Python scripts
  resolve `REPO = Path(__file__).resolve().parents[2]` (the repository root) and insert
  `src/` on `sys.path`, so keep the `src/<package>/<script>.py` layout or set
  `PYTHONPATH=$PWD/src`.
* `data/prepare_hwdb.py` reads the BCTR frequency table from
  `OpenOCR-main/output/rec/bctr_scheme51_ab_bs64/train_character_frequency.json` and falls
  back to `data/bctr/train_character_frequency.json`, which is included here.
* OpenOCR-based modules (`inference/*`, several training/evaluation/analysis scripts) need
  OpenOCR and its Python environment; the HWDB scripts use the `numpy<2` environment
  described in the README.

## Utilities

| Module | Purpose |
|---|---|
| `core/selftest.py` | Sanity checks for the correction and decoding helpers |
| `analysis/posterior_ceiling.py` | Exploratory posterior-ceiling analysis for the CTC dumps |
| `data/restore_bctr_splits.py` | Restore the BCTR split files from the benchmark bundle |
| `training/line_swap_sampler.py` | OpenOCR sampler with per-line exposure weights (C-13) |
| `inference/check_ctc_dumps.py` | Verify that the exported CTC posteriors reproduce the frozen predictions |

## Original path mapping

The supplementary material refers to the authors' original paths `ly/scripts/<name>.py`.
The table above gives the new location. The main renames are:

| Original | New |
|---|---|
| `a0_prior_correction.py` | `core/prior_correction.py` |
| `a0_dump_ctc.py`, `a0_check_dump.py` | `inference/dump_ctc_posteriors.py`, `inference/check_ctc_dumps.py` |
| `a1_hwdb_prepare.py` | `data/prepare_hwdb.py` |
| `a1_train_hccr.py` | `training/train_hwdb_classifier.py` |
| `c1_reeval_hwdb.py` | `evaluation/reevaluate_hwdb.py` |
| `c2_finetune.py`, `c2_eval.py`, `c2b_eval.py` | `training/finetune_bctr_la.py`, `evaluation/evaluate_bctr.py`, `evaluation/evaluate_bctr_seeds.py` |
| `c3_analysis.py`, `c3b_asymmetry.py` | `analysis/c03_confusion_analysis.py`, `analysis/c03_confusion_asymmetry.py` |
| `c4_dump_external.py`, `c4b_dump_industrial.py`, `c4_analysis.py` | `inference/dump_external_models.py`, `inference/dump_industrial_models.py`, `analysis/c04_external_analysis.py` |
| `c5_train_crnn.py` | `training/train_crnn.py` |
| `c7_flip_direction.py`, `c7_figure.py` | `analysis/c07_flip_direction.py`, `figures/c07_flip_figure.py` |
| `c9_smtr_decode.py`, `c9_analysis.py` | `inference/decode_smtr.py`, `analysis/c09_smtr_analysis.py` |
| `c11_train_crnn_la.py`, `c11_eval.py` | `training/train_crnn_la.py`, `evaluation/evaluate_crnn_la.py` |
| `c13_flip_lines.py`, `c13_sampler.py`, `c13_train_crnn_flip.py` | `data/line_swap.py`, `training/line_swap_sampler.py`, `training/train_crnn_line_swap.py` |
| `c16_bctr_baselines.py`, `c16_hwdb.py` | `inference/run_bctr_baselines.py`, `inference/run_hwdb_baselines.py` |
| `longtail_quickcheck.py`, `lookalike_ceiling.py` | `analysis/d03_longtail_quickcheck.py`, `analysis/d03_lookalike_ceiling.py` |
| `w1_figures.py`, `v4_figures.py` | `figures/paper_figures.py`, `figures/revision_figures.py` |
| `*_pipeline.sh`, `*_run_queue*.sh`, `m10_*`, `m11_*` | `pipelines/` (descriptive names) |
