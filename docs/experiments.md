# Experiment map

The paper reports experiments D-03, A0/A1 and C-01 … C-17. The table maps each experiment to
its code, its result directory under `ly/exp/`, and the section of the paper it supports.
Section numbers refer to the current manuscript (`ly/paper/latex/main.tex`).

| ID | What it measures | Code (`ly/scripts/`) | Results (`ly/exp/`) | Paper |
|---|---|---|---|---|
| D-03 | BCTR long-tail diagnosis: rare-line accuracy gap, substitution types, look-alike ceiling | `longtail_quickcheck.py`, `lookalike_ceiling.py` | `D-03_bctr_longtail_diagnosis/` | §1, §5.1 |
| A0-01 | CTC top-16 posterior dumps, frozen-prediction check | `a0_dump_ctc.py`, `a0_check_dump.py` | `A0-01_ctc_dumps/` | §5.1 |
| A0-02 | Look-alike sets from IDS/glyph neighbours | derived arrays; used by `a0_prior_correction.py` | `A0-02_lookalike/` | §5.1, §5.3 |
| A0-03 | G-LA / L-LA correction on BCTR (core functions used by later analyses) | `a0_prior_correction.py` | `A0-03_prior_correction/` | §5.1, §5.5 |
| A0-05 | Glyph-neighbour lists (used by the error analysis) | derived arrays; used by `c3_analysis.py` | `A0-05_glyph_neighbors/` | §5.1 |
| A1-00 | HWDB conversion and split preparation | `a1_hwdb_prepare.py` | `A1-00_prep/` | §4.1, §5.3 |
| A1-02 | HWDB-LT training (12 losses/seeds) and pre-registered verdict | `a1_train_hccr.py`, `a1_run_queue.sh`, `a1_judge.py` | `A1-02_hwdb_lt/` | §5.3 |
| A1-03 | LA + smoothing on HWDB-LT (exploratory) | `a1_train_hccr.py`, `a1_run_queue_la.sh` | `A1-03_la_plus/` | §5.3 |
| C-01 | Balanced vs natural-frequency re-evaluation, τ curves | `c1_reeval_hwdb.py`, `c1_plot.py` | `C-01_bal_vs_nat/` | §5.3, §5.4 |
| C-01b | Additional seeds for C-01 | `c1b_run_queue.sh` | `C-01b_hwdb_seeds/` | §5.3 |
| C-01n | HWDB-NAT (class counts ∝ real text) | `a1_hwdb_prepare.py natural`, `c1_reeval_hwdb.py` | `C-01n_hwdb_nat/` | §5.3, §5.4 |
| C-02 | BCTR training-side conditional LA | `c2_finetune.py`, `c2_eval.py`, `c2b_eval.py` | `C-02_bctr_la/` | §5.5 |
| C-03 | Error mechanism: direction asymmetry, β, candidate coverage | `c3_analysis.py`, `c3b_asymmetry.py` | `C-03_analysis/` | §5.1 |
| C-04 | External official SVTRv2 T/S/B models | `c4_dump_external.py`, `c4_analysis.py` | `C-04_external/` | §5.1, §5.5 |
| C-04b | External industrial OpenOCR models | `c4b_dump_industrial.py` | `C-04b_industrial/` | §5.5 |
| C-05 | CRNN trained from scratch on BCTR | `c5_train_crnn.py` | `C-05_crnn/` | §5.1, §5.5 |
| C-06 | Prior estimated from unlabelled text | `c6_selfprior.py` | `C-06_selfprior/` | §5.5 |
| C-07 | HWDB frequency flip / random order; direction test | `a1_hwdb_prepare.py flip`, `c7_pipeline.sh`, `c7_flip_direction.py`, `c7_figure.py` | `C-07_freq_flip/` | §5.2 |
| C-08 | G-LA ablations, sensitivity, annotation audit | `c8_ablation.py`, `c8_audit.py`, `c8_examples.py`, `c8_after_c5.sh` | `C-08_ablation/` | §5.3, §5.5 |
| C-09 | SMTR autoregressive step-wise correction | `c9_smtr_decode.py`, `c9_analysis.py`, `c9_pipeline.sh` | `C-09_smtr/` | §5.4, §5.5 |
| C-10 | HWDB long-tail baselines and τ-normalisation | `a1_train_hccr.py`, `c10_tnorm.py`, `c10_report.py`, `c10_pipeline.sh` | `C-10_lt_baselines/` | §5.3 |
| C-11 | CRNN trained from scratch with LA in the loss | `c11_train_crnn_la.py`, `c11_pipeline.sh`, `c11_eval.py`, `c11_after.sh` | `C-11_crnn_la/` | §5.5 |
| C-13 | Frequency swap of 54 confusable pairs in text lines | `c13_flip_lines.py`, `c13_sampler.py`, `c13_train_crnn_flip.py` | `C-13_crnn_flip/` | §5.2 |
| C-14 | Dose–response of the training-frequency profile | `c14_dose.py`, `c14_early.sh` | `C-14_dose/` | §5.2 |
| C-15 | Frequency-matched random-class smoothing control | `a1_train_hccr.py`, `c15_report.py` | `C-15_rand_smooth/` | §5.3 |
| C-16 | External baselines (ASTER, TransOCR, HWDB lines) | `c16_bctr_baselines.py`, `c16_hwdb.py`, `v4_c16.sh` | `C-16_external/` | §5.1, §5.5 |
| C-17 | Fine τ curves for CTC and SMTR | `c17_curves.py` | `C-17_tau_curves/` | §5.4 |
| V4A | Descriptive direction / self-prior counts on CPU | `v4a_direction.py`, `v4a_run.sh`, `v4a_selfprior_counts.py` | `V4A_cpu/` | §5.1, §5.5 |
| — | Paper figures | `w1_figures.py`, `v4_figures.py` | `ly/paper/figures/` (not in this repository) | — |
| — | Pipeline orchestration and restart | `m10_restart_pipeline.sh`, `m11_c5_rerun.sh`, `v4_pipeline.sh`, `c5_cpu.sh`, `c10_cpu.sh` | logs in `ly/exp/` | — |

## Notes

* Results are aggregate: metric JSON, `report.md`, model SHA-256 and small derived arrays.
  Per-sample predictions and checkpoints are not in the repository (see the README).
* The scripts are the original research code. Shell pipelines contain absolute paths from the
  authors' server (`/root/autodl-tmp/...`); edit them before running. Python scripts resolve
  `REPO = Path(__file__).parents[2]`, so keep the `ly/scripts/` layout.
* `a1_hwdb_prepare.py` reads the BCTR frequency table from
  `OpenOCR-main/output/rec/bctr_scheme51_ab_bs64/train_character_frequency.json` and falls
  back to `ly/data/bctr/train_character_frequency.json`, which is included here.
* OpenOCR-based scripts (`c4*`, `c5`/`c9`/`c11` variants, `c16*`, `v4*`) need OpenOCR and
  its Python environment; the HWDB scripts use the `numpy<2` environment described in the
  README.
