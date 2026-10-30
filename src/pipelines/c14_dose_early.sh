#!/bin/bash
# C-14 dumps + analysis ahead of the pipeline (same commands as v4_pipeline.sh step 4; the pipeline skips existing dumps
# and reruns the analysis with identical inputs). Run once all six dose models have result.json.
cd /root/autodl-tmp/work
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
PY=/root/miniconda3/bin/python; PYO=/root/autodl-tmp/venv_ocr/bin/python
E14=results/C-14_dose; N=results/C-01b_hwdb_seeds/dumps_all; REV=results/C-07_freq_flip/dumps_rev
for a in 025 050 075; do [ -f $E14/dumps_$a/dose${a}_ce_s1/test.npz ] || $PY src/evaluation/reevaluate_hwdb.py dump \
  --data data/hwdb --split lt100dose$a --runs $E14/dose${a}_ce_s0 $E14/dose${a}_ce_s1 --out $E14/dumps_$a \
  > logs/v4/c14_dump_$a.log 2>&1 || echo "C14_EARLY_DUMP_FAIL $a"; done
nice -n 10 $PYO -W ignore src/analysis/c14_dose_response.py --data data/hwdb --out $E14 \
  --cond 0=$N/ce_s0 0=$N/ce_s1 0=$N/ce_s2 0=$N/ce_s3 \
  0.25=$E14/dumps_025/dose025_ce_s0 0.25=$E14/dumps_025/dose025_ce_s1 \
  0.5=$E14/dumps_050/dose050_ce_s0 0.5=$E14/dumps_050/dose050_ce_s1 \
  0.75=$E14/dumps_075/dose075_ce_s0 0.75=$E14/dumps_075/dose075_ce_s1 \
  1=$REV/rev_ce_s0 1=$REV/rev_ce_s1 1=$REV/rev_ce_s2 1=$REV/rev_ce_s3 > logs/v4/c14_analysis_early.log 2>&1 \
  && echo "C14_EARLY_DONE $(date '+%m-%d %H:%M')" || echo "C14_EARLY_FAIL $(date '+%m-%d %H:%M')"
