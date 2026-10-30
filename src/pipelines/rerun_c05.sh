#!/bin/bash
# M-11 (server, 2026-10-13 22:20): rerun the C-05 downstream steps of m10_restart_pipeline.sh that stopped on the baseline
# sanity check (crnn_s2 test scene: 1 of 127k lines differs between the dump's top-1 and the model's argmax, an exact
# probability tie). The checks now allow a few such ties (core/prior_correction.py (check_baseline)). Same commands as before.
# The strict C-08 pass started by m10_restart_pipeline.sh (old code, will stop at crnn_s2) is left to finish; this script
# waits for it before starting its own strict pass.
#   cd /root/autodl-tmp/work && (nohup bash src/pipelines/rerun_c05.sh > logs/m11_c5_rerun.log 2>&1 < /dev/null &)
cd /root/autodl-tmp/work
PYO=/root/autodl-tmp/venv_ocr/bin/python
E5=results/C-05_crnn
for norm in strict official; do
  ( $PYO src/analysis/c04_external_analysis.py --dumps $E5/dumps --out $E5 --norm $norm --tags crnn_s1 crnn_s2 \
      > logs/c5/analysis_$norm.log 2>&1 && echo "C5X_${norm}_DONE $(date +%H:%M)" || echo "C5X_${norm}_FAILED" ) &
done
( $PYO -W ignore src/analysis/c08_correction_ablation.py --out results/C-08_ablation/crnn --norm official --workers 2 --tags crnn_s1 crnn_s2 \
    > logs/c8/official_crnn_rerun.log 2>&1 && echo "C5X_C8_DONE $(date +%H:%M)" || echo "C5X_C8_FAILED" ) &
wait
$PYO -W ignore src/analysis/c06_self_prior.py > logs/c6_with_crnn.log 2>&1 && echo "C5X_C6_DONE $(date +%H:%M)" || echo "C5X_C6_FAILED"
$PYO -W ignore src/figures/paper_figures.py --out paper/figures > logs/figs_with_crnn.log 2>&1 \
  && echo "C5X_FIGS_DONE $(date +%H:%M)" || echo "C5X_FIGS_FAILED"
while grep -q "c08_correction_ablation.py --out results/C-08_ablation --norm strict" <(ps -eo args | grep "^/root/autodl-tmp/venv_ocr"); do
  sleep 60
done
$PYO -W ignore src/analysis/c08_correction_ablation.py --out results/C-08_ablation --norm strict --workers 5 \
  > logs/c8/strict_rerun.log 2>&1 && echo "C5X_C8_STRICT_DONE $(date +%H:%M)" || echo "C5X_C8_STRICT_FAILED"
echo "C5X_ALL_DONE $(date +%H:%M)"
