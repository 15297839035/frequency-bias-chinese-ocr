#!/bin/bash
# After C-05's dumps: C-08 for the two CRNN runs (official), then the whole strict-normalisation pass.
cd /root/autodl-tmp/work
until grep -qE "C5_official_DONE|C5_.*FAIL" logs/c5_pipeline.log 2>/dev/null; do sleep 300; done
PY=/root/autodl-tmp/venv_ocr/bin/python
$PY -W ignore src/analysis/c08_correction_ablation.py --out results/C-08_ablation/crnn --norm official --workers 2 --tags crnn_s1 crnn_s2 \
  > logs/c8/official_crnn.log 2>&1 && echo "C8_CRNN_DONE $(date +%H:%M)" || echo "C8_CRNN_FAILED"
$PY -W ignore src/analysis/c08_correction_ablation.py --out results/C-08_ablation --norm strict --workers 5 \
  > logs/c8/strict.log 2>&1 && echo "C8_STRICT_DONE $(date +%H:%M)" || echo "C8_STRICT_FAILED"
