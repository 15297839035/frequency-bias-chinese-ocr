#!/bin/bash
# After c11_pipeline.sh has dumped all four runs (and C-05 has its dumps): C-11 evaluation.
cd /root/autodl-tmp/work
until grep -qE "C11_DUMPS_DONE|C11_TRAIN_FAILED" logs/c11_pipeline.log 2>/dev/null; do sleep 300; done
grep -q "C11_TRAIN_FAILED" logs/c11_pipeline.log && { echo "C11_EVAL_SKIPPED"; exit 1; }
until grep -qE "C5_DUMPS_DONE" logs/c5_pipeline.log 2>/dev/null; do sleep 300; done
/root/autodl-tmp/venv_ocr/bin/python -W ignore src/evaluation/evaluate_crnn_la.py --out results/C-11_crnn_la > logs/c11/eval.log 2>&1 \
  && echo "C11_EVAL_DONE $(date +%H:%M)" || echo "C11_EVAL_FAILED"
