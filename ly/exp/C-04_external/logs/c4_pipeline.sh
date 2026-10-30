cd /root/autodl-tmp/work
PY=/root/autodl-tmp/venv_ocr/bin/python
until grep -q C4_DUMPS_DONE logs/c4_dumps.log; do sleep 60; done
grep -q C4_DUMP_FAIL logs/c4_dumps.log && { echo C4_PIPELINE_ABORT_DUMP_FAIL; exit 1; }
$PY ly/scripts/c4_analysis.py --dumps ly/exp/C-04_external/dumps --out ly/exp/C-04_external --norm official > logs/c4_analysis_official.log 2>&1 \
  && echo "C4_OFFICIAL_DONE $(date +%H:%M)" || echo C4_OFFICIAL_FAILED
$PY ly/scripts/c4_analysis.py --dumps ly/exp/C-04_external/dumps --out ly/exp/C-04_external --norm strict > logs/c4_analysis_strict.log 2>&1 \
  && echo "C4_STRICT_DONE $(date +%H:%M)" || echo C4_STRICT_FAILED
