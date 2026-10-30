cd /root/autodl-tmp/work
until grep -qE "C2B_DUMPS_DONE|FAILED" logs/c2b_pipeline.log; do sleep 60; done
grep -qE "FAILED|DUMP_FAIL" logs/c2b_pipeline.log && { echo C2B_EVAL_ABORT; exit 1; }
/root/autodl-tmp/venv_ocr/bin/python ly/scripts/c2b_eval.py --dumps ly/exp/C-02_bctr_la/dumps --out ly/exp/C-02_bctr_la > logs/c2b_eval.log 2>&1 \
  && echo "C2B_EVAL_DONE $(date +%H:%M)" || echo C2B_EVAL_FAILED
