cd /root/autodl-tmp/work
PY=/root/autodl-tmp/venv_ocr/bin/python
E=ly/exp/C-02_bctr_la
until [ "$(ls $E/*/epoch_10.pth 2>/dev/null | wc -l)" -eq 6 ] && ! pgrep -f "[c]2_finetune.py train" > /dev/null; do sleep 60; done
echo "TRAINING_DONE $(date +%H:%M)"
grep -l Traceback logs/c2/*.log && echo "TRAIN_TRACEBACK_FOUND"
(cd $E && for r in c0_s1 c0_s2 la025_s1 la025_s2 la1_s1 la1_s2; do echo "$r $(sha256sum $r/epoch_10.pth | cut -c1-12)"; done > model_sha256_12.txt)
dump() { gpu=$1; run=$2
  for split in val test; do for domain in scene web document; do
    CUDA_VISIBLE_DEVICES=$gpu $PY ly/scripts/a0_dump_ctc.py --checkpoint $E/$run/epoch_10.pth --tag $run --split $split \
      --domain $domain --device cuda --workers 4 --threads 4 --dataset-root /root/autodl-tmp/bctr --out $E/dumps \
      > logs/c2/dump_${run}_${split}_${domain}.log 2>&1 || echo "DUMP_FAIL $run $split $domain"
  done; done; }
(dump 0 c0_s1; dump 0 la1_s1) & (dump 0 la025_s1) & (dump 1 c0_s2; dump 1 la1_s2) & (dump 1 la025_s2) & wait
echo "DUMPS_DONE $(date +%H:%M)"
until grep -q C3_DUMPS_DONE logs/c3_dumps.log; do sleep 60; done
$PY ly/scripts/c2_eval.py --dumps $E/dumps --ref-dumps ly/exp/C-03_analysis/dumps --out $E > logs/c2_eval.log 2>&1 \
  && echo "EVAL_DONE $(date +%H:%M)" || echo "EVAL_FAIL"
