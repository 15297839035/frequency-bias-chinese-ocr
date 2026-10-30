cd /root/autodl-tmp/work
PY=/root/autodl-tmp/venv_ocr/bin/python
E=ly/exp/C-02_bctr_la
until grep -q TRAINING_DONE logs/c2_pipeline.log; do sleep 60; done
echo "C2B_START $(date +%H:%M)"
run() { gpu=$1; name=$2; tau=$3
  CUDA_VISIBLE_DEVICES=$gpu nohup $PY ly/scripts/c2_finetune.py train --tau $tau --seed 3 --epochs 10 \
    --out $E/$name --workers 8 > logs/c2/$name.log 2>&1 < /dev/null &
}
run 0 c0_s3 0; run 0 la1_s3 1.0; run 1 la025_s3 0.25
sleep 120
until [ "$(ls $E/*_s3/epoch_10.pth 2>/dev/null | wc -l)" -eq 3 ] && ! pgrep -f "[c]2_finetune.py train" > /dev/null; do
  if ! pgrep -f "[c]2_finetune.py train" > /dev/null; then echo "C2B_TRAIN_FAILED $(date +%H:%M)"; exit 1; fi
  sleep 60; done
echo "C2B_TRAINING_DONE $(date +%H:%M)"
(cd $E && for r in c0_s3 la025_s3 la1_s3; do echo "$r $(sha256sum $r/epoch_10.pth | cut -c1-12)"; done >> model_sha256_12.txt)
dump() { gpu=$1; run=$2
  for split in val test; do for domain in scene web document; do
    CUDA_VISIBLE_DEVICES=$gpu $PY ly/scripts/a0_dump_ctc.py --checkpoint $E/$run/epoch_10.pth --tag $run --split $split \
      --domain $domain --device cuda --workers 4 --threads 4 --dataset-root /root/autodl-tmp/bctr --out $E/dumps \
      > logs/c2/dump_${run}_${split}_${domain}.log 2>&1 || echo "DUMP_FAIL $run $split $domain"
  done; done; }
dump 0 c0_s3 & dump 1 la025_s3 & dump 0 la1_s3 & wait
echo "C2B_DUMPS_DONE $(date +%H:%M)"
