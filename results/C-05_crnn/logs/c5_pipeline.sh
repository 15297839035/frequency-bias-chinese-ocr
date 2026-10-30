cd /root/autodl-tmp/work
PY=/root/autodl-tmp/venv_ocr/bin/python
E=ly/exp/C-05_crnn
mkdir -p $E logs/c5
launch() { gpu=$1; seed=$2
  CUDA_VISIBLE_DEVICES=$gpu nohup $PY ly/scripts/c5_train_crnn.py train --seed $seed --out $E/crnn_s$seed --workers 12 \
    > logs/c5/crnn_s$seed.log 2>&1 < /dev/null &
  echo "C5_LAUNCH crnn_s$seed gpu$gpu $(date +%H:%M)"
}
s1=0; s2=0
while [ $s1 -eq 0 ] || [ $s2 -eq 0 ]; do
  if [ $s1 -eq 0 ] && grep -qE "C1N_TRAINING_DONE|C1N_.*FAIL" logs/c1n_pipeline.log 2>/dev/null; then launch 1 1; s1=1; fi
  if [ $s2 -eq 0 ] && grep -qE "C2B_TRAINING_DONE|FAILED" logs/c2b_pipeline.log 2>/dev/null; then launch 0 2; s2=1; fi
  sleep 60
done
sleep 300
until [ "$(ls $E/crnn_s1/epoch_20.pth $E/crnn_s2/epoch_20.pth 2>/dev/null | wc -l)" -eq 2 ]; do
  if ! ps aux | grep "[c]5_train_crnn.py train" | grep -qv "bash -c"; then echo "C5_TRAIN_FAILED $(date +%H:%M)"; exit 1; fi
  sleep 120; done
sleep 60
echo "C5_TRAINING_DONE $(date +%H:%M)"
(cd $E && for r in crnn_s1 crnn_s2; do echo "$r $(sha256sum $r/epoch_20.pth | cut -c1-12)"; done > model_sha256_12.txt)
dump() { gpu=$1; run=$2
  for split in val test; do for domain in scene web document; do
    CUDA_VISIBLE_DEVICES=$gpu $PY ly/scripts/c4_dump_external.py --config $E/$run/config.yml --checkpoint $E/$run/epoch_20.pth \
      --tag $run --split $split --domain $domain --out $E/dumps --trusted > logs/c5/dump_${run}_${split}_${domain}.log 2>&1 \
      || echo "C5_DUMP_FAIL $run $split $domain"
  done; done; }
dump 0 crnn_s1 & dump 1 crnn_s2 & wait
echo "C5_DUMPS_DONE $(date +%H:%M)"
for norm in strict official; do
  $PY ly/scripts/c4_analysis.py --dumps $E/dumps --out $E --norm $norm --tags crnn_s1 crnn_s2 > logs/c5/analysis_$norm.log 2>&1 \
    && echo "C5_${norm}_DONE $(date +%H:%M)" || echo "C5_${norm}_FAILED"
done
