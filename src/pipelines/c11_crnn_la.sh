#!/bin/bash
# C-11 (server): CRNN from scratch with training-side conditional LA. Seed 1 starts at once (GPU0 / GPU1);
# seed 2 starts when C-05 training has finished; then dumps (val / test, 3 domains) as in C-05.
#   cd /root/autodl-tmp/work && (nohup bash src/pipelines/c11_crnn_la.sh > logs/c11_pipeline.log 2>&1 < /dev/null &)
cd /root/autodl-tmp/work
PY=/root/autodl-tmp/venv_ocr/bin/python
E=results/C-11_crnn_la
mkdir -p $E logs/c11
launch() { gpu=$1; tau=$2; tag=$3
  CUDA_VISIBLE_DEVICES=$gpu nohup $PY src/training/train_crnn_la.py train --tau $tau --seed ${tag##*_s} --out $E/$tag \
    --workers 12 > logs/c11/$tag.log 2>&1 < /dev/null &
  echo "C11_LAUNCH $tag tau=$tau gpu$gpu $(date +%H:%M)"
}
launch 0 0.25 la025_s1
launch 1 1 la1_s1
until grep -qE "C5_TRAINING_DONE|C5_TRAIN_FAILED" logs/c5_pipeline.log 2>/dev/null; do sleep 120; done
launch 0 0.25 la025_s2
launch 1 1 la1_s2
sleep 300
until [ "$(ls $E/*/epoch_20.pth 2>/dev/null | wc -l)" -eq 4 ]; do
  if ! ps aux | grep "[c]11_train_crnn_la.py train" | grep -qv "bash -c"; then echo "C11_TRAIN_FAILED $(date +%H:%M)"; exit 1; fi
  sleep 120; done
sleep 60
echo "C11_TRAINING_DONE $(date +%H:%M)"
(cd $E && for r in la025_s1 la1_s1 la025_s2 la1_s2; do echo "$r $(sha256sum $r/epoch_20.pth | cut -c1-12)"; done > model_sha256_12.txt)
dump() { gpu=$1; shift
  for run in "$@"; do for split in val test; do for domain in scene web document; do
    CUDA_VISIBLE_DEVICES=$gpu $PY src/inference/dump_external_models.py --config $E/$run/config.yml --checkpoint $E/$run/epoch_20.pth \
      --tag $run --split $split --domain $domain --out $E/dumps --trusted > logs/c11/dump_${run}_${split}_${domain}.log 2>&1 \
      || echo "C11_DUMP_FAIL $run $split $domain"
  done; done; done; }
dump 0 la025_s1 la025_s2 & dump 1 la1_s1 la1_s2 & wait
echo "C11_DUMPS_DONE $(date +%H:%M)"
