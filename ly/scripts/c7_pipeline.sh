#!/bin/bash
# C-07 (server): flipped HWDB-LT splits -> ce training (one queue per GPU) -> prediction dumps -> direction analysis.
#   cd /root/autodl-tmp/work && (nohup bash ly/scripts/c7_pipeline.sh > logs/c7_pipeline.log 2>&1 < /dev/null &)
cd /root/autodl-tmp/work
PY=/root/miniconda3/bin/python
PYO=/root/autodl-tmp/venv_ocr/bin/python
E=ly/exp/C-07_freq_flip
mkdir -p $E logs/c7
for order in rev perm; do
  [ -f ly/data/hwdb/train_lt100${order}_indices.npy ] || $PY ly/scripts/a1_hwdb_prepare.py flip --out ly/data/hwdb --order $order
  cp ly/data/hwdb/train_lt100${order}_summary.json $E/
done
queue() { gpu=$1; shift
  for job in "$@"; do order=${job%%:*}; seed=${job##*:}; dir=$E/${order}_ce_s$seed
    [ -f $dir/result.json ] && continue
    CUDA_VISIBLE_DEVICES=$gpu $PY ly/scripts/a1_train_hccr.py train --data ly/data/hwdb --split lt100$order --loss ce \
      --seed $seed --out $dir > $dir.log 2>&1 && echo "C7_RUN_DONE $job $(date +%H:%M)" || echo "C7_TRAIN_FAIL $job"
  done; }
queue 0 rev:0 rev:2 perm:0 &
queue 1 rev:1 rev:3 perm:1 &
wait
echo "C7_TRAINING_DONE $(date +%H:%M)"
(cd $E && for r in rev_ce_s0 rev_ce_s1 rev_ce_s2 rev_ce_s3 perm_ce_s0 perm_ce_s1; do
  echo "$r $(sha256sum $r/model.pt | cut -c1-12)"; done > model_sha256_12.txt)
CUDA_VISIBLE_DEVICES=0 $PY ly/scripts/c1_reeval_hwdb.py dump --data ly/data/hwdb --split lt100rev \
  --runs $E/rev_ce_s0 $E/rev_ce_s1 $E/rev_ce_s2 $E/rev_ce_s3 --out $E/dumps_rev > logs/c7/dump_rev.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 $PY ly/scripts/c1_reeval_hwdb.py dump --data ly/data/hwdb --split lt100perm \
  --runs $E/perm_ce_s0 $E/perm_ce_s1 --out $E/dumps_perm > logs/c7/dump_perm.log 2>&1 &
wait
echo "C7_DUMPS_DONE $(date +%H:%M)"
N=ly/exp/C-01b_hwdb_seeds/dumps_all
$PYO -W ignore ly/scripts/c7_flip_direction.py --data ly/data/hwdb --nat $N/ce_s0 $N/ce_s1 $N/ce_s2 $N/ce_s3 \
  --rev $E/dumps_rev/rev_ce_s0 $E/dumps_rev/rev_ce_s1 $E/dumps_rev/rev_ce_s2 $E/dumps_rev/rev_ce_s3 \
  --perm $E/dumps_perm/perm_ce_s0 $E/dumps_perm/perm_ce_s1 --out $E > logs/c7/analysis.log 2>&1 \
  && echo "C7_ANALYSIS_DONE $(date +%H:%M)" || echo "C7_ANALYSIS_FAILED"
for d in rev perm; do
  $PY ly/scripts/c1_reeval_hwdb.py analyze --dumps $E/dumps_$d --out $E/curves_$d --skip-judge > logs/c7/curves_$d.log 2>&1
done
echo "C7_ALL_DONE $(date +%H:%M)"
