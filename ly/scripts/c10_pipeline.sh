#!/bin/bash
# C-10 (server): tau-norm on the existing ce models at once; the 16 training runs start after the C-07 training queue.
#   cd /root/autodl-tmp/work && (nohup bash ly/scripts/c10_pipeline.sh > logs/c10_pipeline.log 2>&1 < /dev/null &)
cd /root/autodl-tmp/work
PY=/root/miniconda3/bin/python
PYO=/root/autodl-tmp/venv_ocr/bin/python
E=ly/exp/C-10_lt_baselines
mkdir -p $E/lt100 $E/nat logs/c10
CUDA_VISIBLE_DEVICES=1 $PY ly/scripts/c10_tnorm.py --data ly/data/hwdb --split lt100 --runs ly/exp/A1-02_hwdb_lt/ce_s0 \
  ly/exp/A1-02_hwdb_lt/ce_s1 ly/exp/C-01b_hwdb_seeds/ce_s2 ly/exp/C-01b_hwdb_seeds/ce_s3 --out $E/tnorm_lt100.json \
  > logs/c10/tnorm_lt100.log 2>&1 && echo "C10_TNORM_LT_DONE $(date +%H:%M)" || echo "C10_TNORM_LT_FAIL"
CUDA_VISIBLE_DEVICES=1 $PY ly/scripts/c10_tnorm.py --data ly/data/hwdb --split nat --runs ly/exp/C-01n_hwdb_nat/ce_s0 \
  ly/exp/C-01n_hwdb_nat/ce_s1 ly/exp/C-01n_hwdb_nat/ce_s2 ly/exp/C-01n_hwdb_nat/ce_s3 --out $E/tnorm_nat.json \
  > logs/c10/tnorm_nat.log 2>&1 && echo "C10_TNORM_NAT_DONE $(date +%H:%M)" || echo "C10_TNORM_NAT_FAIL"
until grep -q "C7_TRAINING_DONE" logs/c7_pipeline.log 2>/dev/null; do sleep 120; done
queue() { gpu=$1; shift
  for job in "$@"; do split=${job%%:*}; rest=${job#*:}; loss=${rest%%:*}; seed=${rest##*:}; dir=$E/$split/${loss}_s$seed
    [ -f $dir/result.json ] && continue
    CUDA_VISIBLE_DEVICES=$gpu $PY ly/scripts/a1_train_hccr.py train --data ly/data/hwdb --split $split --loss $loss \
      --seed $seed --out $dir > $dir.log 2>&1 && echo "C10_RUN_DONE $job $(date +%H:%M)" || echo "C10_TRAIN_FAIL $job"
  done; }
queue 0 lt100:cb:0 lt100:focal:0 lt100:rs_sqrt:0 lt100:rad:0 nat:cb:0 nat:focal:0 nat:rs_sqrt:0 nat:rad:0 &
queue 1 lt100:cb:1 lt100:focal:1 lt100:rs_sqrt:1 lt100:rad:1 nat:cb:1 nat:focal:1 nat:rs_sqrt:1 nat:rad:1 &
wait
echo "C10_TRAINING_DONE $(date +%H:%M)"
(cd $E && for d in lt100/*_s? nat/*_s?; do echo "$d $(sha256sum $d/model.pt | cut -c1-12)"; done > model_sha256_12.txt)
for split in lt100 nat; do
  CUDA_VISIBLE_DEVICES=0 $PY ly/scripts/c1_reeval_hwdb.py dump --data ly/data/hwdb --split $split --runs $E/$split/*_s? \
    --out $E/dumps_$split > logs/c10/dump_$split.log 2>&1 || echo "C10_DUMP_FAIL $split"
  $PY ly/scripts/c1_reeval_hwdb.py analyze --dumps $E/dumps_$split --out $E/analysis_$split --skip-judge \
    > logs/c10/analyze_$split.log 2>&1 || echo "C10_ANALYZE_FAIL $split"
done
echo "C10_DUMPS_DONE $(date +%H:%M)"
$PYO ly/scripts/c10_report.py --out $E > logs/c10/report.log 2>&1 && echo "C10_REPORT_DONE $(date +%H:%M)" || echo "C10_REPORT_FAIL"
