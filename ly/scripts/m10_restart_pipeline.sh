#!/bin/bash
# M-10 (server, 2026-10-13): after the instance restart (GPU access lost at about 15:40), resume / relaunch every
# interrupted GPU job and the downstream steps that the killed pipelines (c5_pipeline, after_c5, c8_after_c5,
# c10_pipeline, c11_pipeline, c11_after) would have run. Same scripts and settings as before; trainings continue
# from their last whole epoch (latest.pth; the data order of each epoch is seeded by (seed, epoch)).
#   cd /root/autodl-tmp/work && (nohup bash ly/scripts/m10_restart_pipeline.sh > logs/m10_restart.log 2>&1 < /dev/null &)
cd /root/autodl-tmp/work
PYO=/root/autodl-tmp/venv_ocr/bin/python
PY=/root/miniconda3/bin/python
E5=ly/exp/C-05_crnn; E10=ly/exp/C-10_lt_baselines; E11=ly/exp/C-11_crnn_la
mkdir -p logs/c5 logs/c10 logs/c11 logs/c8

train() { gpu=$1; log=$2; shift 2
  CUDA_VISIBLE_DEVICES=$gpu nohup $PYO "$@" --workers 12 >> $log 2>&1 < /dev/null &
}
train 1 logs/c5/crnn_s1.log ly/scripts/c5_train_crnn.py train --seed 1 --out $E5/crnn_s1 --resume
train 0 logs/c5/crnn_s2.log ly/scripts/c5_train_crnn.py train --seed 2 --out $E5/crnn_s2 --resume
train 0 logs/c11/la025_s1.log ly/scripts/c11_train_crnn_la.py train --tau 0.25 --seed 1 --out $E11/la025_s1 --resume
train 1 logs/c11/la1_s1.log ly/scripts/c11_train_crnn_la.py train --tau 1 --seed 1 --out $E11/la1_s1 --resume
train 0 logs/c11/la025_s2.log ly/scripts/c11_train_crnn_la.py train --tau 0.25 --seed 2 --out $E11/la025_s2
train 1 logs/c11/la1_s2.log ly/scripts/c11_train_crnn_la.py train --tau 1 --seed 2 --out $E11/la1_s2
echo "M10_TRAININGS_LAUNCHED $(date +%H:%M)"

wait_runs() {  # wait_runs <pattern for ps> <file>...: until all files exist; fail if no training process is left
  pat=$1; shift
  until [ "$(ls "$@" 2>/dev/null | wc -l)" -eq $# ]; do
    if ! ps aux | grep "[${pat:0:1}]${pat:1}" | grep -qv "bash -c"; then return 1; fi
    sleep 120
  done
  sleep 60
}

(  # ---- C-10: HWDB-LT dumps now, HWDB-NAT trainings, then analysis and report
  CUDA_VISIBLE_DEVICES=0 $PY ly/scripts/c1_reeval_hwdb.py dump --data ly/data/hwdb --split lt100 --runs $E10/lt100/*_s? \
    --out $E10/dumps_lt100 > logs/c10/dump_lt100.log 2>&1 || echo "C10R_DUMP_FAIL lt100"
  queue() { gpu=$1; shift
    for job in "$@"; do split=${job%%:*}; rest=${job#*:}; loss=${rest%%:*}; seed=${rest##*:}; dir=$E10/$split/${loss}_s$seed
      [ -f $dir/result.json ] && continue
      CUDA_VISIBLE_DEVICES=$gpu $PY ly/scripts/a1_train_hccr.py train --data ly/data/hwdb --split $split --loss $loss \
        --seed $seed --out $dir > $dir.log 2>&1 && echo "C10R_RUN_DONE $job $(date +%H:%M)" || echo "C10R_TRAIN_FAIL $job"
    done; }
  queue 0 nat:cb:0 nat:focal:0 nat:rs_sqrt:0 nat:rad:0 &
  queue 1 nat:cb:1 nat:focal:1 nat:rs_sqrt:1 nat:rad:1 &
  wait
  echo "C10R_TRAINING_DONE $(date +%H:%M)"
  (cd $E10 && for d in lt100/*_s? nat/*_s?; do echo "$d $(sha256sum $d/model.pt | cut -c1-12)"; done > model_sha256_12.txt)
  CUDA_VISIBLE_DEVICES=1 $PY ly/scripts/c1_reeval_hwdb.py dump --data ly/data/hwdb --split nat --runs $E10/nat/*_s? \
    --out $E10/dumps_nat > logs/c10/dump_nat.log 2>&1 || echo "C10R_DUMP_FAIL nat"
  for split in lt100 nat; do
    $PY ly/scripts/c1_reeval_hwdb.py analyze --dumps $E10/dumps_$split --out $E10/analysis_$split --skip-judge \
      > logs/c10/analyze_$split.log 2>&1 || echo "C10R_ANALYZE_FAIL $split"
  done
  $PYO ly/scripts/c10_report.py --out $E10 > logs/c10/report.log 2>&1 && echo "C10R_REPORT_DONE $(date +%H:%M)" \
    || echo "C10R_REPORT_FAIL"
) &

(  # ---- C-05: dumps, analysis, C-06, C-08 (CRNN + strict pass), figures
  wait_runs c5_train_crnn.py $E5/crnn_s1/epoch_20.pth $E5/crnn_s2/epoch_20.pth || { echo "C5R_TRAIN_FAILED $(date +%H:%M)"; exit 1; }
  echo "C5R_TRAINING_DONE $(date +%H:%M)"
  (cd $E5 && for r in crnn_s1 crnn_s2; do echo "$r $(sha256sum $r/epoch_20.pth | cut -c1-12)"; done > model_sha256_12.txt)
  dump() { gpu=$1; run=$2
    for split in val test; do for domain in scene web document; do
      CUDA_VISIBLE_DEVICES=$gpu $PYO ly/scripts/c4_dump_external.py --config $E5/$run/config.yml --checkpoint $E5/$run/epoch_20.pth \
        --tag $run --split $split --domain $domain --out $E5/dumps --trusted > logs/c5/dump_${run}_${split}_${domain}.log 2>&1 \
        || echo "C5R_DUMP_FAIL $run $split $domain"
    done; done; }
  dump 0 crnn_s1 & dump 1 crnn_s2 & wait
  echo "C5R_DUMPS_DONE $(date +%H:%M)"
  for norm in strict official; do
    $PYO ly/scripts/c4_analysis.py --dumps $E5/dumps --out $E5 --norm $norm --tags crnn_s1 crnn_s2 \
      > logs/c5/analysis_$norm.log 2>&1 && echo "C5R_${norm}_DONE $(date +%H:%M)" || echo "C5R_${norm}_FAILED"
  done
  $PYO -W ignore ly/scripts/c6_selfprior.py > logs/c6_with_crnn.log 2>&1 && echo "C5R_C6_DONE $(date +%H:%M)" || echo "C5R_C6_FAILED"
  $PYO -W ignore ly/scripts/c8_ablation.py --out ly/exp/C-08_ablation/crnn --norm official --workers 2 --tags crnn_s1 crnn_s2 \
    > logs/c8/official_crnn.log 2>&1 && echo "C5R_C8_DONE $(date +%H:%M)" || echo "C5R_C8_FAILED"
  $PYO -W ignore ly/scripts/w1_figures.py --out ly/paper/figures > logs/figs_with_crnn.log 2>&1 \
    && echo "C5R_FIGS_DONE $(date +%H:%M)" || echo "C5R_FIGS_FAILED"
  $PYO -W ignore ly/scripts/c8_ablation.py --out ly/exp/C-08_ablation --norm strict --workers 5 \
    > logs/c8/strict.log 2>&1 && echo "C5R_C8_STRICT_DONE $(date +%H:%M)" || echo "C5R_C8_STRICT_FAILED"
) &

(  # ---- C-11: dumps and evaluation once all four runs (and C-05's dumps) exist
  wait_runs c11_train_crnn_la.py $E11/la025_s1/epoch_20.pth $E11/la1_s1/epoch_20.pth $E11/la025_s2/epoch_20.pth \
    $E11/la1_s2/epoch_20.pth || { echo "C11R_TRAIN_FAILED $(date +%H:%M)"; exit 1; }
  echo "C11R_TRAINING_DONE $(date +%H:%M)"
  (cd $E11 && for r in la025_s1 la1_s1 la025_s2 la1_s2; do echo "$r $(sha256sum $r/epoch_20.pth | cut -c1-12)"; done > model_sha256_12.txt)
  dump() { gpu=$1; shift
    for run in "$@"; do for split in val test; do for domain in scene web document; do
      CUDA_VISIBLE_DEVICES=$gpu $PYO ly/scripts/c4_dump_external.py --config $E11/$run/config.yml --checkpoint $E11/$run/epoch_20.pth \
        --tag $run --split $split --domain $domain --out $E11/dumps --trusted > logs/c11/dump_${run}_${split}_${domain}.log 2>&1 \
        || echo "C11R_DUMP_FAIL $run $split $domain"
    done; done; done; }
  dump 0 la025_s1 la025_s2 & dump 1 la1_s1 la1_s2 & wait
  echo "C11R_DUMPS_DONE $(date +%H:%M)"
  until grep -q "C5R_DUMPS_DONE" logs/m10_restart.log 2>/dev/null; do sleep 300; done
  $PYO -W ignore ly/scripts/c11_eval.py --out $E11 > logs/c11/eval.log 2>&1 && echo "C11R_EVAL_DONE $(date +%H:%M)" \
    || echo "C11R_EVAL_FAILED"
) &
wait
echo "M10_ALL_DONE $(date +%H:%M)"
