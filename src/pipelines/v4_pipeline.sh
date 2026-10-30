#!/bin/bash
# Plan v4 on ONE H20 (~24 h): C-13, C-14, C-15, C-17 (protocol: docs/experiments.md, "C-13 至 C-17 预注册").
# C-16 needs downloaded data and runs separately: src/pipelines/c16_external_baselines.sh (same GPU, own log).
#   cd /root/autodl-tmp/work && (nohup bash src/pipelines/v4_pipeline.sh > logs/v4_pipeline.log 2>&1 < /dev/null &)
# Idempotent: every step skips work whose outputs exist and trainings continue from latest.pth, so after a crash or an
# instance restart the same command picks up where it stopped. Markers: V4_* lines in logs/v4_pipeline.log.
# Order = priority: C-13 trainings (4 CRNN on the card) + one HWDB queue (C-14, C-15) first; C-13 dumps and analysis;
# then the extra SMTR decodes of C-17, which are cut if time runs out.
cd /root/autodl-tmp/work
PYO=/root/autodl-tmp/venv_ocr/bin/python    # OpenOCR side (numpy < 2)
PY=/root/miniconda3/bin/python              # HWDB side (as A1 / C-07 / C-10)
BCTR=/root/autodl-tmp/bctr
E5=results/C-05_crnn; E13=results/C-13_crnn_flip; E14=results/C-14_dose; E15=results/C-15_rand_smooth
E17=results/C-17_tau_curves
N=results/C-01b_hwdb_seeds/dumps_all; REV=results/C-07_freq_flip/dumps_rev
O=OpenOCR-main/output/rec
export CUDA_VISIBLE_DEVICES=0
mkdir -p logs/v4 $E13 $E14 $E15 $E17
exec 9> logs/v4_pipeline.lock  # children (the trainings) inherit the lock: no second copy while anything still runs
flock -n 9 || { echo "V4_ALREADY_RUNNING $(date '+%m-%d %H:%M')"; exit 1; }
mark() { echo "V4_$1 $(date '+%m-%d %H:%M')"; }
WORKERS=$([ "$(nproc)" -ge 24 ] && echo 12 || echo 6)   # speed only; C-05 used 12 with 32 cores
mark "START nproc=$(nproc) workers=$WORKERS $(free -g | awk '/Mem/{print "mem_free="$7"G"}') \
$(nvidia-smi --query-gpu=name,memory.used --format=csv,noheader 2>&1 | head -1)"

# ---------------------------------------------------------------- 0. preflight: inputs that must already exist
for f in $E5/dumps/crnn_s1_val_scene.jsonl $E5/crnn_s1/config.yml $E5/crnn_s1/epoch_20.pth $N/ce_s0/test.npz \
         $REV/rev_ce_s0/test.npz results/C-09_smtr/preds/gtc_test_scene.jsonl data/hwdb/train_labels.npy \
         $BCTR/scene/scene_train/data.mdb results/C-01_bal_vs_nat/weights.npz; do
  [ -e "$f" ] || { mark "PREFLIGHT_MISSING $f"; exit 1; }
done
mark "PREFLIGHT_OK"

# ---------------------------------------------------------------- 1. CPU preparation (seconds to minutes)
[ -f $E13/pairs.json ] || $PYO src/data/line_swap.py pairs --dumps $E5/dumps --out $E13 > logs/v4/c13_pairs.log 2>&1 \
  || mark "C13_PAIRS_FAIL"
[ -f $E13/weights_bal.npz ] || $PYO src/data/line_swap.py weights --out $E13 --dataset-root $BCTR \
  > logs/v4/c13_weights.log 2>&1 || mark "C13_WEIGHTS_FAIL"
$PYO src/data/line_swap.py selftest > logs/v4/c13_selftest.log 2>&1 && mark "C13_SELFTEST_OK" || mark "C13_SELFTEST_FAIL"
C13_OK=0
grep -q '"check_90pct_within_10pct": true' $E13/weights_report.json 2>/dev/null && grep -q "SELFTEST OK" logs/v4/c13_selftest.log \
  && C13_OK=1
mark "C13_PREP ok=$C13_OK"
for pair in 0.25:025 0.5:050 0.75:075; do a=${pair%:*}; s=${pair#*:}; [ -f data/hwdb/train_lt100dose${s}_indices.npy ] \
  || $PY src/data/prepare_hwdb.py dose --out data/hwdb --alpha $a >> logs/v4/c14_prep.log 2>&1 || mark "C14_PREP_FAIL $a"
  cp data/hwdb/train_lt100dose${s}_summary.json $E14/ 2>/dev/null; done
GLYPH=results/A1-00_prep/glyph_neighbors_gb2312_l1/neighbors_k50.npy
[ -f $E15/random_neighbors_k10.npy ] || $PY src/training/train_hwdb_classifier.py randnb --data data/hwdb --glyph-neighbors $GLYPH \
  --out $E15/random_neighbors_k10.npy > logs/v4/c15_prep.log 2>&1 || mark "C15_PREP_FAIL"
mark "PREP_DONE"

# ---------------------------------------------------------------- 2. C-13: four CRNN trainings on the card
c13_launch() { run=$1; arm=${run%_s*}; seed=${run#*_s}; dir=$E13/$run
  [ -f $dir/epoch_20.pth ] && return 0
  if [ -f $dir/latest.pth ]; then extra=--resume
  else [ -d $dir ] && mv $dir $dir.aborted_$(date +%m%d%H%M); extra=""; fi
  nohup $PYO src/training/train_crnn_line_swap.py train --arm $arm --seed $seed --weights $E13/weights_$arm.npz --out $dir \
    --workers $WORKERS $extra >> logs/v4/c13_$run.log 2>&1 < /dev/null &
  mark "C13_LAUNCH $run $extra"; }
if [ $C13_OK = 1 ]; then
  if ! ls $E13/rev_s1/latest.pth $E13/rev_s1/epoch_20.pth > /dev/null 2>&1; then  # first start only: 2 x 100-step smoke
    rm -rf $E13/smoke
    $PYO src/training/train_crnn_line_swap.py train --arm rev --seed 1 --weights $E13/weights_rev.npz --out $E13/smoke \
      --workers $WORKERS --epochs 2 --warmup-epoch 0.5 --smoke-steps 100 > logs/v4/c13_smoke.log 2>&1 && mark "C13_SMOKE_OK" \
      || { mark "C13_SMOKE_FAIL"; C13_OK=0; }
    rm -rf $E13/smoke
  fi
fi
if [ $C13_OK = 1 ]; then for run in rev_s1 rev_s2 bal_s1 bal_s2; do c13_launch $run; done
else mark "C13_SKIPPED (see logs/v4/c13_*.log)"; fi

# ---------------------------------------------------------------- 4. HWDB queue (one process at a time on the card)
(
  hwdb() { split=$1; loss=$2; seed=$3; dir=$4; shift 4
    [ -f $dir/result.json ] && return 0
    $PY src/training/train_hwdb_classifier.py train --data data/hwdb --split $split --loss $loss --seed $seed --out $dir "$@" \
      > $dir.log 2>&1 && mark "HWDB_RUN_DONE $dir" || mark "HWDB_RUN_FAIL $dir"; }
  for s in 0 1; do for a in 050 025 075; do hwdb lt100dose$a ce $s $E14/dose${a}_ce_s$s; done; done
  for s in 0 1 2 3; do hwdb lt100 rgls $s $E15/rgls_s$s --random-neighbors $E15/random_neighbors_k10.npy; done
  mark "HWDB_TRAINING_DONE"
  (cd $E14 && for d in dose*_ce_s?; do echo "$d $(sha256sum $d/model.pt | cut -c1-12)"; done > model_sha256_12.txt)
  (cd $E15 && for d in rgls_s?; do echo "$d $(sha256sum $d/model.pt | cut -c1-12)"; done > model_sha256_12.txt)
  for a in 025 050 075; do [ -f $E14/dumps_$a/dose${a}_ce_s1/test.npz ] || $PY src/evaluation/reevaluate_hwdb.py dump \
    --data data/hwdb --split lt100dose$a --runs $E14/dose${a}_ce_s0 $E14/dose${a}_ce_s1 --out $E14/dumps_$a \
    > logs/v4/c14_dump_$a.log 2>&1 || mark "C14_DUMP_FAIL $a"; done
  [ -f $E15/dumps/rgls_s3/test.npz ] || $PY src/evaluation/reevaluate_hwdb.py dump --data data/hwdb --split lt100 \
    --runs $E15/rgls_s0 $E15/rgls_s1 $E15/rgls_s2 $E15/rgls_s3 --out $E15/dumps > logs/v4/c15_dump.log 2>&1 || mark "C15_DUMP_FAIL"
  mark "HWDB_DUMPS_DONE"
  nice -n 10 $PYO -W ignore src/analysis/c14_dose_response.py --data data/hwdb --out $E14 \
    --cond 0=$N/ce_s0 0=$N/ce_s1 0=$N/ce_s2 0=$N/ce_s3 \
    0.25=$E14/dumps_025/dose025_ce_s0 0.25=$E14/dumps_025/dose025_ce_s1 \
    0.5=$E14/dumps_050/dose050_ce_s0 0.5=$E14/dumps_050/dose050_ce_s1 \
    0.75=$E14/dumps_075/dose075_ce_s0 0.75=$E14/dumps_075/dose075_ce_s1 \
    1=$REV/rev_ce_s0 1=$REV/rev_ce_s1 1=$REV/rev_ce_s2 1=$REV/rev_ce_s3 > logs/v4/c14_analysis.log 2>&1 \
    && mark "C14_DONE" || mark "C14_ANALYSIS_FAIL"
  nice -n 10 $PYO -W ignore src/analysis/c15_random_smoothing_report.py --dumps $E15/dumps $N --out $E15 > logs/v4/c15_report.log 2>&1 \
    && mark "C15_DONE" || mark "C15_REPORT_FAIL"

) &
HWDB_PID=$!

# ---------------------------------------------------------------- 5. CTC tau curves (A6 part of C-17; CPU, now)
( [ -f $E17/ctc/c17_results.json ] || nice -n 10 $PYO -W ignore src/analysis/c17_tau_curves.py --out $E17/ctc \
    --ctc gtc=results/C-03_analysis/dumps b=results/C-03_analysis/dumps off_T=results/C-04_external/dumps \
    off_S=results/C-04_external/dumps off_B=results/C-04_external/dumps crnn_s1=$E5/dumps crnn_s2=$E5/dumps \
    > logs/v4/c17_ctc.log 2>&1 && mark "C17_CTC_CURVES_DONE" || mark "C17_CTC_CURVES_FAIL" ) &

# ---------------------------------------------------------------- 6. C-13: wait, dump, analyse
if [ $C13_OK = 1 ]; then
  until [ "$(ls $E13/{rev,bal}_s{1,2}/epoch_20.pth 2>/dev/null | wc -l)" -eq 4 ]; do
    if ! ps aux | grep "[c]13_train_crnn_flip.py" | grep -qv "bash -c"; then mark "C13_TRAIN_STOPPED (re-run the pipeline)"; break; fi
    sleep 300
  done
  if [ "$(ls $E13/{rev,bal}_s{1,2}/epoch_20.pth 2>/dev/null | wc -l)" -eq 4 ]; then
    sleep 60
    mark "C13_TRAINING_DONE"
    (cd $E13 && for r in rev_s1 rev_s2 bal_s1 bal_s2; do echo "$r $(sha256sum $r/epoch_20.pth | cut -c1-12)"; done > model_sha256_12.txt)
    for r in rev_s1 rev_s2 bal_s1 bal_s2; do for s in val test; do for d in scene web document; do
      [ -f $E13/dumps/${r}_${s}_$d.meta.json ] || $PYO -W ignore src/inference/dump_external_models.py --config $E13/$r/config.yml \
        --checkpoint $E13/$r/epoch_20.pth --tag $r --split $s --domain $d --out $E13/dumps --trusted \
        > logs/v4/c13_dump_${r}_${s}_$d.log 2>&1 || mark "C13_DUMP_FAIL $r $s $d"
    done; done; done
    mark "C13_DUMPS_DONE"
    nice -n 5 $PYO -W ignore src/data/line_swap.py analyze --nat-dumps $E5/dumps --dumps $E13/dumps --out $E13 \
      > logs/v4/c13_analysis.log 2>&1 && mark "C13_DONE" || mark "C13_ANALYSIS_FAIL"
  fi
fi

# ---------------------------------------------------------------- 8. C-17: extra SMTR decodes (lowest priority)
declare -A CFG=( [gtc]=$O/bctr_scheme51_bs64/svtrv2_tiny_51_gtc_seed48
                 [b]=$O/bctr_scheme51_ab_bs64/svtrv2_tiny_51_ucr_b_mixedfreq_seed48
                 [off_T]=results/C-04_external/official/T [off_S]=results/C-04_external/official/S
                 [off_B]=results/C-04_external/official/B )
ckpt() { [ -f ${CFG[$1]}/best.pth ] && echo ${CFG[$1]}/best.pth || echo ${CFG[$1]}/epoch_100.pth; }
for s in test val; do for tag in gtc off_B off_T off_S b; do for d in scene web document; do
  [ -f $E17/smtr_preds/${tag}_${s}_$d.meta.json ] || $PYO -W ignore src/inference/decode_smtr.py --config ${CFG[$tag]}/config.yml \
    --checkpoint $(ckpt $tag) --tag $tag --split $s --domain $d --out $E17/smtr_preds \
    --grid 0.05:1.01 0.15:1.01 0.25:1.01 0.4:1.01 0.75:1.01 1.25:1.01 1.5:1.01 > logs/v4/c17_${tag}_${s}_$d.log 2>&1 \
    || mark "C17_DECODE_FAIL $tag $s $d"
done; done; mark "C17_DECODES_DONE $s"; done
nice -n 5 $PYO -W ignore src/analysis/c17_tau_curves.py --out $E17/smtr \
  --smtr gtc=results/C-09_smtr/preds,$E17/smtr_preds b=results/C-09_smtr/preds,$E17/smtr_preds \
  off_T=results/C-09_smtr/preds,$E17/smtr_preds off_S=results/C-09_smtr/preds,$E17/smtr_preds \
  off_B=results/C-09_smtr/preds,$E17/smtr_preds > logs/v4/c17_smtr.log 2>&1 && mark "C17_DONE" || mark "C17_SMTR_CURVES_FAIL"

wait $HWDB_PID
wait
mark "ALL_DONE"
