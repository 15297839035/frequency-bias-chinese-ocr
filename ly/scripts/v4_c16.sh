#!/bin/bash
# C-16 (protocol: ly/commit.md, "C-13 至 C-17 预注册"), run next to v4_pipeline.sh once the downloads are on the server:
#   $BCTR/hwdb/hwdb_val, $BCTR/hwdb/hwdb_test          <- BCTR "hwdb_ic13": handwriting_hwdb_val, handwriting_ic13_test
#   $BASE/{transocr,aster}_{scene,web,document}.pth[.tar] <- BCTR base checkpoints (ly/third_party/bctr_baselines/SOURCE.md)
#   cd /root/autodl-tmp/work && (nohup bash ly/scripts/v4_c16.sh > logs/v4_c16.log 2>&1 < /dev/null &)
# Idempotent (finished files are skipped); parts whose inputs are missing are skipped with a marker.
cd /root/autodl-tmp/work
PYO=/root/autodl-tmp/venv_ocr/bin/python
BCTR=/root/autodl-tmp/bctr
BASE=/root/autodl-tmp/bctr_baselines
E5=ly/exp/C-05_crnn; E16=ly/exp/C-16_external
export CUDA_VISIBLE_DEVICES=0
mkdir -p logs/v4 $E16
exec 9> logs/v4_c16.lock
flock -n 9 || { echo "C16_ALREADY_RUNNING $(date '+%m-%d %H:%M')"; exit 1; }
mark() { echo "V4_$1 $(date '+%m-%d %H:%M')"; }
mark "C16_START"

# ---------------------------------------------------------------- (b, c) handwritten lines
if [ -f $BCTR/hwdb/hwdb_test/data.mdb ] && [ -f $BCTR/hwdb/hwdb_val/data.mdb ]; then
  H=$E16/hwdb_dumps; mkdir -p $H
  for s in val test; do
    for m in gtc b; do [ -f $H/${m}_${s}_hwdb.meta.json ] || $PYO -W ignore ly/scripts/a0_dump_ctc.py --model $m --split $s \
      --domain hwdb --dataset-root $BCTR --device cuda --workers 4 --out $H > logs/v4/c16_dump_${m}_$s.log 2>&1 \
      || mark "C16_DUMP_FAIL $m $s"; done
    for t in T S B; do d=ly/exp/C-04_external/official/$t; ck=$d/best.pth; [ -f $ck ] || ck=$d/epoch_100.pth
      [ -f $H/off_${t}_${s}_hwdb.meta.json ] || $PYO -W ignore ly/scripts/c4_dump_external.py --config $d/config.yml \
      --checkpoint $ck --tag off_$t --split $s --domain hwdb --out $H > logs/v4/c16_dump_off_${t}_$s.log 2>&1 \
      || mark "C16_DUMP_FAIL off_$t $s"; done
    for r in crnn_s1 crnn_s2; do [ -f $H/${r}_${s}_hwdb.meta.json ] || $PYO -W ignore ly/scripts/c4_dump_external.py \
      --config $E5/$r/config.yml --checkpoint $E5/$r/epoch_20.pth --tag $r --split $s --domain hwdb --out $H --trusted \
      > logs/v4/c16_dump_${r}_$s.log 2>&1 || mark "C16_DUMP_FAIL $r $s"; done
  done
  mark "C16BC_DUMPS_DONE"
  nice -n 10 $PYO -W ignore ly/scripts/c16_hwdb.py --dumps $H --out $E16/hwdb --tags off_T off_S off_B gtc b crnn_s1 crnn_s2 \
    > logs/v4/c16_hwdb.log 2>&1 && mark "C16BC_DONE" || mark "C16BC_ANALYSIS_FAIL"
else mark "C16BC_NO_DATA"; fi

# ---------------------------------------------------------------- (a) BCTR TransOCR / ASTER
TAGS=""
for m in transocr aster; do
  ext=.pth; [ $m = aster ] && ext=.pth.tar
  if ! ls $BASE/${m}_scene$ext $BASE/${m}_web$ext $BASE/${m}_document$ext > /dev/null 2>&1; then mark "C16A_NO_CHECKPOINTS $m"; continue; fi
  [ -f $E16/train_counts_document.json ] || $PYO ly/scripts/c16_bctr_baselines.py counts --dataset-root $BCTR --out $E16 \
    > logs/v4/c16_counts.log 2>&1 || { mark "C16A_COUNTS_FAIL"; break; }
  $PYO -W ignore ly/scripts/c16_bctr_baselines.py infer --model $m --checkpoint $BASE/${m}_scene$ext --domain scene --split val \
    --limit 2 --counts $E16/train_counts_scene.json --ref $E5/dumps/crnn_s1_val_scene.jsonl --out $E16/smoke \
    > logs/v4/c16_smoke_$m.log 2>&1 && mark "C16A_SMOKE_OK $m" || { mark "C16A_SMOKE_FAIL $m"; continue; }
  for d in scene web document; do for s in val test; do
    [ -f $E16/baselines/${m}_${s}_$d.meta.json ] || $PYO -W ignore ly/scripts/c16_bctr_baselines.py infer --model $m \
      --checkpoint $BASE/${m}_$d$ext --domain $d --split $s --counts $E16/train_counts_$d.json \
      --ref $E5/dumps/crnn_s1_${s}_$d.jsonl --out $E16/baselines > logs/v4/c16_${m}_${s}_$d.log 2>&1 \
      || mark "C16A_INFER_FAIL $m $s $d"
  done; done
  TAGS="$TAGS $m"
done
if [ -n "$TAGS" ]; then
  nice -n 5 $PYO -W ignore ly/scripts/c9_analysis.py --preds $E16/baselines --out $E16/baselines --norm official --tags $TAGS \
    > logs/v4/c16_baselines_analysis.log 2>&1 && mark "C16A_DONE" || mark "C16A_ANALYSIS_FAIL"
fi
mark "C16_ALL_DONE"
