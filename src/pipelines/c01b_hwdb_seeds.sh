#!/bin/bash
# C-01b: extra seeds for the HWDB-LT matrix (A1-02's 8 losses + A1-03's 4 la+smoothing losses), same settings.
# Merged from a1_run_queue.sh and a1_run_queue_la.sh; only the output directory differs.
#   CUDA_VISIBLE_DEVICES=0 bash src/pipelines/c01b_hwdb_seeds.sh 2
set -euo pipefail
SEED=${1:?seed}
PY=${PY:-$(command -v python || echo /root/miniconda3/bin/python)}
DATA=${DATA:-data/hwdb}
OUT=results/C-01b_hwdb_seeds
GLYPH=results/A1-00_prep/glyph_neighbors_gb2312_l1/neighbors_k50.npy
CSLS=$OUT/ce_s$SEED/csls_neighbors_k10.npy
run() {  # loss, extra args...
  local loss=$1; shift
  local dir=$OUT/${loss}_s$SEED
  if [ -f "$dir/result.json" ]; then echo "skip $dir"; return; fi
  $PY src/training/train_hwdb_classifier.py train --data "$DATA" --split lt100 --loss "$loss" --seed "$SEED" --out "$dir" "$@" \
    > "$dir.log" 2>&1
}
mkdir -p $OUT
run ce
$PY src/training/train_hwdb_classifier.py prototypes --data "$DATA" --split lt100 --ckpt $OUT/ce_s$SEED/model.pt --out $OUT/ce_s$SEED
for loss in ls la sals; do run $loss; done
for loss in gls ugls ugls_f; do run $loss --glyph-neighbors $GLYPH; done
run csls --neighbors $CSLS
run la_ls
run la_sals
run la_ugls --glyph-neighbors $GLYPH
run la_csls --neighbors $CSLS
echo "QUEUE_DONE seed=$SEED"
