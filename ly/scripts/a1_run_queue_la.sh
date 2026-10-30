#!/bin/bash
# A1-03 (exploratory, ly/commit.md): la + smoothing on HWDB-LT, same data/model/training as A1-02.
# Reuses the A1-02 ce run of the same seed for the csls neighbours.
#   CUDA_VISIBLE_DEVICES=0 bash ly/scripts/a1_run_queue_la.sh 0
set -euo pipefail
SEED=${1:?seed}
PY=${PY:-$(command -v python || echo /root/miniconda3/bin/python)}
DATA=${DATA:-ly/data/hwdb}
OUT=ly/exp/A1-03_la_plus
GLYPH=ly/exp/A1-00_prep/glyph_neighbors_gb2312_l1/neighbors_k50.npy
CSLS=ly/exp/A1-02_hwdb_lt/ce_s$SEED/csls_neighbors_k10.npy
run() {  # loss, extra args...
  local loss=$1; shift
  local dir=$OUT/${loss}_s$SEED
  if [ -f "$dir/result.json" ]; then echo "skip $dir"; return; fi
  $PY ly/scripts/a1_train_hccr.py train --data "$DATA" --split lt100 --loss "$loss" --seed "$SEED" --out "$dir" "$@" \
    2>&1 | tee "$dir.log"
}
mkdir -p $OUT
run la_ls
run la_sals
run la_ugls --glyph-neighbors $GLYPH
run la_csls --neighbors $CSLS
echo "QUEUE_DONE seed=$SEED"
