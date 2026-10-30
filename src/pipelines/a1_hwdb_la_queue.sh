#!/bin/bash
# A1-03 (exploratory, docs/experiments.md): la + smoothing on HWDB-LT, same data/model/training as A1-02.
# Reuses the A1-02 ce run of the same seed for the csls neighbours.
#   CUDA_VISIBLE_DEVICES=0 bash src/pipelines/a1_hwdb_la_queue.sh 0
set -euo pipefail
SEED=${1:?seed}
PY=${PY:-$(command -v python || echo /root/miniconda3/bin/python)}
DATA=${DATA:-data/hwdb}
OUT=results/A1-03_la_plus
GLYPH=results/A1-00_prep/glyph_neighbors_gb2312_l1/neighbors_k50.npy
CSLS=results/A1-02_hwdb_lt/ce_s$SEED/csls_neighbors_k10.npy
run() {  # loss, extra args...
  local loss=$1; shift
  local dir=$OUT/${loss}_s$SEED
  if [ -f "$dir/result.json" ]; then echo "skip $dir"; return; fi
  $PY src/training/train_hwdb_classifier.py train --data "$DATA" --split lt100 --loss "$loss" --seed "$SEED" --out "$dir" "$@" \
    2>&1 | tee "$dir.log"
}
mkdir -p $OUT
run la_ls
run la_sals
run la_ugls --glyph-neighbors $GLYPH
run la_csls --neighbors $CSLS
echo "QUEUE_DONE seed=$SEED"
