#!/bin/bash
# A1-02: HWDB-LT run matrix (pre-registered in docs/experiments.md, A1-00 + revision).
# Usage on the GPU server, from the repository root (with OpenOCR-main/ cloned next to src/):
#   bash src/pipelines/a1_hwdb_queue.sh 0        # seed 0
#   bash src/pipelines/a1_hwdb_queue.sh 1        # seed 1 (can run on a second GPU: CUDA_VISIBLE_DEVICES=1)
set -euo pipefail
SEED=${1:?seed}
PY=${PY:-$(command -v python || echo /root/miniconda3/bin/python)}
DATA=${DATA:-data/hwdb}
OUT=results/A1-02_hwdb_lt
GLYPH=results/A1-00_prep/glyph_neighbors_gb2312_l1/neighbors_k50.npy
run() {  # loss, extra args...
  local loss=$1; shift
  local dir=$OUT/${loss}_s$SEED
  if [ -f "$dir/result.json" ]; then echo "skip $dir"; return; fi
  $PY src/training/train_hwdb_classifier.py train --data "$DATA" --split lt100 --loss "$loss" --seed "$SEED" --out "$dir" "$@" \
    2>&1 | tee "$dir.log"
}
mkdir -p $OUT
run ce
$PY src/training/train_hwdb_classifier.py prototypes --data "$DATA" --split lt100 --ckpt $OUT/ce_s$SEED/model.pt --out $OUT/ce_s$SEED
for loss in ls la sals; do run $loss; done
for loss in gls ugls ugls_f; do run $loss --glyph-neighbors $GLYPH; done
run csls --neighbors $OUT/ce_s$SEED/csls_neighbors_k10.npy
echo "QUEUE_DONE seed=$SEED"
