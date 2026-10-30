#!/bin/bash
# C-05 fallback (server, 2026-10-13): new processes cannot see the GPUs since about 15:40 ("Failed to initialize NVML"),
# while the running CRNN trainings keep their CUDA contexts. When c5_pipeline.sh has finished (its GPU dumps fail),
# dump on the CPU (FP32, same data pipeline), then run the analyses that after_c5.sh / c8_after_c5.sh would have run.
cd /root/autodl-tmp/work
PY=/root/autodl-tmp/venv_ocr/bin/python
E=ly/exp/C-05_crnn
until grep -qE "C5_official_DONE|C5_official_FAILED|C5_TRAIN_FAILED" logs/c5_pipeline.log 2>/dev/null; do sleep 120; done
grep -q "C5_TRAIN_FAILED" logs/c5_pipeline.log && { echo "C5CPU_SKIP_TRAIN_FAILED"; exit 1; }
if ! grep -q "C5_official_DONE" logs/c5_pipeline.log; then
  export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
  dump() { run=$1
    for split in val test; do for domain in scene web document; do
      [ -f $E/dumps/${run}_${split}_${domain}.meta.json ] && continue
      $PY ly/scripts/c4_dump_external.py --config $E/$run/config.yml --checkpoint $E/$run/epoch_20.pth --tag $run \
        --split $split --domain $domain --out $E/dumps --trusted --device cpu --threads 8 --workers 4 \
        > logs/c5/cpudump_${run}_${split}_${domain}.log 2>&1 || echo "C5CPU_DUMP_FAIL $run $split $domain"
    done; done; }
  dump crnn_s1 & dump crnn_s2 & wait
  echo "C5CPU_DUMPS_DONE $(date +%H:%M)"
  for norm in strict official; do
    $PY ly/scripts/c4_analysis.py --dumps $E/dumps --out $E --norm $norm --tags crnn_s1 crnn_s2 \
      > logs/c5/analysis_${norm}_cpu.log 2>&1 && echo "C5CPU_${norm}_DONE $(date +%H:%M)" || echo "C5CPU_${norm}_FAILED"
  done
fi
$PY -W ignore ly/scripts/c6_selfprior.py > logs/c6_with_crnn.log 2>&1 && echo "C5CPU_C6_DONE $(date +%H:%M)" || echo "C5CPU_C6_FAILED"
$PY -W ignore ly/scripts/c8_ablation.py --out ly/exp/C-08_ablation/crnn --norm official --workers 2 --tags crnn_s1 crnn_s2 \
  > logs/c8/official_crnn_cpu.log 2>&1 && echo "C5CPU_C8_DONE $(date +%H:%M)" || echo "C5CPU_C8_FAILED"
$PY -W ignore ly/scripts/w1_figures.py --out ly/paper/figures > logs/figs_with_crnn.log 2>&1 \
  && echo "C5CPU_FIGS_DONE $(date +%H:%M)" || echo "C5CPU_FIGS_FAILED"
