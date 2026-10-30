#!/bin/bash
# C-10 fallback (server, 2026-10-13): the GPUs became unavailable to new processes at about 15:40, after the eight
# HWDB-LT runs had finished; dump and analyse them on the CPU (FP32). HWDB-NAT runs need a GPU and are not run here.
cd /root/autodl-tmp/work
PY=/root/miniconda3/bin/python
PYO=/root/autodl-tmp/venv_ocr/bin/python
E=results/C-10_lt_baselines
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
$PY src/evaluation/reevaluate_hwdb.py dump --data data/hwdb --split lt100 --runs $E/lt100/*_s? --out $E/dumps_lt100 \
  --device cpu > logs/c10/dump_lt100_cpu.log 2>&1 && echo "C10_LT_DUMP_CPU_DONE $(date +%H:%M)" || echo "C10_LT_DUMP_CPU_FAIL"
$PY src/evaluation/reevaluate_hwdb.py analyze --dumps $E/dumps_lt100 --out $E/analysis_lt100 --skip-judge \
  > logs/c10/analyze_lt100_cpu.log 2>&1 && echo "C10_LT_ANALYZE_DONE $(date +%H:%M)" || echo "C10_LT_ANALYZE_FAIL"
$PYO src/analysis/c10_longtail_report.py --out $E > logs/c10/report.log 2>&1 && echo "C10_REPORT2_DONE $(date +%H:%M)" || echo "C10_REPORT2_FAIL"
