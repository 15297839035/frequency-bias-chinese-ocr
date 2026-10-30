#!/bin/bash
# V4A CPU analyses (plan v4 section A), run once on the server next to the v4 pipeline; outputs in ly/exp/V4A_cpu/
cd /root/autodl-tmp/work
PYO="nice -n 10 /root/autodl-tmp/venv_ocr/bin/python -W ignore"
O=ly/exp/V4A_cpu; mkdir -p $O logs/v4
L=ly/exp/C-01b_hwdb_seeds/dumps_all; NAT=ly/exp/C-01n_hwdb_nat/dumps; REV=ly/exp/C-07_freq_flip/dumps_rev; PERM=ly/exp/C-07_freq_flip/dumps_perm
mark() { echo "V4A_$1 $(date '+%m-%d %H:%M')"; }
$PYO ly/scripts/v4a_direction.py hwdb --out $O/hwdb --data ly/data/hwdb \
  --group lt_ce=$L/ce_s0,$L/ce_s1,$L/ce_s2,$L/ce_s3 lt_rev=$REV/rev_ce_s0,$REV/rev_ce_s1,$REV/rev_ce_s2,$REV/rev_ce_s3 \
          lt_perm=$PERM/perm_ce_s0,$PERM/perm_ce_s1 lt_la=$L/la_s0,$L/la_s1,$L/la_s2,$L/la_s3 \
          nat_ce=$NAT/ce_s0,$NAT/ce_s1,$NAT/ce_s2,$NAT/ce_s3 nat_la=$NAT/la_s0,$NAT/la_s1,$NAT/la_s2,$NAT/la_s3 \
  > logs/v4/v4a_hwdb.log 2>&1 && mark HWDB_DONE || mark HWDB_FAIL
$PYO ly/scripts/v4a_direction.py hwdb_tau --out $O/hwdb_tau --lt $L --nat $NAT > logs/v4/v4a_hwdb_tau.log 2>&1 \
  && mark HWDB_TAU_DONE || mark HWDB_TAU_FAIL
$PYO ly/scripts/v4a_direction.py lines --out $O/lines --procs 6 \
  --ctc gtc=ly/exp/C-03_analysis/dumps b=ly/exp/C-03_analysis/dumps off_T=ly/exp/C-04_external/dumps \
        off_S=ly/exp/C-04_external/dumps off_B=ly/exp/C-04_external/dumps crnn_s1=ly/exp/C-05_crnn/dumps \
        crnn_s2=ly/exp/C-05_crnn/dumps svtrv2_ch=ly/exp/C-04b_industrial/dumps repsvtr_ch=ly/exp/C-04b_industrial/dumps \
  --smtr gtc=ly/exp/C-09_smtr/preds b=ly/exp/C-09_smtr/preds off_T=ly/exp/C-09_smtr/preds off_S=ly/exp/C-09_smtr/preds \
         off_B=ly/exp/C-09_smtr/preds > logs/v4/v4a_lines.log 2>&1 && mark LINES_DONE || mark LINES_FAIL
mark ALL_DONE
