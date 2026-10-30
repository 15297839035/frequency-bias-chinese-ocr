#!/bin/bash
# C-09 (server): SMTR step-wise correction on 5 GTC models (val / test x 3 domains), then analysis.
#   cd /root/autodl-tmp/work && (nohup bash src/pipelines/c09_smtr.sh > logs/c9_pipeline.log 2>&1 < /dev/null &)
cd /root/autodl-tmp/work
PY=/root/autodl-tmp/venv_ocr/bin/python
E=results/C-09_smtr
O=OpenOCR-main/output/rec
mkdir -p $E/preds logs/c9
declare -A CFG=( [gtc]=$O/bctr_scheme51_bs64/svtrv2_tiny_51_gtc_seed48
                 [b]=$O/bctr_scheme51_ab_bs64/svtrv2_tiny_51_ucr_b_mixedfreq_seed48
                 [off_T]=results/C-04_external/official/T [off_S]=results/C-04_external/official/S
                 [off_B]=results/C-04_external/official/B )
ckpt() { [ -f ${CFG[$1]}/best.pth ] && echo ${CFG[$1]}/best.pth || echo ${CFG[$1]}/epoch_100.pth; }
queue() { gpu=$1; shift
  for tag in "$@"; do for split in val test; do for domain in scene web document; do
    [ -f $E/preds/${tag}_${split}_${domain}.meta.json ] && continue
    CUDA_VISIBLE_DEVICES=$gpu $PY -W ignore src/inference/decode_smtr.py --config ${CFG[$tag]}/config.yml \
      --checkpoint $(ckpt $tag) --tag $tag --split $split --domain $domain --out $E/preds \
      > logs/c9/${tag}_${split}_${domain}.log 2>&1 || echo "C9_DECODE_FAIL $tag $split $domain"
  done; done; echo "C9_MODEL_DONE $tag $(date +%H:%M)"; done; }
queue 0 gtc off_T b &
queue 1 off_B off_S &
wait
echo "C9_DECODES_DONE $(date +%H:%M)"
$PY -W ignore src/analysis/c09_smtr_analysis.py --preds $E/preds --out $E --norm official --tags off_T off_S off_B gtc b \
  > logs/c9/analysis_official.log 2>&1 && echo "C9_official_DONE $(date +%H:%M)" || echo "C9_official_FAILED"
$PY -W ignore src/analysis/c09_smtr_analysis.py --preds $E/preds --out $E --norm strict --tags gtc b off_T off_S off_B \
  > logs/c9/analysis_strict.log 2>&1 && echo "C9_strict_DONE $(date +%H:%M)" || echo "C9_strict_FAILED"
