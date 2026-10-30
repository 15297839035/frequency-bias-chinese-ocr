cd /root/autodl-tmp/work
PY=/root/autodl-tmp/venv_ocr/bin/python
E=ly/exp/C-04b_industrial
mkdir -p logs/c4b
run() { gpu=$1; m=$2
  for split in val test; do for domain in scene web document; do
    CUDA_VISIBLE_DEVICES=$gpu $PY ly/scripts/c4b_dump_industrial.py --model $m --split $split --domain $domain \
      > logs/c4b/dump_${m}_${split}_${domain}.log 2>&1 || echo "C4B_DUMP_FAIL $m $split $domain"
  done; done; }
run 0 svtrv2_ch & run 1 repsvtr_ch & wait
echo "C4B_DUMPS_DONE $(date +%H:%M)"
(cd $E/models && sha256sum *.pth | cut -c1-12,65- > ../model_sha256_12.txt)
$PY ly/scripts/c4_analysis.py --dumps $E/dumps --out $E --norm official --tags svtrv2_ch repsvtr_ch > logs/c4b/analysis_official.log 2>&1 \
  && echo "C4B_DONE $(date +%H:%M)" || echo C4B_ANALYSIS_FAILED
