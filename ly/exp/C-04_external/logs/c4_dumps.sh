cd /root/autodl-tmp/work
PY=/root/autodl-tmp/venv_ocr/bin/python
E=ly/exp/C-04_external
for k in T S B; do for split in val test; do for domain in scene web document; do
  CUDA_VISIBLE_DEVICES=1 $PY ly/scripts/c4_dump_external.py --config $E/official/$k/config.yml --checkpoint $E/official/$k/best.pth \
    --tag off_$k --split $split --domain $domain --out $E/dumps > logs/c4_dump_${k}_${split}_${domain}.log 2>&1 \
    || echo "C4_DUMP_FAIL $k $split $domain"
done; done; done
echo "C4_DUMPS_DONE $(date +%H:%M)"
