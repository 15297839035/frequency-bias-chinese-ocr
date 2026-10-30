cd /root/autodl-tmp/work
PY=/root/autodl-tmp/venv_ocr/bin/python
run() { gpu=$1; name=$2; tau=$3; seed=$4
  CUDA_VISIBLE_DEVICES=$gpu nohup $PY ly/scripts/c2_finetune.py train --tau $tau --seed $seed --epochs 10 \
    --out ly/exp/C-02_bctr_la/$name --workers 8 > logs/c2/$name.log 2>&1 < /dev/null &
}
run 0 c0_s1 0 1
run 0 la025_s1 0.25 1
run 0 la1_s1 1.0 1
run 1 c0_s2 0 2
run 1 la025_s2 0.25 2
run 1 la1_s2 1.0 2
