cd /root/autodl-tmp/work
PY=/root/miniconda3/bin/python
E=ly/exp/C-01n_hwdb_nat
mkdir -p $E
cp ly/data/hwdb/train_nat_summary.json $E/
stream() { for s in "$@"; do for loss in ce la; do
  d=$E/${loss}_s$s
  [ -f $d/result.json ] || CUDA_VISIBLE_DEVICES=1 $PY ly/scripts/a1_train_hccr.py train --data ly/data/hwdb --split nat \
     --loss $loss --seed $s --out $d > $d.log 2>&1 || echo "C1N_TRAIN_FAIL $loss $s"
done; done; }
stream 0 1 & stream 2 3 & wait
echo "C1N_TRAINING_DONE $(date +%H:%M)"
(cd $E && for d in *_s?; do echo "$d $(sha256sum $d/model.pt | cut -c1-12)"; done > model_sha256_12.txt)
CUDA_VISIBLE_DEVICES=1 $PY ly/scripts/c1_reeval_hwdb.py dump --data ly/data/hwdb --split nat --runs $(ls -d $E/*_s?) \
  --out $E/dumps > logs/c1n_dump.log 2>&1 || { echo C1N_DUMP_FAILED; exit 1; }
$PY -W ignore ly/scripts/c1_reeval_hwdb.py analyze --dumps $E/dumps --out $E > logs/c1n_analyze.log 2>&1 \
  && echo "C1N_DONE $(date +%H:%M)" || echo C1N_ANALYZE_FAILED
