cd /root/autodl-tmp/work
PY=/root/miniconda3/bin/python
E=ly/exp/C-01b_hwdb_seeds
until grep -q QUEUE_DONE logs/c1b_queue_s2.log && grep -q QUEUE_DONE logs/c1b_queue_s3.log; do
  if ! pgrep -f "[c]1b_run_queue.sh" > /dev/null; then echo "C1B_QUEUE_FAILED $(date +%H:%M)"; exit 1; fi
  sleep 60; done
echo "C1B_TRAINING_DONE $(date +%H:%M)"
(cd $E && for d in *_s?; do echo "$d $(sha256sum $d/model.pt | cut -c1-12)"; done > model_sha256_12.txt)
CUDA_VISIBLE_DEVICES=1 $PY ly/scripts/c1_reeval_hwdb.py dump --data ly/data/hwdb --runs $(ls -d $E/*_s?) --out $E/dumps_new \
  > logs/c1b_dump.log 2>&1 || { echo C1B_DUMP_FAILED; exit 1; }
mkdir -p $E/dumps_all
for d in ly/exp/C-01_bal_vs_nat/dumps/*/ $E/dumps_new/*/; do ln -sfn "$(readlink -f $d)" $E/dumps_all/$(basename $d); done
cp ly/exp/C-01_bal_vs_nat/dumps/lt_counts.npy $E/dumps_all/
cmp ly/exp/C-01_bal_vs_nat/dumps/lt_counts.npy $E/dumps_new/lt_counts.npy && echo LT_COUNTS_SAME
$PY -W ignore ly/scripts/c1_reeval_hwdb.py analyze --dumps $E/dumps_all --out $E/all_seeds > logs/c1b_analyze_all.log 2>&1 \
  && echo "C1B_ALL_DONE $(date +%H:%M)" || echo C1B_ALL_FAILED
$PY -W ignore ly/scripts/c1_reeval_hwdb.py analyze --dumps $E/dumps_new --out $E/new_seeds_only > logs/c1b_analyze_new.log 2>&1 \
  && echo "C1B_NEW_DONE $(date +%H:%M)" || echo C1B_NEW_FAILED
