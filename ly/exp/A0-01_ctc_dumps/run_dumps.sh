#!/bin/bash
# A0-01: dump CTC posteriors for GTC and B (seed48, epoch 100) on BCTR val/test, then check test dumps.
set -e
cd "$(dirname "$0")/../../.."
for model in gtc b; do
  for split in val test; do
    for domain in scene web document; do
      ly/.venv/bin/python ly/scripts/a0_dump_ctc.py --model $model --split $split --domain $domain --device mps
      if [ "$split" = test ]; then
        ly/.venv/bin/python ly/scripts/a0_check_dump.py ly/exp/A0-01_ctc_dumps/dumps/${model}_${split}_${domain}.jsonl
      fi
    done
  done
done
echo ALL_DONE
