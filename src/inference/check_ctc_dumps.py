"""A0 sanity check: greedy decode in a dump must match the frozen seed48 test predictions.
Usage: .venv/bin/python src/inference/check_ctc_dumps.py results/A0-01_ctc_dumps/dumps/gtc_test_web.jsonl
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
path = Path(sys.argv[1])
model, split, domain = path.stem.split("_")
assert split == "test", "frozen reference predictions exist only for test"
frozen = {}
for line in open(REPO / f"OpenOCR-main/output/rec/scheme51_frozen_test_seed48/{domain}_predictions.jsonl", encoding="utf-8"):
    r = json.loads(line)
    frozen[r["sample_id"]] = (r["target_normalized"], r["models"][model]["prediction_normalized"])
n = same_target = same_pred = 0
diffs = []
for line in open(path, encoding="utf-8"):
    r = json.loads(line)
    n += 1
    ref_target, ref_pred = frozen[r["sample_id"]]
    same_target += r["target_normalized"] == ref_target
    if r["prediction_normalized"] == ref_pred:
        same_pred += 1
    else:
        diffs.append((r["sample_id"], ref_pred, r["prediction_normalized"]))
print(f"{path.name}: n={n}, target match={same_target}, prediction match={same_pred} ({100*same_pred/n:.3f}%)")
for d in diffs[:10]:
    print("  diff", d)
