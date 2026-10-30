"""Upper bound on what a look-alike-aware fix could recover, from the seed48 GTC test predictions.

An error position is "look-alike" when the reference and predicted Han characters share at least
one IDS component. A line is "recoverable" when every error in it is a look-alike substitution.
No model is run. Usage: python3 src/analysis/d03_lookalike_ceiling.py OpenOCR-main
"""
import json
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from analysis.d03_longtail_quickcheck import align, cnt, ids, is_han, shares_component  # noqa: E402

ROOT = sys.argv[1] if len(sys.argv) > 1 else "OpenOCR-main"
for dom in ["scene", "web", "document"]:
    stats = {"rare": [0, 0, 0, 0], "all": [0, 0, 0, 0]}  # lines, correct, recoverable_lines, recoverable_rare_only
    sub_rare = sub_rare_la = sub_rare_la_up = 0
    for raw in open(f"{ROOT}/output/rec/scheme51_frozen_test_seed48/{dom}_predictions.jsonl", encoding="utf-8"):
        r = json.loads(raw)
        ref, hyp = r["target_normalized"], r["models"]["gtc"]["prediction_normalized"]
        rare_line = any(is_han(c) and cnt.get(c, 0) < 100 for c in ref)
        ops = [o for o in align(ref, hyp) if o[0] != "M"]
        la = lambda a, b: a is not None and b is not None and is_han(a) and is_han(b) and a in ids and b in ids and shares_component(a, b)
        recoverable = bool(ops) and all(o[0] == "S" and la(o[1], o[2]) for o in ops)
        rare_recoverable = bool(ops) and all(o[0] == "S" and la(o[1], o[2]) and cnt.get(o[1], 0) < 100 for o in ops)
        for key in (["rare", "all"] if rare_line else ["all"]):
            s = stats[key]
            s[0] += 1
            s[1] += not ops
            s[2] += recoverable
            s[3] += rare_recoverable
        for op, a, b in ops:
            if op == "S" and a is not None and is_han(a) and cnt.get(a, 0) < 100:
                sub_rare += 1
                if la(a, b):
                    sub_rare_la += 1
                    sub_rare_la_up += cnt.get(b, 0) > cnt.get(a, 0)
    print(f"\n=== {dom} (GTC seed48) ===")
    for key, name in [("rare", "lines with a <100-count char"), ("all", "all lines")]:
        n, ok, rec, rec_rare = stats[key]
        print(f"{name}: n={n}, acc={100*ok/n:.2f}%, +all-look-alike-errors-fixed -> {100*(ok+rec)/n:.2f}%, "
              f"+only-rare-target look-alike errors fixed -> {100*(ok+rec_rare)/n:.2f}%")
    print(f"substitutions whose reference char has <100 train count: {sub_rare}; look-alike: {100*sub_rare_la/max(1,sub_rare):.1f}%; "
          f"look-alike AND predicted char more frequent: {100*sub_rare_la_up/max(1,sub_rare):.1f}%")
