"""C-10 report: HWDB long-tail baselines vs ce / la, before and after post-hoc frequency correction.

Inputs are reevaluate_hwdb.py `analyze` outputs (c1_results.json) for the new runs and for the reference ce / la runs
(C-01b all seeds on HWDB-LT, C-01n on HWDB-NAT), plus c10_tau_normalization.py outputs. Seeds 0 and 1 (the C-10 seeds).
Descriptive rules (docs/experiments.md, "C-07 至 C-11 预注册", C-10), means over seeds:
  (b) Acc_bal(m, own tau*(bal)) - Acc_bal(ce, tau*(bal)) <= 0.3pp
  (c) Acc_nat(m, own tau*(nat)) <= Acc_nat(ce, tau*(nat)) + 0.2pp
A method meeting both is "explained by frequency correction"; otherwise it is reported as an exception.

    python src/analysis/c10_longtail_report.py --out results/C-10_lt_baselines
"""
import argparse
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
EXP = REPO / "results"
REF = {"lt100": EXP / "C-01b_hwdb_seeds/all_seeds/c1_results.json", "nat": EXP / "C-01n_hwdb_nat/c1_results.json"}
SEEDS = (0, 1)
METHODS = ("ce", "la", "cb", "focal", "rs_sqrt", "rad", "tnorm")


def rows_for(split, out):
    ref = json.loads(REF[split].read_text())["runs"]
    new = json.loads((out / f"analysis_{split}" / "c1_results.json").read_text())["runs"]
    tn_path = out / f"tnorm_{split}.json"
    tn = json.loads(tn_path.read_text())["runs"] if tn_path.exists() else {}
    table = {}
    for m in METHODS:
        vals = []
        for s in SEEDS:
            if m == "tnorm":
                r = tn.get(f"ce_s{s}")
                if r is None:
                    continue
                vals.append({"default_bal": r["default"]["bal"], "default_nat": r["default"]["nat"],
                             "default_few": r["default"]["few"], "tuned_bal": r["test_frozen"]["t*(bal)"]["bal"],
                             "tuned_nat": r["test_frozen"]["t*(nat)"]["nat"], "tuned_tr": r["test_frozen"]["t*(tr)"]["tr"],
                             "few_at_tuned_nat": r["test_frozen"]["t*(nat)"]["few"],
                             "knob_bal": r["select"]["bal"]["t"], "knob_nat": r["select"]["nat"]["t"]})
                continue
            r = (ref if m in ("ce", "la") else new).get(f"{m}_s{s}")
            if r is None:
                continue
            f = r["test_frozen"]
            vals.append({"default_bal": f["default"]["bal"], "default_nat": f["default"]["nat"],
                         "default_few": f["default"]["few"], "tuned_bal": f["tau*(bal)"]["bal"],
                         "tuned_nat": f["tau*(nat)"]["nat"], "tuned_tr": f["tau*(tr)"]["tr"],
                         "few_at_tuned_nat": f["tau*(nat)"]["few"],
                         "knob_bal": r["select"]["bal"]["tau_eff"], "knob_nat": r["select"]["nat"]["tau_eff"]})
        if vals:
            table[m] = {k: float(np.mean([v[k] for v in vals])) for k in vals[0]} | {"seeds": len(vals)}
    ce = table["ce"]
    for m, v in table.items():
        v["gap_tuned_bal_vs_ce"] = v["tuned_bal"] - ce["tuned_bal"]
        v["gap_tuned_nat_vs_ce"] = v["tuned_nat"] - ce["tuned_nat"]
        v["rule_b"] = v["gap_tuned_bal_vs_ce"] <= 0.3
        v["rule_c"] = v["gap_tuned_nat_vs_ce"] <= 0.2
    return table


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()
    res, lines = {}, ["# C-10 HWDB long-tail baselines (test, mean of seeds 0 and 1)", ""]
    for split in ("lt100", "nat"):
        if not (args.out / f"analysis_{split}" / "c1_results.json").exists():
            lines += [f"## {split}", "", "(not run)", ""]
            continue
        table = rows_for(split, args.out)
        res[split] = table
        lines += [f"## {split}", "",
                  "| method | default bal | default nat | default few | tuned bal (knob) | tuned nat (knob) | few at tuned nat "
                  "| tuned tr | gap bal vs ce | gap nat vs ce | (b) | (c) |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|"]
        for m, v in table.items():
            lines.append(f"| {m} | {v['default_bal']:.2f} | {v['default_nat']:.2f} | {v['default_few']:.1f} | "
                         f"{v['tuned_bal']:.2f} ({v['knob_bal']:.2f}) | {v['tuned_nat']:.2f} ({v['knob_nat']:.2f}) | "
                         f"{v['few_at_tuned_nat']:.1f} | {v['tuned_tr']:.2f} | {v['gap_tuned_bal_vs_ce']:+.2f} | "
                         f"{v['gap_tuned_nat_vs_ce']:+.2f} | {v['rule_b']} | {v['rule_c']} |")
        lines.append("")
    lines.append("knob = tau_eff of the post-hoc logit adjustment (t of tau-normalisation for tnorm), selected on val.")
    (args.out / "c10_report.json").write_text(json.dumps(res, indent=1))
    (args.out / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
