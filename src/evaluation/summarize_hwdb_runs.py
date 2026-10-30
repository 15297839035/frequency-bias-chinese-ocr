"""A1-02: tabulate HWDB-LT results and apply the pass lines pre-registered in docs/experiments.md (A1-00 定稿).

Main line: ugls test top-1 >= ls + 0.5pp and >= csls + 0.5pp (mean over seeds), and ugls > ls, ugls > csls
in every seed. Auxiliary line: ugls few-group accuracy >= ls + 2pp and overall top-1 not below ls (means).

    python src/evaluation/summarize_hwdb_runs.py results/A1-02_hwdb_lt            # writes report.md next to the runs
"""
import json
import sys
from pathlib import Path

import numpy as np

ORDER = ["ce", "ls", "la", "csls", "sals", "gls", "ugls", "ugls_f"]
METRICS = [("acc", "top-1"), ("acc_many", "many"), ("acc_medium", "medium"), ("acc_few", "few"), ("ece", "ECE"),
           ("error_ids_ed_mean", "err IDS-ED"), ("error_lookalike_share", "err look-alike %")]


def load(root):
    runs = {}
    for f in sorted(Path(root).glob("*_s*/result.json")):
        loss, seed = f.parent.name.rsplit("_s", 1)
        runs.setdefault(loss, {})[int(seed)] = json.loads(f.read_text())["test"]
    return runs


def main(root):
    runs = load(root)
    seeds = sorted({s for r in runs.values() for s in r})
    lines = [f"# A1-02 HWDB-LT results (ICDAR2013 test, seeds {seeds})", "",
             "| loss | " + " | ".join(name for _, name in METRICS) + " | top-1 per seed |",
             "|---|" + "---:|" * len(METRICS) + "---|"]
    mean = {}
    for loss in ORDER + sorted(set(runs) - set(ORDER)):
        if loss not in runs:
            continue
        r = runs[loss]
        mean[loss] = {k: float(np.mean([r[s][k] for s in r if r[s][k] is not None])) for k, _ in METRICS}
        per_seed = ", ".join(f"s{s}: {r[s]['acc']:.2f}" for s in sorted(r))
        lines.append(f"| {loss} | " + " | ".join(f"{mean[loss][k]:.2f}" for k, _ in METRICS) + f" | {per_seed} |")
    lines.append("")
    if all(l in runs and set(runs[l]) == set(seeds) for l in ("ugls", "ls", "csls")) and len(seeds) >= 2:
        u, l, c = runs["ugls"], runs["ls"], runs["csls"]
        d_ls = mean["ugls"]["acc"] - mean["ls"]["acc"]
        d_cs = mean["ugls"]["acc"] - mean["csls"]["acc"]
        each = all(u[s]["acc"] > l[s]["acc"] and u[s]["acc"] > c[s]["acc"] for s in seeds)
        main_ok = d_ls >= 0.5 and d_cs >= 0.5 and each
        d_few = mean["ugls"]["acc_few"] - mean["ls"]["acc_few"]
        aux_ok = d_few >= 2.0 and d_ls >= 0
        lines += ["## Pre-registered verdict (A1-00 定稿)", "",
                  f"- ugls - ls (top-1, mean): {d_ls:+.2f}pp (need >= +0.50)",
                  f"- ugls - csls (top-1, mean): {d_cs:+.2f}pp (need >= +0.50)",
                  f"- ugls beats ls and csls in every seed: {each}",
                  f"- **Main line: {'PASS' if main_ok else 'FAIL'}**",
                  f"- ugls - ls (few group, mean): {d_few:+.2f}pp (need >= +2.00, with top-1 not below ls)",
                  f"- **Auxiliary line (tail-effective): {'PASS' if aux_ok else 'FAIL'}**"]
    else:
        lines.append("Verdict pending: ugls, ls and csls need results for every seed (>= 2 seeds).")
    report = "\n".join(lines) + "\n"
    (Path(root) / "report.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/A1-02_hwdb_lt")
