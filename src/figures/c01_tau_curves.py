"""C-01 figure: test accuracy vs effective prior-correction strength (descriptive; selection was on val).

    .venv/bin/python src/figures/c01_tau_curves.py   -> results/C-01_bal_vs_nat/tau_tradeoff.{png,pdf}
"""
import csv
import collections
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

EXP = Path(__file__).resolve().parents[2] / "results/C-01_bal_vs_nat"
SERIES = {"ce": ("CE + post-hoc τ", "#2a78d6", "-"), "la": ("LA-trained + post-hoc τ", "#eb6834", "--")}
PANELS = (("bal", "Balanced (per-class mean)"), ("nat", "Natural text frequency (BCTR)"), ("few", "Few-shot classes (<20 train)"))
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3dd"


def main():
    curves = collections.defaultdict(dict)
    for r in csv.DictReader(open(EXP / "curves.csv")):
        if r["split"] == "test":
            curves[r["run"]][float(r["tau_eff"])] = r
    res = json.loads((EXP / "c1_results.json").read_text())["runs"]
    taus = np.array(sorted(curves["ce_s0"]))
    keep = (taus >= -1.0) & (taus <= 1.75)
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                         "ytick.color": MUTED, "font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2), sharex=True)
    for ax, (metric, title) in zip(axes, PANELS):
        for loss, (label, color, ls) in SERIES.items():
            y = np.mean([[float(curves[f"{loss}_s{s}"][t][metric]) for t in taus] for s in (0, 1)], 0)
            ax.plot(taus[keep], y[keep], color=color, ls=ls, lw=2, label=label)
        for m, name in (("bal", "τ*(bal)"), ("nat", "τ*(nat)")):
            t = np.mean([res[f"ce_s{s}"]["select"][m]["tau_eff"] for s in (0, 1)])
            ax.axvline(t, color=MUTED, lw=0.8, ls=":")
            if metric == "few":
                ax.text(t, ax.get_ylim()[0], f" {name}", color=MUTED, fontsize=7.5, va="bottom", rotation=90)
        ax.set_title(title, color=INK, fontsize=9.5, loc="left")
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlabel("effective correction strength τ_eff")
    axes[0].set_ylabel("ICDAR2013 accuracy (%)")
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(EXP / f"tau_tradeoff.{ext}", dpi=200)
    print("saved", EXP / "tau_tradeoff.png")


if __name__ == "__main__":
    main()
