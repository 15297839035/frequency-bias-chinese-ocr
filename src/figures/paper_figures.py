"""W-01 paper figures (run on the server from the repo root; reads existing result files, trains nothing).

    python src/figures/paper_figures.py --out paper/figures
fig2_direction  : (a) confusion direction vs frequency ratio (GTC, BCTR val), (b) beta per model (test)
fig3_tau_hwdb   : accuracy vs effective correction strength on HWDB-LT and HWDB-NAT (4 seeds, test)
fig4_bctr_tau   : BCTR val, change in rare-line / overall accuracy vs post-hoc tau (gamma = 0.9)
"""
import argparse
import collections
import csv
import json
import math
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
REPO = Path(__file__).resolve().parents[2]
EXP = REPO / "results"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3dd"
plt.rcParams.update({"font.size": 8.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.titlesize": 9, "axes.titlelocation": "left", "legend.frameon": False})


def style(ax):
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)


def save(fig, out, name):
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}", dpi=220)
    plt.close(fig)
    print("saved", name)


# ----------------------------------------------------------------------------- fig 2
def beta_table():
    """(label, beta, lo, hi, T/E, decoder) on BCTR test, uncorrected, pairs with N >= 3, official normalisation where
    available (C-08 for the CTC heads, C-09 for the SMTR heads); falls back to the earlier per-experiment results."""
    rows = []
    c8 = EXP / "C-08_ablation/c8_results_official.json"
    c8_crnn = EXP / "C-08_ablation/crnn/c8_results_official.json"
    names = {"gtc": "GTC (ours, T)", "b": "UCR-B (ours, T)", "off_T": "Official SVTRv2-T", "off_S": "Official SVTRv2-S",
             "off_B": "Official SVTRv2-B", "svtrv2_ch": "OpenOCR server", "repsvtr_ch": "OpenOCR mobile",
             "crnn_s1": "CRNN seed 1", "crnn_s2": "CRNN seed 2"}
    c8d = {}
    for path in (c8, c8_crnn):
        if path.exists():
            c8d.update(json.loads(path.read_text()))
    for tag, label in names.items():
        if tag in c8d and "beta" in c8d[tag]["beta_min_errors"]["3"]:
            r = c8d[tag]["beta_min_errors"]["3"]
            rows.append((label, r["beta"], *r["ci95"], r["T/E"], "CTC"))
    if not rows:  # older sources (strict for ours)
        c3b = EXP / "C-03_analysis/c3b_asymmetry.json"
        if c3b.exists():
            d = json.loads(c3b.read_text())
            for tag in ("gtc", "b"):
                r = d[f"{tag}_test_B0"]["all"]
                rows.append((names[tag], r["beta"]["estimate"], *r["beta"]["ci95"], r["toward_more_frequent"]["ratio"], "CTC"))
    c9 = EXP / "C-09_smtr/c9_results_official.json"
    if c9.exists():
        d = json.loads(c9.read_text())["models"]
        for tag in ("gtc", "b", "off_T", "off_S", "off_B"):
            if tag in d:
                r = d[tag]["mechanism"]["test"]["B0"]["asymmetry"]["all"]
                rows.append((names[tag].replace("Official ", "").replace(" (ours, T)", "") + " · SMTR", r["beta"]["estimate"],
                             *r["beta"]["ci95"],
                             r["toward_more_frequent"]["ratio"], "SMTR"))
    return rows


def pair_points():
    from core import prior_correction as a0
    from analysis import c03_confusion_asymmetry as c3b
    from analysis.c04_external_analysis import official_normalizer
    a0.DUMPS = EXP / "C-03_analysis/dumps"
    a0.normalize = official_normalizer()
    data = {dom: a0.load("gtc", "val", dom) for dom in a0.DOMAINS}
    refs = [t for dom in a0.DOMAINS for t in data[dom]["targets"]]
    hyps = [s for dom in a0.DOMAINS for s in a0.strings(data[dom], a0.decode(data[dom], None))]
    rows = c3b.oriented(c3b.pairs_from(refs, hyps))
    k, N, p0, x = c3b.arrays(rows)
    obs = np.log((k + 0.5) / (N - k + 0.5))
    exp = np.log(p0 / (1 - p0))
    look = np.array([r["look"] for r in rows])
    beta = c3b.fit_beta(k, N, p0, x)
    return x, obs - exp, N, look, beta


def fig2(out):
    x, y, N, look, beta = pair_points()
    rows = beta_table()
    fig, (a, b) = plt.subplots(1, 2, figsize=(9.6, 2.5 + 0.1 * len(rows)), gridspec_kw={"width_ratios": [1.25, 1]})
    for sel, color, label in ((look, BLUE, "look-alike pairs"), (~look, ORANGE, "other pairs")):
        a.scatter(x[sel], y[sel], s=np.clip(N[sel] * 3, 6, 80), color=color, alpha=0.55, lw=0, label=label)
    xs = np.linspace(0, x.max(), 50)
    a.plot(xs, beta * xs, color=INK, lw=1.4, label=f"fit: slope β = {beta:.2f}")
    a.axhline(0, color=MUTED, lw=0.8, ls=":")
    a.set_xlabel("log frequency ratio of the pair (frequent / rare), training set")
    a.set_ylabel("log odds of erring toward the frequent\ncharacter, minus symmetric expectation")
    a.set_title("(a) Confusions flow toward the more frequent character (GTC, BCTR val)")
    a.legend(loc="upper left", fontsize=7.5)
    style(a)
    labels = [r[0] for r in rows][::-1]
    est = np.array([r[1] for r in rows])[::-1]
    lo, hi = np.array([r[2] for r in rows])[::-1], np.array([r[3] for r in rows])[::-1]
    ys = np.arange(len(rows))
    dec = np.array([r[5] for r in rows])[::-1]
    for kind, color in (("CTC", BLUE), ("SMTR", ORANGE)):
        m = dec == kind
        if m.any():
            b.errorbar(est[m], ys[m], xerr=[est[m] - lo[m], hi[m] - est[m]], fmt="o", color=color, ms=5, capsize=2,
                       lw=1.2, label=f"{kind} decoder")
    b.legend(fontsize=7, loc="lower left")
    for yi, r in zip(ys, rows[::-1]):
        b.text(hi[yi] + 0.03, yi, f"T/E {r[4]:.2f}", va="center", fontsize=7, color=MUTED)
    b.axvline(0, color=MUTED, lw=0.8, ls=":")
    b.axvline(1, color=GRID, lw=0.8)
    b.set_yticks(ys, labels, fontsize=7.5)
    b.set_xlim(0, max(1.6, hi.max() + 0.35))
    b.set_xlabel("β (95% CI), BCTR test, official normalisation")
    b.set_title("(b) Same frequency-driven direction in every model")
    b.spines[["top", "right"]].set_visible(False)
    save(fig, out, "fig2_direction")


# ----------------------------------------------------------------------------- fig 3
def curves(path):
    acc = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in csv.DictReader(open(path)):
        if r["split"] == "test":
            loss = r["run"].rsplit("_s", 1)[0]
            acc[(loss, float(r["tau_eff"]))]["bal"].append(float(r["bal"]))
            acc[(loss, float(r["tau_eff"]))]["nat"].append(float(r["nat"]))
            acc[(loss, float(r["tau_eff"]))]["few"].append(float(r["few"]))
    return acc


def selected_tau(path, metric):
    runs = json.loads(Path(path).read_text())["runs"]
    return np.mean([r["select"][metric]["tau_eff"] for n, r in runs.items() if n.startswith("ce_s")])


def fig3(out):
    sets = [("HWDB-LT (exponential tail, 100×)", EXP / "C-01b_hwdb_seeds/all_seeds"),
            ("HWDB-NAT (real text frequencies)", EXP / "C-01n_hwdb_nat")]
    cols = (("bal", "Class-balanced accuracy"), ("nat", "Natural-frequency accuracy"), ("few", "Few-shot classes"))
    fig, axes = plt.subplots(2, 3, figsize=(10, 3.25), sharex=True)
    for row, (name, d) in enumerate(sets):
        acc = curves(d / "curves.csv")
        taus = np.array(sorted({t for (l, t) in acc if l == "ce"}))
        keep = (taus >= -1) & (taus <= 1.75)
        for col, (m, title) in enumerate(cols):
            ax = axes[row, col]
            for loss, color, ls, label in (("ce", BLUE, "-", "CE + post-hoc τ"), ("la", ORANGE, "--", "LA-trained + post-hoc τ")):
                y = np.array([np.mean(acc[(loss, t)][m]) for t in taus])
                ax.plot(taus[keep], y[keep], color=color, ls=ls, lw=1.8, label=label)
            for metric, txt in (("bal", "τ*bal"), ("nat", "τ*nat")):
                t = selected_tau(d / "c1_results.json", metric)
                ax.axvline(t, color=MUTED, lw=0.7, ls=":")
                if col == 2:
                    ax.text(t, ax.get_ylim()[0], " " + txt, rotation=90, va="bottom", fontsize=7, color=MUTED)
            ax.set_title(f"{title}" if row == 0 else "")
            style(ax)
            if row == 1:
                ax.set_xlabel("effective correction strength τ$_{eff}$")
        axes[row, 0].set_ylabel(name.replace(" (", "\n("), fontsize=8)
    axes[0, 0].legend(loc="lower right", fontsize=7.5)
    save(fig, out, "fig3_tau_hwdb")


# ----------------------------------------------------------------------------- fig 4
def fig4(out):
    series = []
    a0 = json.loads((EXP / "A0-03_prior_correction/val_grid.json").read_text())
    series.append(("GTC (ours)", a0["baseline"], [g for g in a0["grid"] if g["method"] == "G"]))
    for path, names in ((EXP / "C-04_external/c4_results_official.json", {"off_T": "Official T", "off_S": "Official S",
                                                                           "off_B": "Official B"}),
                        (EXP / "C-04b_industrial/c4_results_official.json", {"svtrv2_ch": "OpenOCR server",
                                                                              "repsvtr_ch": "OpenOCR mobile"})):
        if path.exists():
            d = json.loads(path.read_text())["models"]
            series += [(label, d[t]["val_B0"], d[t]["val_grid"]) for t, label in names.items() if t in d]
    colors = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, "#7a5af5", MUTED]
    fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3.2), sharex=True)
    for (label, base, grid), color in zip(series, colors):
        g = sorted([x for x in grid if abs(x["gamma"] - 0.9) < 1e-9], key=lambda x: x["tau"])
        taus = [0.0] + [x["tau"] for x in g]
        a.plot(taus, [0.0] + [x["rare_acc"] - base["rare_acc"] for x in g], "-o", ms=3, color=color, lw=1.5, label=label)
        b.plot(taus, [0.0] + [x["acc"] - base["acc"] for x in g], "-o", ms=3, color=color, lw=1.5, label=label)
    a.set_title("(a) Rare-character lines")
    b.set_title("(b) All lines")
    for ax in (a, b):
        ax.axhline(0, color=MUTED, lw=0.8, ls=":")
        ax.axvspan(0.15, 0.3, color=GRID, alpha=0.6, lw=0)
        ax.set_xlabel("post-hoc correction strength τ (γ = 0.9), BCTR val")
        style(ax)
    a.set_ylabel("accuracy change vs. no correction (pp)")
    a.legend(fontsize=7, loc="upper left")
    save(fig, out, "fig4_bctr_tau")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=REPO / "paper/figures")
    p.add_argument("--only", nargs="*", default=["fig2", "fig3", "fig4"])
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for name in args.only:
        {"fig2": fig2, "fig3": fig3, "fig4": fig4}[name](args.out)
