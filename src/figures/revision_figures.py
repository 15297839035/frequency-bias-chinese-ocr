"""Plan v4 paper figures (run on the server from the repo root; reads result files only).

    python src/figures/revision_figures.py --out paper/figures [--only fig3 fig5]
fig3_tau_hwdb : HWDB-LT / HWDB-NAT, ICDAR 2013 test, four seeds: (a) class-balanced and (b) natural-frequency accuracy
                against the effective strength (CE + post-hoc, LA + post-hoc), (c) accuracy-maximising strength for test
                distributions q ~ (n + 1)^lambda with its 0.05-point plateau and the calibrated line 1 - lambda (V4A-7)
fig2_direction: (supplement) (a) direction vs training log-ratio for GTC, (b) beta per decoder with its Bayes reference
fig5_flip     : (a) pairs confused under both the natural and the reversed order, (b) direction log-odds against the
                training-image ratio (C-07), (c) dose-response: share of errors toward the BCTR-frequent member against
                alpha (C-14; pair set of the flip test)
Fonts: STIX (Times-like, matches the elsarticle text), TrueType embedding (pdf.fonttype 42).
"""
import argparse
import collections
import csv
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
REPO = Path(__file__).resolve().parents[2]
EXP = REPO / "results"
BLUE, ORANGE, AQUA, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#e87ba4"
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3dd"
plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 8.5,
                     "pdf.fonttype": 42, "ps.fonttype": 42, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlesize": 9, "axes.titlelocation": "left",
                     "legend.frameon": False, "axes.linewidth": 0.7})


def style(ax):
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)


def save(fig, out, name):
    fig.tight_layout(pad=0.4)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}", dpi=300)
    plt.close(fig)
    print("saved", name)


# ----------------------------------------------------------------------------- fig 3
def curves(path):
    acc = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in csv.DictReader(open(path)):
        if r["split"] == "test":
            loss = r["run"].rsplit("_s", 1)[0]
            for m in ("bal", "nat"):
                acc[(loss, float(r["tau_eff"]))][m].append(float(r[m]))
    return acc


def selected(path, metric):
    runs = json.loads(Path(path).read_text())["runs"]
    return float(np.mean([r["select"][metric]["tau_eff"] for n, r in runs.items() if n.startswith("ce_s")]))


def fig3(out):
    """One row: (a) HWDB-LT and (b) HWDB-NAT accuracy minus its own maximum, class-balanced vs natural-frequency, CE
    (solid) and LA training (dashed) plus post-hoc correction; (c) tau* against the test-distribution exponent."""
    sets = [("HWDB-LT", "lt", EXP / "C-01b_hwdb_seeds/all_seeds"), ("HWDB-NAT", "nat", EXP / "C-01n_hwdb_nat")]
    tau_star = json.loads((EXP / "V4A_cpu/hwdb_tau/v4a_hwdb_tau.json").read_text())
    lambdas = np.array(tau_star["lambdas"])
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.35))
    for col, (name, key, d) in enumerate(sets):
        ax = axes[col]
        acc = curves(d / "curves.csv")
        taus = np.array(sorted({t for (l, t) in acc if l == "ce"}))
        keep = (taus >= -1) & (taus <= 1.75)
        for m, color, label in (("bal", MAGENTA, "class-balanced"), ("nat", AQUA, "natural-frequency")):
            for loss, ls in (("ce", "-"), ("la", "--")):
                y = np.array([np.mean(acc[(loss, t)][m]) for t in taus])
                ax.plot(taus[keep], (y - y.max())[keep], color=color, ls=ls, lw=1.4 if loss == "ce" else 1.0,
                        label=f"{label}" if loss == "ce" else None)
                if loss == "ce":
                    k = int(np.argmax(y))
                    ax.plot([taus[k]], [0], "o", color=color, ms=3.5)
        ax.plot([], [], color=MUTED, ls="--", lw=1.0, label="LA-trained")
        ax.set_ylim(-5, 0.4)
        ax.set_xlim(-1.05, 1.8)
        ax.set_xlabel(r"effective strength $\tau_{\mathrm{eff}}$")
        ax.set_title(f"({'ab'[col]}) {name}")
        style(ax)
    axes[0].set_ylabel("accuracy minus its maximum (points)")
    axes[0].legend(loc="lower center", fontsize=6.8, handlelength=1.8)
    ax = axes[2]
    for key, color, label in (("lt", BLUE, "HWDB-LT"), ("nat", ORANGE, "HWDB-NAT")):
        runs = [v for k, v in tau_star["runs"].items() if k.startswith(key + "/") and v["loss"] == "ce"]
        ts = np.array([[r["curves"][f"train^{float(lam)}"]["tau_star_test"] for lam in lambdas] for r in runs])
        lo = np.array([[r["curves"][f"train^{float(lam)}"]["plateau_test"][0] for lam in lambdas] for r in runs])
        hi = np.array([[r["curves"][f"train^{float(lam)}"]["plateau_test"][1] for lam in lambdas] for r in runs])
        ax.fill_between(lambdas, lo.mean(0), hi.mean(0), color=color, alpha=0.13, lw=0)
        ax.plot(lambdas, ts.mean(0), color=color, lw=1.4, marker="o", ms=2.3, label=label)
    ax.plot(lambdas, 1 - lambdas, color=INK, lw=0.9, ls=(0, (4, 2)), label=r"$1-\lambda$")
    ax.axvline(1, color=MUTED, lw=0.6, ls=":")
    ax.axhline(0, color=MUTED, lw=0.5)
    ax.text(1.03, -0.45, "test = training\ndistribution", fontsize=6.3, color=MUTED, va="bottom")
    ax.set_xlim(0, 1.5)
    ax.set_xlabel(r"test distribution $q_c\propto(n_c+1)^{\lambda}$")
    ax.set_ylabel(r"optimal strength $\tau^*$")
    ax.set_title(r"(c) $\tau^*$ vs. test distribution")
    ax.legend(loc="upper right", fontsize=6.8)
    style(ax)
    save(fig, out, "fig3_tau_hwdb")


# ----------------------------------------------------------------------------- fig 2 (supplement)
def fig2(out):
    """(a) pairs GTC confuses on BCTR val: direction log-odds vs training log-ratio with the fitted slope; (b) beta per
    decoder on BCTR test with its Bayes reference (V4A-2) -- replaces the old slope-1 reference line."""
    from core import prior_correction as a0
    from analysis import c03_confusion_asymmetry as c3b
    from analysis.c04_external_analysis import official_normalizer
    a0.DUMPS = EXP / "C-03_analysis/dumps"
    a0.normalize = official_normalizer()
    data = {dom: a0.load("gtc", "val", dom) for dom in a0.DOMAINS}
    refs = [t for dom in a0.DOMAINS for t in data[dom]["targets"]]
    hyps = [h for dom in a0.DOMAINS for h in a0.strings(data[dom], a0.decode(data[dom], None))]
    rows = c3b.oriented(c3b.pairs_from(refs, hyps))
    k, N, p0, x = c3b.arrays(rows)
    y = np.log((k + 0.5) / (N - k + 0.5)) - np.log(p0 / (1 - p0))
    look = np.array([r["look"] for r in rows])
    beta = c3b.fit_beta(k, N, p0, x)
    v4a = json.loads((EXP / "V4A_cpu/lines/v4a_lines.json").read_text())
    attn = EXP / "V4A_cpu/lines_attn/v4a_lines.json"  # TransOCR / ASTER (C-16 a), per-domain training counts
    if attn.exists():
        v4a.update(json.loads(attn.read_text()))
    names = {"gtc": "GTC", "b": "UCR-B", "off_T": "SVTRv2-T", "off_S": "SVTRv2-S", "off_B": "SVTRv2-B",
             "svtrv2_ch": "OpenOCR server", "repsvtr_ch": "OpenOCR mobile", "crnn_s1": "CRNN s1", "crnn_s2": "CRNN s2",
             "transocr": "TransOCR", "aster": "ASTER"}
    items = [(names[k.split(':')[1]] if k.startswith("attn:") else f"{names[k.split(':')[1]]} ({k.split(':')[0].upper()})",
              v["direction"])
             for k, v in v4a.items() if "beta" in v["direction"]]
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 0.9 + 0.17 * len(items)), gridspec_kw={"width_ratios": [1.15, 1]})
    for sel, color, label in ((look, BLUE, "look-alike pairs"), (~look, ORANGE, "other pairs")):
        a.scatter(x[sel], y[sel], s=np.clip(N[sel] * 2.5, 4, 60), color=color, alpha=0.5, lw=0, label=label)
    xs = np.linspace(0, x.max(), 50)
    a.plot(xs, beta * xs, color=INK, lw=1.2, label=rf"fit: $\beta$ = {beta:.2f}")
    a.axhline(0, color=MUTED, lw=0.6, ls=":")
    a.set_xlabel("log training-frequency ratio (frequent / rare)")
    a.set_ylabel("log-odds of erring toward the frequent\ncharacter minus symmetric expectation")
    a.set_title("(a) GTC (CTC), BCTR validation")
    a.legend(loc="upper left", fontsize=7)
    style(a)
    ys = np.arange(len(items))[::-1]
    for yi, (label, d) in zip(ys, items):
        b.errorbar([d["beta"]], [yi], xerr=[[d["beta"] - d["beta_ci95"][0]], [d["beta_ci95"][1] - d["beta"]]], fmt="o",
                   color=BLUE, ms=4, capsize=2, lw=1)
        b.plot([d["bayes"]["beta"]], [yi], marker="D", mfc="white", mec=ORANGE, ms=4.5, lw=0)
        b.text(d["beta_ci95"][1] + 0.03, yi, f"T/E {d['T/E']:.2f}", va="center", fontsize=6.5, color=MUTED)
    b.plot([], [], "o", color=BLUE, ms=4, label=r"$\beta$ (95% CI)")
    b.plot([], [], "D", mfc="white", mec=ORANGE, ms=4.5, label=r"Bayes reference $\beta_{\mathrm{B}}$")
    b.set_yticks(ys, [it[0] for it in items], fontsize=7)
    b.axvline(0, color=MUTED, lw=0.6, ls=":")
    b.set_xlim(0, 2.1)
    b.set_xlabel(r"$\beta$, BCTR test, uncorrected decoder")
    b.set_title("(b) every decoder, with its Bayes reference")
    b.legend(fontsize=6.8, loc="upper left")
    b.spines[["top", "right"]].set_visible(False)
    save(fig, out, "fig2_direction")


# ----------------------------------------------------------------------------- fig 5
def fig5(out, data):
    from core import prior_correction as a0
    from analysis.c07_flip_direction import confusion, load_run
    L = EXP / "C-01b_hwdb_seeds/dumps_all"
    R = EXP / "C-07_freq_flip/dumps_rev"
    P = EXP / "C-07_freq_flip/dumps_perm"
    conds = {"natural order": ([L / f"ce_s{s}" for s in range(4)], BLUE),
             "reversed order": ([R / f"rev_ce_s{s}" for s in range(4)], ORANGE),
             "random order": ([P / f"perm_ce_s{s}" for s in range(2)], AQUA)}
    classes = (Path(data) / "classes.txt").read_text(encoding="utf-8")
    bctr = np.array([a0.COUNT_OF.get(c, 0) for c in classes], float)

    def pooled(runs):
        n = collections.Counter()
        for run in runs:
            pred, labels, counts = load_run(run, "test")
            n.update(confusion(pred, labels))
        return n, counts, np.bincount(labels, minlength=len(counts))
    pdata = {name: pooled(runs) for name, (runs, _) in conds.items()}
    dose = json.loads((EXP / "C-14_dose/c14_results.json").read_text()) if (EXP / "C-14_dose/c14_results.json").exists() else None
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.55), gridspec_kw={"width_ratios": [1, 1.15, 1]})
    a, b, c = axes
    n_nat, _, occ = pdata["natural order"]
    n_rev = pdata["reversed order"][0]
    xs, ys, sz = [], [], []
    for u, v in {tuple(sorted(k)) for k in n_nat}:
        if bctr[u] == bctr[v]:
            continue
        r, f = (u, v) if bctr[u] < bctr[v] else (v, u)
        Nn, Nr = n_nat[(r, f)] + n_nat[(f, r)], n_rev[(r, f)] + n_rev[(f, r)]
        if Nn >= 3 and Nr >= 3:
            xs.append(n_nat[(r, f)] / Nn)
            ys.append(n_rev[(r, f)] / Nr)
            sz.append(Nn + Nr)
    xs, ys, sz = map(np.asarray, (xs, ys, sz))
    a.scatter(xs, ys, s=np.clip(sz, 3, 40), color=BLUE, alpha=0.3, lw=0)
    a.axhline(0.5, color=MUTED, lw=0.6, ls=":")
    a.axvline(0.5, color=MUTED, lw=0.6, ls=":")
    a.scatter([np.average(xs, weights=sz)], [np.average(ys, weights=sz)], marker="X", s=55, color=INK, zorder=3,
              label="error-weighted mean")
    a.set_xlim(-0.02, 1.02)
    a.set_ylim(-0.02, 1.02)
    a.set_xlabel("share toward BCTR-frequent member,\nnatural order")
    a.set_ylabel("same share, reversed order")
    a.set_title(f"(a) {len(xs):,} pairs, both orders")
    a.legend(loc="upper left", fontsize=7, handletextpad=0.2)
    style(a)
    edges = np.array([0.0, 0.25, 0.75, 1.5, 2.5, 3.5, 5.0])
    for name, (runs, color) in conds.items():
        n, counts, occ = pdata[name]
        rows = []
        for u, v in {tuple(sorted(k)) for k in n}:
            N = n[(u, v)] + n[(v, u)]
            if N < 3 or counts[u] == counts[v]:
                continue
            lo, hi = (u, v) if counts[u] < counts[v] else (v, u)
            rows.append((np.log((counts[hi] + 1) / (counts[lo] + 1)), n[(lo, hi)], N, occ[lo] / (occ[lo] + occ[hi])))
        Rr = np.array(rows)
        mids, vals = [], []
        for e0, e1 in zip(edges[:-1], edges[1:]):
            m = (Rr[:, 0] >= e0) & (Rr[:, 0] < e1)
            if m.sum() < 5:
                continue
            k, N, p0 = Rr[m, 1].sum(), Rr[m, 2].sum(), np.average(Rr[m, 3], weights=Rr[m, 2])
            obs = (k + 0.5) / (N + 1)
            mids.append(Rr[m, 0].mean())
            vals.append(np.log(obs / (1 - obs)) - np.log(p0 / (1 - p0)))
        b.plot(mids, vals, "-o", color=color, ms=3, lw=1.3, label=name)
    b.axhline(0, color=MUTED, lw=0.6, ls=":")
    b.set_xlabel("log ratio of training images (more / fewer)")
    b.set_ylabel("log-odds of erring toward the class\nwith more images (vs. symmetric)")
    b.set_title("(b) direction follows training images")
    b.legend(fontsize=7, loc="upper left")
    style(b)
    if dose:
        s = dose["test"]["share_to_bctr_frequent"]
        al = np.array(sorted(float(k) for k in s))
        sh = np.array([s[f"{x:g}" if f"{x:g}" in s else str(x)]["share"] for x in al])
        lo = np.array([s[f"{x:g}" if f"{x:g}" in s else str(x)]["ci95"][0] for x in al])
        hi = np.array([s[f"{x:g}" if f"{x:g}" in s else str(x)]["ci95"][1] for x in al])
        sym = np.mean([s[k]["symmetric"] for k in s])
        c.fill_between(al, lo, hi, color=BLUE, alpha=0.15, lw=0)
        c.plot(al, sh, "-o", color=BLUE, ms=3.5, lw=1.5)
        c.axhline(sym, color=MUTED, lw=0.8, ls="--")
        c.text(0.02, sym + 0.02, "symmetric", fontsize=7, color=MUTED)
        c.set_xticks(al, ["0\nnatural", "0.25", "0.5\nequal", "0.75", "1\nreversed"])
        c.set_ylim(0, 1)
    c.set_xlabel(r"dose $\alpha$")
    c.set_ylabel("share toward BCTR-frequent member")
    c.set_title("(c) dose--response".replace("--", "–"))
    style(c)
    save(fig, out, "fig5_flip")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--data", default="data/hwdb")
    p.add_argument("--only", nargs="*", default=["fig2", "fig3", "fig5"])
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for name in args.only:
        {"fig2": lambda: fig2(args.out), "fig3": lambda: fig3(args.out), "fig5": lambda: fig5(args.out, args.data)}[name]()
