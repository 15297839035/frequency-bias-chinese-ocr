"""C-07 figure (server): flipping the training frequencies flips the direction of confusions (HWDB, ICDAR 2013 test).

(a) pairs confused under both orders (N >= 3 each, 4 seeds pooled per order): share of errors that go to the
    BCTR-frequent member, natural order (x) vs reversed order (y); marker size ~ errors.
(b) exposure-corrected log-odds of erring toward the member with more training images, binned by the log ratio of
    training images, for natural / reversed / random orders (C-03b style, all pairs with N >= 3).

    python ly/scripts/c7_figure.py --data ly/data/hwdb --nat ... --rev ... --perm ... --out ly/paper/figures
"""
import argparse
import collections
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import a0_prior_correction as a0  # noqa: E402
from c7_flip_direction import confusion, load_run  # noqa: E402
from w1_figures import AQUA, BLUE, GRID, INK, MUTED, ORANGE, save, style  # noqa: E402


def pooled(runs):
    n = collections.Counter()
    for run in runs:
        pred, labels, counts = load_run(run, "test")
        n.update(confusion(pred, labels))
    return n, counts, np.bincount(labels, minlength=len(counts))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True)
    p.add_argument("--nat", nargs="+", required=True, type=Path)
    p.add_argument("--rev", nargs="+", required=True, type=Path)
    p.add_argument("--perm", nargs="*", default=[], type=Path)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()
    classes = (Path(args.data) / "classes.txt").read_text(encoding="utf-8")
    bctr = np.array([a0.COUNT_OF.get(c, 0) for c in classes], float)
    conds = {"natural order": (args.nat, BLUE), "reversed order": (args.rev, ORANGE)}
    if args.perm:
        conds["random order"] = (args.perm, AQUA)
    data = {name: pooled(runs) for name, (runs, _) in conds.items()}
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 3.1), gridspec_kw={"width_ratios": [1, 1.25]})
    # (a) same pairs, both orders
    n_nat, _, occ = data["natural order"]
    n_rev = data["reversed order"][0]
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
    a.scatter(xs, ys, s=np.clip(sz, 4, 60), color=BLUE, alpha=0.35, lw=0)
    a.axhline(0.5, color=MUTED, lw=0.8, ls=":")
    a.axvline(0.5, color=MUTED, lw=0.8, ls=":")
    a.plot([0, 1], [0, 1], color=GRID, lw=0.8)
    a.scatter([np.average(xs, weights=sz)], [np.average(ys, weights=sz)], marker="X", s=70, color=INK, zorder=3,
              label="error-weighted mean")
    a.set_xlim(-0.02, 1.02)
    a.set_ylim(-0.02, 1.02)
    a.set_xlabel("share toward BCTR-frequent char\n(trained in natural order)")
    a.set_ylabel("same share, trained in reversed order")
    a.set_title(f"(a) {len(xs)} pairs confused under both orders")
    a.legend(loc="upper left", fontsize=7)
    style(a)
    # (b) binned direction log-odds vs training-image ratio
    edges = np.array([0.0, 0.25, 0.75, 1.5, 2.5, 3.5, 5.0])
    for name, (runs, color) in conds.items():
        n, counts, occ = data[name]
        rows = []
        for u, v in {tuple(sorted(k)) for k in n}:
            N = n[(u, v)] + n[(v, u)]
            if N < 3 or counts[u] == counts[v]:
                continue
            lo, hi = (u, v) if counts[u] < counts[v] else (v, u)  # hi = more training images
            rows.append((np.log((counts[hi] + 1) / (counts[lo] + 1)), n[(lo, hi)], N, occ[lo] / (occ[lo] + occ[hi])))
        R = np.array(rows)
        mids, vals = [], []
        for e0, e1 in zip(edges[:-1], edges[1:]):
            m = (R[:, 0] >= e0) & (R[:, 0] < e1)
            if m.sum() < 5:
                continue
            k, N, p0 = R[m, 1].sum(), R[m, 2].sum(), np.average(R[m, 3], weights=R[m, 2])
            obs = (k + 0.5) / (N + 1)
            mids.append(R[m, 0].mean())
            vals.append(np.log(obs / (1 - obs)) - np.log(p0 / (1 - p0)))
        b.plot(mids, vals, "-o", color=color, ms=4, lw=1.6, label=name)
    xx = np.linspace(0, 5, 10)
    b.plot(xx, xx, color=GRID, lw=1, ls="--", label="slope 1 (Bayes prior)")
    b.axhline(0, color=MUTED, lw=0.8, ls=":")
    b.set_xlabel("log ratio of training images (more / fewer)")
    b.set_ylabel("log-odds of erring toward the class\nwith more images (vs symmetric)")
    b.set_title("(b) direction follows training frequency")
    b.legend(fontsize=7, loc="upper left")
    style(b)
    args.out.mkdir(parents=True, exist_ok=True)
    save(fig, args.out, "fig5_flip")


if __name__ == "__main__":
    main()
