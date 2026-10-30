"""C-07: frequency-flip intervention on HWDB -- does confusion direction follow the *training* frequency?

The same HWDB-LT count profile is assigned to classes in natural BCTR-frequency order (the A1-02 / C-01b ce models),
reversed order (lt100rev) or a random order (lt100perm). ICDAR 2013 has about 60 images per class, so exposure is
nearly equal. Per model, C-03b's exposure-controlled statistics (T/E, z, beta) are computed with "more frequent" =
more training images in that model's own split. The flip test compares, on pairs confused under both orders, the share
of errors that go to the BCTR-frequent member. A joint conditional-binomial model separates the training-frequency
effect (beta) from a character-complexity effect (eta; IDS component count and HWDB ink, standardised).
Protocol: ly/commit.md, entry "C-07 至 C-11 预注册" (F1, F2).

    python ly/scripts/c7_flip_direction.py --data ly/data/hwdb \
        --nat ly/exp/C-01b_hwdb_seeds/dumps_all/ce_s{0,1,2,3} --rev ly/exp/C-07_freq_flip/dumps_rev/rev_ce_s{0,1,2,3} \
        --perm ly/exp/C-07_freq_flip/dumps_perm/perm_ce_s{0,1} --out ly/exp/C-07_freq_flip
"""
import argparse
import collections
import json
import math
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import a0_prior_correction as a0  # noqa: E402
import c3_analysis as c3  # noqa: E402
import c3b_asymmetry as c3b  # noqa: E402
from c1_reeval_hwdb import TAUS  # noqa: E402

IDC = set("⿰⿱⿲⿳⿴⿵⿶⿷⿸⿹⿺⿻")


def load_run(run, split):
    meta = json.loads((run / "meta.json").read_text())
    k0 = int(np.argmin(np.abs(TAUS - meta["tau_train"])))
    d = np.load(run / f"{split}.npz")
    counts = np.load(run.parent / "lt_counts.npy")
    return d["grid"][:, k0].astype(np.int64), d["labels"].astype(np.int64), counts


def confusion(pred, labels):
    n = collections.Counter()
    wrong = pred != labels
    for a, b in zip(labels[wrong], pred[wrong]):
        n[(int(a), int(b))] += 1
    return n


def rows_from(n, occ, counts, classes, min_errors=3):
    rows = []
    for a, b in {tuple(sorted(k)) for k in n}:
        N = n[(a, b)] + n[(b, a)]
        if N < min_errors or occ[a] == 0 or occ[b] == 0:
            continue
        rows.append(dict(a=a, b=b, n_ab=n[(a, b)], n_ba=n[(b, a)], occ_a=int(occ[a]), occ_b=int(occ[b]),
                         cnt_a=float(counts[a]), cnt_b=float(counts[b]), look=c3.lookalike(classes[a], classes[b])))
    return rows


def per_model(rows):
    return {grp: c3b.analyse([r for r in rows if sel(r)]) for grp, sel in
            (("all", lambda r: True), ("lookalike", lambda r: r["look"]), ("other", lambda r: not r["look"]))}


def complexity(classes, data):
    """Per class: IDS component count (non-IDC tokens, with repeats) and mean ink of the HWDB val images."""
    ids = {}
    for line in open(c3.IDS, encoding="utf-8-sig"):
        ch, sep, seq = line.rstrip("\n").partition(":")
        if sep:
            ids[ch] = sum(t not in IDC for t in seq.split())
    comp = np.array([ids.get(c, np.nan) for c in classes], float)
    images = np.load(Path(data) / "val_images.npy", mmap_mode="r")
    labels = np.load(Path(data) / "val_labels.npy")
    ink, n = np.zeros(len(classes)), np.zeros(len(classes))
    for s in range(0, len(labels), 50000):
        lab = labels[s:s + 50000]
        dark = (np.asarray(images[s:s + 50000]) < 128).mean((1, 2))
        ok = lab >= 0
        np.add.at(ink, lab[ok], dark[ok])
        np.add.at(n, lab[ok], 1)
    return comp, ink / np.maximum(n, 1)


def flip_test(n_nat, n_rev, occ, bctr, reps=1000, seed=0):
    """Pairs with N >= 3 under both orders; share of errors toward the BCTR-frequent member, nat vs rev."""
    pairs = []
    for a, b in {tuple(sorted(k)) for k in n_nat}:
        if bctr[a] == bctr[b]:
            continue
        r, f = (a, b) if bctr[a] < bctr[b] else (b, a)
        Nn, Nr = n_nat[(r, f)] + n_nat[(f, r)], n_rev[(r, f)] + n_rev[(f, r)]
        if Nn >= 3 and Nr >= 3:
            pairs.append((r, f, n_nat[(r, f)], Nn, n_rev[(r, f)], Nr, occ[r] / (occ[r] + occ[f])))
    P = np.array(pairs, float)
    if len(P) < 10:
        return {"pairs": len(P)}
    kn, Nn, kr, Nr, p0 = P[:, 2], P[:, 3], P[:, 4], P[:, 5], P[:, 6]

    def stats(i):
        return kn[i].sum() / Nn[i].sum(), kr[i].sum() / Nr[i].sum()
    sn, sr = stats(np.arange(len(P)))
    rng = np.random.default_rng(seed)
    diffs = [np.subtract(*stats(rng.integers(0, len(P), len(P)))) for _ in range(reps)]
    from scipy.stats import wilcoxon
    share_n, share_r = kn / Nn, kr / Nr
    flipped = ((share_n > 0.5) & (share_r < 0.5)).sum()
    reversed_ = ((share_n < 0.5) & (share_r > 0.5)).sum()
    return {"pairs": len(P), "errors_nat": float(Nn.sum()), "errors_rev": float(Nr.sum()),
            "share_to_bctr_frequent_nat": float(sn), "share_to_bctr_frequent_rev": float(sr),
            "symmetric_expectation": float((Nn * p0).sum() / Nn.sum()),
            "diff_nat_minus_rev": float(sn - sr),
            "diff_ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
            "mean_pair_share_nat": float(share_n.mean()), "mean_pair_share_rev": float(share_r.mean()),
            "wilcoxon_p": float(wilcoxon(share_n, share_r).pvalue),
            "pairs_majority_flipped_to_rev_direction": int(flipped), "pairs_majority_reversed_other_way": int(reversed_)}


def joint_fit(rows_by_cond, comp, ink, bctr, reps=1000, seed=0):
    """logit P(r->f) = logit p0 + beta*log((cnt_f+1)/(cnt_r+1)) + eta_c*dcomp + eta_i*dink (+ alpha), r = BCTR-rarer;
    rows pooled over orders (training counts flip sign, complexity does not)."""
    from scipy.optimize import minimize
    X, K, N, OFF = [], [], [], []
    zc = (comp - np.nanmean(comp)) / np.nanstd(comp)
    zi = (ink - ink.mean()) / ink.std()
    for rows in rows_by_cond:
        for r in rows:
            a, b = r["a"], r["b"]
            if bctr[a] == bctr[b] or np.isnan(zc[a]) or np.isnan(zc[b]):
                continue
            if bctr[a] < bctr[b]:  # a = rarer in BCTR
                k, cr, cf, oa, ob = r["n_ab"], r["cnt_a"], r["cnt_b"], r["occ_a"], r["occ_b"]
                rr, ff = a, b
            else:
                k, cr, cf, oa, ob = r["n_ba"], r["cnt_b"], r["cnt_a"], r["occ_b"], r["occ_a"]
                rr, ff = b, a
            X.append([math.log((cf + 1) / (cr + 1)), zc[rr] - zc[ff], zi[rr] - zi[ff], 1.0])
            K.append(k)
            N.append(r["n_ab"] + r["n_ba"])
            p0 = oa / (oa + ob)
            OFF.append(math.log(p0) - math.log1p(-p0))
    X, K, N, OFF = map(np.asarray, (X, K, N, OFF))

    def fit(i, cols):
        x, k, n, off = X[i][:, cols], K[i], N[i], OFF[i]

        def nll(w):
            z = off + x @ w
            return float(np.sum(k * np.logaddexp(0, -z) + (n - k) * np.logaddexp(0, z)))
        return minimize(nll, np.zeros(len(cols)), method="BFGS").x
    out = {"rows": int(len(K))}
    rng = np.random.default_rng(seed)
    for name, cols in (("freq_only", [0]), ("freq+complexity", [0, 1, 2]), ("freq+complexity+intercept", [0, 1, 2, 3])):
        est = fit(np.arange(len(K)), cols)
        boots = np.array([fit(rng.integers(0, len(K), len(K)), cols) for _ in range(reps)])
        labels = ["beta_freq", "eta_ids_components", "eta_ink", "alpha"]
        out[name] = {labels[c]: {"estimate": float(est[j]), "ci95": [float(np.percentile(boots[:, j], 2.5)),
                                                                   float(np.percentile(boots[:, j], 97.5))]}
                     for j, c in enumerate(cols)}
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True)
    p.add_argument("--nat", nargs="+", required=True, type=Path)
    p.add_argument("--rev", nargs="+", required=True, type=Path)
    p.add_argument("--perm", nargs="*", default=[], type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--reps", type=int, default=1000)
    args = p.parse_args()
    classes = (Path(args.data) / "classes.txt").read_text(encoding="utf-8")
    bctr = np.array([a0.COUNT_OF.get(c, 0) for c in classes], float)
    comp, ink = complexity(classes, args.data)
    res = {"complexity": {"ids_components_vs_log_bctr_spearman": c3.spearman(np.log1p(bctr[~np.isnan(comp)]),
                                                                              comp[~np.isnan(comp)]),
                          "ink_vs_log_bctr_spearman": c3.spearman(np.log1p(bctr), ink),
                          "ids_missing": int(np.isnan(comp).sum())}}
    pooled_n, pooled_rows = {}, {}
    for split in ("test", "val"):
        for cond, runs in (("nat", args.nat), ("rev", args.rev), ("perm", args.perm)):
            if not runs:
                continue
            pooled = collections.Counter()
            for run in runs:
                pred, labels, counts = load_run(run, split)
                occ = np.bincount(labels, minlength=len(classes))
                n = confusion(pred, labels)
                pooled.update(n)
                rows = rows_from(n, occ, counts, classes)
                res[f"{split}_{run.name}"] = {"order": cond, "top1": float(100 * (pred == labels).mean()),
                                             "spearman_count_vs_bctr": c3.spearman(np.log1p(counts), np.log1p(bctr)),
                                             **per_model(rows)}
                t = res[f"{split}_{run.name}"]["all"]
                print(split, run.name, f"top1={res[f'{split}_{run.name}']['top1']:.2f}",
                      f"pairs={t['pairs']}", f"T/E={t['toward_more_frequent']['ratio']:.2f}",
                      f"z={t['toward_more_frequent']['z']:.1f}",
                      f"beta={t['beta']['estimate']:.2f} {t['beta']['ci95']}", flush=True)
            pooled_n[(split, cond)] = pooled
            pooled_rows[(split, cond)] = rows_from(pooled, occ, counts, classes)
            res[f"{split}_{cond}_pooled"] = per_model(pooled_rows[(split, cond)])
        occ = np.bincount(load_run(args.nat[0], split)[1], minlength=len(classes))
        res[f"{split}_flip_test"] = flip_test(pooled_n[(split, "nat")], pooled_n[(split, "rev")], occ, bctr, args.reps)
        print(split, "flip test", json.dumps(res[f"{split}_flip_test"]), flush=True)
        res[f"{split}_joint"] = joint_fit([pooled_rows[(split, "nat")], pooled_rows[(split, "rev")]], comp, ink, bctr,
                                          args.reps)
        print(split, "joint", json.dumps(res[f"{split}_joint"]), flush=True)
    # pre-registered criteria (test)
    f1 = {}
    for run in args.rev:
        r = res[f"test_{run.name}"]["all"]
        t = r["toward_more_frequent"]
        f1[run.name] = bool(t["ratio"] > 1.5 and t["z"] > 3 and (r["beta"]["ci95"][0] > 0 or r["beta"]["ci95"][1] < 0))
    ft = res["test_flip_test"]
    f2 = bool(ft.get("share_to_bctr_frequent_nat", 0) > 0.5 and ft.get("share_to_bctr_frequent_rev", 1) < 0.5
              and ft["diff_ci95"][0] > 0)
    res["criteria"] = {"F1_per_rev_model": f1, "F1": bool(f1) and all(f1.values()), "F2": f2}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "c7_results.json").write_text(json.dumps(res, indent=1))
    print("CRITERIA", json.dumps(res["criteria"]))


if __name__ == "__main__":
    main()
