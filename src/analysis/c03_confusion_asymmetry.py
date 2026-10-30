"""C-03b (exploratory): exposure-controlled test of whether confusion direction follows character frequency.

The pre-registered C-03 (c) statistic, log[(n_ab+0.5)/occ_a] - log[(n_ba+0.5)/occ_b], is biased: the +0.5
inflates the rate of the rarer character, and occurrence counts track training frequency. Here:
  1. symmetric null: n_ab ~ Binomial(N, occ_a/(occ_a+occ_b)); observed vs expected errors toward the
     more frequent character (T/E, z)
  2. conditional binomial regression, no intercept: logit P(a->b | N) = logit(p0) + beta * log((cnt_b+1)/(cnt_a+1))
  3. null distribution of the original (c) Spearman rho under the symmetric null
Protocol: docs/experiments.md, entry "C-03b 探索性补充".

    python src/analysis/c03_confusion_asymmetry.py --dumps results/C-03_analysis/dumps --out results/C-03_analysis
"""
import argparse
import collections
import json
import math
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import prior_correction as a0  # noqa: E402
from analysis import c03_confusion_analysis as c3  # noqa: E402


def pairs_from(refs, hyps, min_errors=3):
    occ = collections.Counter(ch for ref in refs for ch in ref)
    n = collections.Counter()
    for ref, hyp in zip(refs, hyps):
        for a, b in c3.substitutions(ref, hyp):
            if a0.is_han(a) and a0.is_han(b):
                n[(a, b)] += 1
    rows = []
    for a, b in {tuple(sorted(k)) for k in n}:
        N = n[(a, b)] + n[(b, a)]
        if N < min_errors or occ[a] == 0 or occ[b] == 0:
            continue
        rows.append(dict(a=a, b=b, n_ab=n[(a, b)], n_ba=n[(b, a)], occ_a=occ[a], occ_b=occ[b],
                         cnt_a=c3.count(a), cnt_b=c3.count(b), look=c3.lookalike(a, b)))
    return rows


def oriented(rows):
    """Re-orient so that b is the more frequent character in training; drop ties."""
    out = []
    for r in rows:
        if r["cnt_a"] == r["cnt_b"]:
            continue
        if r["cnt_a"] > r["cnt_b"]:
            r = dict(a=r["b"], b=r["a"], n_ab=r["n_ba"], n_ba=r["n_ab"], occ_a=r["occ_b"], occ_b=r["occ_a"],
                     cnt_a=r["cnt_b"], cnt_b=r["cnt_a"], look=r["look"])
        out.append(r)
    return out


def arrays(rows):
    k = np.array([r["n_ab"] for r in rows], float)  # errors toward the more frequent char
    N = np.array([r["n_ab"] + r["n_ba"] for r in rows], float)
    p0 = np.array([r["occ_a"] / (r["occ_a"] + r["occ_b"]) for r in rows])
    x = np.array([math.log((r["cnt_b"] + 1) / (r["cnt_a"] + 1)) for r in rows])
    return k, N, p0, x


def fit_beta(k, N, p0, x):
    """MLE of beta on [-5, 5] (bounded; numerically stable log-sigmoid)."""
    from scipy.optimize import minimize_scalar
    off = np.log(p0) - np.log1p(-p0)

    def nll(beta):
        z = off + beta * x
        return float(np.sum(k * np.logaddexp(0, -z) + (N - k) * np.logaddexp(0, z)))
    return float(minimize_scalar(nll, bounds=(-5, 5), method="bounded", options={"xatol": 1e-6}).x)


def original_rho(rows, n_ab, n_ba):
    y = np.array([math.log((u + 0.5) / r["occ_a"]) - math.log((v + 0.5) / r["occ_b"]) for r, u, v in zip(rows, n_ab, n_ba)])
    x = np.array([math.log((r["cnt_b"] + 1) / (r["cnt_a"] + 1)) for r in rows])
    return c3.spearman(x, y)


def analyse(rows, reps=1000, seed=0):
    rng = np.random.default_rng(seed)
    res = {"pairs": len(rows)}
    if len(rows) < 10:
        return res
    # (3) original statistic on the unoriented (sorted) pairs, observed and under the symmetric null
    obs_rho = original_rho(rows, [r["n_ab"] for r in rows], [r["n_ba"] for r in rows])
    N_u = np.array([r["n_ab"] + r["n_ba"] for r in rows])
    p_u = np.array([r["occ_a"] / (r["occ_a"] + r["occ_b"]) for r in rows])
    null = []
    for _ in range(reps):
        sim = rng.binomial(N_u, p_u)
        null.append(original_rho(rows, sim, N_u - sim))
    res["original_rho"] = {"observed": obs_rho, "null_mean": float(np.mean(null)),
                           "null_ci95": [float(np.percentile(null, 2.5)), float(np.percentile(null, 97.5))]}
    # (1) symmetric null on oriented pairs
    o = oriented(rows)
    k, N, p0, x = arrays(o)
    T, E = float(k.sum()), float((N * p0).sum())
    var = float((N * p0 * (1 - p0)).sum())
    res["toward_more_frequent"] = {"pairs": len(o), "errors": float(N.sum()), "observed": T, "expected_symmetric": E,
                                   "ratio": T / E, "z": (T - E) / math.sqrt(var),
                                   "pct_observed": 100 * T / N.sum(), "pct_expected": 100 * E / N.sum()}
    # (2) conditional binomial regression, bootstrap over pairs
    beta = fit_beta(k, N, p0, x)
    boots = []
    for _ in range(reps):
        i = rng.integers(0, len(o), len(o))
        boots.append(fit_beta(k[i], N[i], p0[i], x[i]))
    res["beta"] = {"estimate": beta, "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]}
    return res


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dumps", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()
    a0.DUMPS = args.dumps
    out = {}
    for model in ("gtc", "b"):
        for split in ("val", "test"):
            data = {dom: a0.load(model, split, dom) for dom in a0.DOMAINS}
            refs = [t for dom in a0.DOMAINS for t in data[dom]["targets"]]
            for name, cfg in (("B0", None), ("G*", c3.GSTAR)):
                hyps = [s for dom in a0.DOMAINS for s in a0.strings(data[dom], a0.decode(data[dom], cfg))]
                rows = pairs_from(refs, hyps)
                out[f"{model}_{split}_{name}"] = {grp: analyse([r for r in rows if sel(r)])
                                                  for grp, sel in (("all", lambda r: True), ("lookalike", lambda r: r["look"]),
                                                                   ("other", lambda r: not r["look"]))}
                print("done", model, split, name, flush=True)
    (args.out / "c3b_asymmetry.json").write_text(json.dumps(out, indent=1))
    for key, v in out.items():
        for grp, r in v.items():
            if "beta" not in r:
                print(key, grp, r)
                continue
            t = r["toward_more_frequent"]
            print(f"{key:16s} {grp:9s} pairs={t['pairs']:4d} T/E={t['ratio']:.3f} z={t['z']:.1f} "
                  f"obs%={t['pct_observed']:.1f} exp%={t['pct_expected']:.1f} beta={r['beta']['estimate']:.3f} "
                  f"[{r['beta']['ci95'][0]:.3f},{r['beta']['ci95'][1]:.3f}] rho_obs={r['original_rho']['observed']:.3f} "
                  f"rho_null={r['original_rho']['null_mean']:.3f} [{r['original_rho']['null_ci95'][0]:.3f},"
                  f"{r['original_rho']['null_ci95'][1]:.3f}]")


if __name__ == "__main__":
    main()
