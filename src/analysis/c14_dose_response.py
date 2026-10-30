"""C-14: HWDB dose-response of the confusion direction.

Training counts move from the natural HWDB-LT order (alpha = 0: A1-02 / C-01b ce models) through the dose splits
(alpha = 0.25, 0.5, 0.75; prepare_hwdb.py dose) to the reversed order (alpha = 1: C-07 lt100rev). On a fixed pair set
-- the C-07 flip-test pairs (>= 3 errors under both alpha = 0 and alpha = 1, seeds pooled) -- S(alpha) is the share of
errors that go to the BCTR-frequent member; S_sym is its expectation under symmetric confusion (ICDAR 2013 exposure is
nearly equal). Pre-registered (docs/experiments.md, "C-13 至 C-17 预注册"):
  D1  S(alpha) strictly decreases over alpha = 0, 0.25, 0.5, 0.75, 1 (test, seeds pooled);
  D2  at equal training counts the bias is mostly gone: R = (S(0.5) - S_sym) / (S(0) - S_sym) < 0.25.
Per model: top-1, T/E, z and beta with the model's own training counts (C-03b; beta is undefined at alpha = 0.5).

    python src/analysis/c14_dose_response.py --data data/hwdb --cond 0=results/C-01b_hwdb_seeds/dumps_all/ce_s{0,1,2,3} \
        --cond 0.25=results/C-14_dose/dumps_025/dose025_ce_s{0,1} ... --cond 1=results/C-07_freq_flip/dumps_rev/rev_ce_s{0..3} \
        --out results/C-14_dose
"""
import argparse
import collections
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import prior_correction as a0  # noqa: E402
from analysis import c07_flip_direction as c7  # noqa: E402


def parse_conds(items):
    conds = collections.OrderedDict()
    alpha = None
    for it in items:
        if "=" in it:
            a, path = it.split("=", 1)
            alpha = float(a)
            conds.setdefault(alpha, []).append(Path(path))
        else:  # shell-expanded continuation of the previous --cond
            conds[alpha].append(Path(it))
    return collections.OrderedDict(sorted(conds.items()))


def shares(pooled, P, occ, idx=None):
    sel = P if idx is None else [P[i] for i in idx]
    k = np.array([pooled[(r, f)] for r, f in sel], float)
    N = k + np.array([pooled[(f, r)] for r, f in sel], float)
    p0 = np.array([occ[r] / (occ[r] + occ[f]) for r, f in sel])
    return k.sum() / N.sum(), (N * p0).sum() / N.sum(), N.sum()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True)
    p.add_argument("--cond", nargs="+", action="extend", required=True, help="alpha=dump_dir (repeatable)")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--reps", type=int, default=1000)
    args = p.parse_args()
    conds = parse_conds(args.cond)
    assert 0.0 in conds and 1.0 in conds, "need alpha = 0 and alpha = 1"
    classes = (Path(args.data) / "classes.txt").read_text(encoding="utf-8")
    bctr = np.array([a0.COUNT_OF.get(c, 0) for c in classes], float)
    res = {"conditions": {str(a): [str(r) for r in runs] for a, runs in conds.items()}}
    rng = np.random.default_rng(0)
    for split in ("test", "val"):
        pooled, per = {}, {}
        occ = None
        for alpha, runs in conds.items():
            pc = collections.Counter()
            for run in runs:
                pred, labels, counts = c7.load_run(run, split)
                o = np.bincount(labels, minlength=len(classes))
                occ = o if occ is None else occ
                assert (o == occ).all(), "all models must share the evaluation set"
                n = c7.confusion(pred, labels)
                pc.update(n)
                # own-count direction statistics need unequal counts (alpha = 0.5 gives every class the same count)
                m = c7.per_model(c7.rows_from(n, occ, counts, classes))["all"] if np.ptp(counts) > 0 else {}
                t = m.get("toward_more_frequent", {})
                per[run.name] = {"alpha": alpha, "top1": float(100 * (pred == labels).mean()),
                                 "T/E": t.get("ratio"), "z": t.get("z"), "pairs": m.get("pairs"),
                                 "beta": m.get("beta", {}).get("estimate"), "beta_ci95": m.get("beta", {}).get("ci95")}
            pooled[alpha] = pc
        # the C-07 flip-test pair set: >= 3 errors under alpha = 0 and under alpha = 1
        P = []
        for a, b in {tuple(sorted(k)) for k in pooled[0.0]}:
            if bctr[a] == bctr[b]:
                continue
            r, f = (a, b) if bctr[a] < bctr[b] else (b, a)
            if pooled[0.0][(r, f)] + pooled[0.0][(f, r)] >= 3 and pooled[1.0][(r, f)] + pooled[1.0][(f, r)] >= 3:
                P.append((r, f))
        P.sort()
        S = {a: shares(pooled[a], P, occ) for a in conds}
        boots = collections.defaultdict(list)
        for _ in range(args.reps):
            i = rng.integers(0, len(P), len(P))
            s = {a: shares(pooled[a], P, occ, i) for a in conds}
            for a in conds:
                boots[a].append(s[a][0])
            if 0.5 in s:
                boots["R"].append((s[0.5][0] - s[0.5][1]) / (s[0.0][0] - s[0.0][1]))
        ci = {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in boots.items()}
        R = (S[0.5][0] - S[0.5][1]) / (S[0.0][0] - S[0.0][1]) if 0.5 in S else None
        res[split] = {"pairs": len(P), "per_model": per, "R_residual_at_0.5": R, "R_ci95": ci.get("R"),
                      "share_to_bctr_frequent": {str(a): {"share": float(S[a][0]), "symmetric": float(S[a][1]),
                                                          "errors": float(S[a][2]), "ci95": ci[a]} for a in conds}}
        print(split, {a: round(S[a][0], 3) for a in conds}, "R", None if R is None else round(R, 3), flush=True)
    t = res["test"]["share_to_bctr_frequent"]
    seq = [t[str(a)]["share"] for a in conds]
    res["criteria"] = {"D1_monotone": bool(all(x > y for x, y in zip(seq, seq[1:]))),
                       "D2_residual_lt_0.25": bool(res["test"]["R_residual_at_0.5"] is not None
                                                   and res["test"]["R_residual_at_0.5"] < 0.25)}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "c14_results.json").write_text(json.dumps(res, indent=1, default=float))
    lines = ["# C-14 HWDB dose-response (share of errors toward the BCTR-frequent member)", ""]
    for split in ("test", "val"):
        s = res[split]
        lines += [f"## {split} ({s['pairs']} pairs)", "", "| alpha | share [95% CI] | symmetric | errors |", "|---|---|---:|---:|"]
        for a, v in s["share_to_bctr_frequent"].items():
            lines.append(f"| {a} | {v['share']:.3f} [{v['ci95'][0]:.3f}, {v['ci95'][1]:.3f}] | {v['symmetric']:.3f} | "
                         f"{v['errors']:.0f} |")
        lines += ["", f"- residual R at alpha 0.5: {s['R_residual_at_0.5']} (CI {s['R_ci95']})", "",
                  "| model | alpha | top-1 | T/E | z | beta (own counts) |", "|---|---:|---:|---:|---:|---|"]
        for name, v in s["per_model"].items():
            lines.append(f"| {name} | {v['alpha']} | {v['top1']:.2f} | {v['T/E']} | {v['z']} | {v['beta']} {v['beta_ci95']} |")
        lines.append("")
    lines.append(f"**Criteria (test): {res['criteria']}**")
    (args.out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("CRITERIA", json.dumps(res["criteria"]))


if __name__ == "__main__":
    main()
