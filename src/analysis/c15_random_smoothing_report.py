"""C-15: random-class smoothing control on HWDB-LT.

rgls smooths toward random classes with (nearly) the same training counts as the ugls neighbours but no visual
similarity (train_hwdb_classifier.py randnb / --loss rgls). Each run is read from reevaluate_hwdb.py dumps (lt100 counts); per
run the post-hoc strength is selected on val for the balanced and the natural metric (max; ties -> tau closest to 0) and
test is read once at the default and at the two selected strengths. Paired class-stratified bootstrap per seed
(evaluation/reevaluate_hwdb.py (bootstrap_diff)). Pre-registered (docs/experiments.md, "C-13 至 C-17 预注册"), seeds 0-3, test:
  R1  the default-setting gain is a frequency effect: mean_s[bal(rgls) - bal(ce)] >= 0.5 * mean_s[bal(ugls) - bal(ce)]
  R2  the deficit after correction comes from smoothing itself: at tau*(bal), bal(rgls) - bal(ce) < 0 for every seed
      and its mean is >= 0.5 * the mean deficit of ugls.

    python src/analysis/c15_random_smoothing_report.py --dumps results/C-15_rand_smooth/dumps results/C-01b_hwdb_seeds/dumps_all \
        --out results/C-15_rand_smooth
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation import reevaluate_hwdb as c1  # noqa: E402

LOSSES = ("rgls", "ce", "ugls", "sals", "gls", "csls", "ls")
SEEDS = (0, 1, 2, 3)


def load(roots, name):
    for root in roots:
        if (root / name / "meta.json").exists():
            return root / name
    raise FileNotFoundError(name)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dumps", nargs="+", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--weights", default=str(c1.EXP / "weights.npz"))
    args = p.parse_args()
    lt = [np.load(r / "lt_counts.npy") for r in args.dumps]
    assert all((x == lt[0]).all() for x in lt), "dump roots must use the same LT training counts"
    lt_counts = lt[0]
    tw = c1.target_weights(np.load(args.weights), lt_counts)
    n_cls = len(lt_counts)
    k0 = int(np.argmin(np.abs(c1.TAUS)))
    runs, preds = {}, {}
    for loss in LOSSES:
        for s in SEEDS:
            name = f"{loss}_s{s}"
            d = load(args.dumps, name)
            meta = json.loads((d / "meta.json").read_text())
            assert meta["tau_train"] == 0.0, name
            r = {}
            for split in ("val", "test"):
                z = np.load(d / f"{split}.npz")
                labels = z["labels"].astype(np.int64)
                sc = c1.scores(c1.per_class_acc(z["grid"], labels, n_cls), tw, lt_counts)
                r[split] = {m: np.asarray(sc[m]) for m in ("bal", "nat")}
                if split == "test":
                    preds[name] = (z["grid"], labels)
            sel = {}
            for m in ("bal", "nat"):
                v = r["val"][m]
                best = np.flatnonzero(v == v.max())
                sel[m] = int(best[np.argmin(np.abs(c1.TAUS[best]))])
            runs[name] = {"tau_bal": float(c1.TAUS[sel["bal"]]), "tau_nat": float(c1.TAUS[sel["nat"]]),
                          "k": {"default": k0, "tau*(bal)": sel["bal"], "tau*(nat)": sel["nat"]},
                          "test": {cfg: {m: float(r["test"][m][k]) for m in ("bal", "nat")}
                                   for cfg, k in (("default", k0), ("tau*(bal)", sel["bal"]), ("tau*(nat)", sel["nat"]))}}

    def diff(a, b, cfg, metric):
        ga, labels = preds[a]
        gb, _ = preds[b]
        return c1.bootstrap_diff(ga[:, runs[a]["k"][cfg]], gb[:, runs[b]["k"][cfg]], labels, tw[metric])

    comps = {}
    for a, b in (("rgls", "ce"), ("ugls", "ce"), ("rgls", "ugls"), ("sals", "ce"), ("gls", "ce"), ("csls", "ce"),
                 ("ls", "ce")):
        for cfg, metric in (("default", "bal"), ("tau*(bal)", "bal"), ("default", "nat"), ("tau*(nat)", "nat")):
            comps[f"{a}-{b} {cfg} {metric}"] = {s: diff(f"{a}_s{s}", f"{b}_s{s}", cfg, metric) for s in SEEDS}

    def mean(key):
        return float(np.mean([v[0] for v in comps[key].values()]))
    r_def, u_def = mean("rgls-ce default bal"), mean("ugls-ce default bal")
    r_tun, u_tun = mean("rgls-ce tau*(bal) bal"), mean("ugls-ce tau*(bal) bal")
    criteria = {"R1_default_gain_reproduced": bool(u_def > 0 and r_def >= 0.5 * u_def),
                "R2_tuned_deficit_from_smoothing": bool(all(v[0] < 0 for v in comps["rgls-ce tau*(bal) bal"].values())
                                                        and u_tun < 0 and r_tun <= 0.5 * u_tun),
                "means": {"rgls-ce default": r_def, "ugls-ce default": u_def, "rgls-ce tuned": r_tun,
                          "ugls-ce tuned": u_tun}}
    nb_summary = Path(args.out / "random_neighbors_k10_summary.json")
    out = {"runs": runs, "comparisons": {k: {str(s): v for s, v in d.items()} for k, d in comps.items()},
           "criteria": criteria,
           "neighbour_table": json.loads(nb_summary.read_text()) if nb_summary.exists() else None}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "c15_results.json").write_text(json.dumps(out, indent=1, default=float))
    L = ["# C-15 random-class smoothing control (HWDB-LT, test; mean of seeds 0-3)", "",
         "| method | default bal | tau*(bal) bal | default nat | tau*(nat) nat | tau*(bal) / tau*(nat) |",
         "|---|---:|---:|---:|---:|---|"]
    for loss in LOSSES:
        rs = [runs[f"{loss}_s{s}"] for s in SEEDS]
        L.append(f"| {loss} | {np.mean([r['test']['default']['bal'] for r in rs]):.2f} | "
                 f"{np.mean([r['test']['tau*(bal)']['bal'] for r in rs]):.2f} | "
                 f"{np.mean([r['test']['default']['nat'] for r in rs]):.2f} | "
                 f"{np.mean([r['test']['tau*(nat)']['nat'] for r in rs]):.2f} | "
                 + " ".join(f"{r['tau_bal']:.2f}/{r['tau_nat']:.2f}" for r in rs) + " |")
    L += ["", "## Paired differences (pp, point [95% CI] per seed)", ""]
    for k, d in comps.items():
        L.append(f"- {k}: " + "; ".join(f"s{s}: {v[0]:+.2f} [{v[1]:+.2f}, {v[2]:+.2f}]" for s, v in d.items()))
    L += ["", f"**Criteria: {criteria}**"]
    (args.out / "report.md").write_text("\n".join(L) + "\n")
    print("CRITERIA", json.dumps(criteria))


if __name__ == "__main__":
    main()
