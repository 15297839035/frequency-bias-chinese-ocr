"""C-02b: C-02 evaluation extended with seed 3 (replication), reusing evaluate_bctr.py unchanged.

Same configs, same val selection rule (constraint from the same-seed c0 default), one test read-out.
Reports per-seed comparisons for seeds 1-3, the pre-registered verdict on seeds {1, 2} (unchanged
rule), whether seed 3 agrees with it, and 3-seed means.

    python src/evaluation/evaluate_bctr_seeds.py --dumps results/C-02_bctr_la/dumps --out results/C-02_bctr_la
Protocol: docs/experiments.md, entry "C-01b / C-02b 预注册".
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import prior_correction as a0  # noqa: E402
from evaluation import evaluate_bctr as c2  # noqa: E402

SEEDS = (1, 2, 3)
ARMS = {"c0": 0.0, "la025": 0.25, "la1": 1.0}


def verdict(rd, ad):
    if all(x >= 0.5 for x in rd) and np.mean(ad) >= -0.1:
        return "training-side adds value"
    if abs(np.mean(rd)) < 0.5:
        return "equivalent"
    if all(x <= -0.5 for x in rd):
        return "training-side worse"
    return "inconclusive"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dumps", required=True)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--workers", type=int, default=9)
    args = p.parse_args()
    runs = {f"{arm}_s{s}": (tau, s) for arm, tau in ARMS.items() for s in SEEDS}
    with ProcessPoolExecutor(args.workers) as pool:
        res = dict(pool.map(c2.job, [(args.dumps, tag, tau) for tag, (tau, _) in runs.items()]))
    for tag in res:
        assert res[tag]["test"]["ids"] == res["c0_s1"]["test"]["ids"], tag
        assert res[tag]["val"]["ids"] == res["c0_s1"]["val"]["ids"], tag
    rare = res["c0_s1"]["test"]["rare"]
    selected, out = {}, {"runs": {}, "comparisons": {}}
    for tag, (tau, seed) in runs.items():
        val, test = res[tag]["val"]["rows"], res[tag]["test"]["rows"]
        limit = res[f"c0_s{seed}"]["val"]["rows"][0]["acc"] - 0.05
        ok = [i for i, r in enumerate(val) if r["acc"] >= limit]
        key = lambda i: (-val[i]["rare_correct"], abs(val[i]["cfg"]["tau"]) if val[i]["cfg"] else 0.0,
                         val[i]["cfg"]["gamma"] if val[i]["cfg"] else 0.0)
        k = sorted(ok, key=key)[0] if ok else int(np.argmax([r["acc"] for r in val]))
        selected[tag] = k
        out["runs"][tag] = {"tau_train": tau, "seed": seed, "feasible": bool(ok), "selected_cfg": val[k]["cfg"],
                            "test_default": c2.strip(test[0]), "test_selected": c2.strip(test[k])}
    lines = []
    for arm in ("la025", "la1"):
        per = {}
        for s in SEEDS:
            vt = res[f"{arm}_s{s}"]["test"]["rows"][selected[f"{arm}_s{s}"]]["vec"]
            vc = res[f"c0_s{s}"]["test"]["rows"][selected[f"c0_s{s}"]]["vec"]
            per[s] = {"rare_diff": c2.bootstrap(vt, vc, rare), "acc_diff": c2.bootstrap(vt, vc, np.ones_like(rare))}
        rd12 = [per[s]["rare_diff"][0] for s in (1, 2)]
        ad12 = [per[s]["acc_diff"][0] for s in (1, 2)]
        v12 = verdict(rd12, ad12)
        sign12 = np.sign(np.mean(rd12))
        out["comparisons"][arm] = {"per_seed": per, "verdict_seeds12_preregistered": v12,
                                   "seed3_same_direction": bool(np.sign(per[3]["rare_diff"][0]) == sign12),
                                   "mean3_rare_diff": float(np.mean([per[s]["rare_diff"][0] for s in SEEDS])),
                                   "mean3_acc_diff": float(np.mean([per[s]["acc_diff"][0] for s in SEEDS]))}
        c = out["comparisons"][arm]
        lines.append(f"- {arm} vs c0: seeds 1-2 verdict **{v12}**; seed 3 same direction: {c['seed3_same_direction']}; "
                     f"3-seed mean rare {c['mean3_rare_diff']:+.3f}, acc {c['mean3_acc_diff']:+.3f}")
        for s in SEEDS:
            r, a = per[s]["rare_diff"], per[s]["acc_diff"]
            lines.append(f"  - seed {s}: rare {r[0]:+.3f} [{r[1]:+.3f}, {r[2]:+.3f}], acc {a[0]:+.3f} [{a[1]:+.3f}, {a[2]:+.3f}]")
    table = ["| run | selected (tau_eff, gamma) | test default acc / rare | test selected acc / rare |", "|---|---|---|---|"]
    for tag, r in out["runs"].items():
        cfg = r["selected_cfg"]
        sel = "default" if cfg is None else f"{cfg['tau_eff']}, {cfg['gamma']}"
        table.append(f"| {tag} | {sel} | "
                     f"{r['test_default']['acc']:.3f} / {r['test_default']['rare_acc']:.3f} | "
                     f"{r['test_selected']['acc']:.3f} / {r['test_selected']['rare_acc']:.3f} |")
    (args.out / "c2b_results.json").write_text(json.dumps(out, indent=1, default=float))
    (args.out / "report_3seeds.md").write_text("\n".join(table + [""] + lines) + "\n")
    print("\n".join(table + [""] + lines))


if __name__ == "__main__":
    main()
