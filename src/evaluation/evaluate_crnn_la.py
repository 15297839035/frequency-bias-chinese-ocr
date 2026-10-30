"""C-11 evaluation: CRNN trained from scratch with conditional LA (tau_train 0.25 / 1) vs the C-05 CRNN (tau_train 0).

Same procedure and verdict rules as C-02 (src/evaluation/evaluate_bctr.py, whose functions are reused unchanged): post-hoc
G-LA grid over tau_eff (tau_post = tau_eff - tau_train) x gamma plus the default; per run, the val-selected config
maximises rare-line accuracy subject to pooled val accuracy >= the same-seed control's default - 0.05pp; test is read
once; selected treatment vs selected control per seed. Strict normalisation (C-05's main protocol).

    python src/evaluation/evaluate_crnn_la.py --out results/C-11_crnn_la
Protocol: docs/experiments.md, entry "C-07 至 C-11 预注册" (C-11).
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import prior_correction as a0  # noqa: E402
from evaluation.evaluate_bctr import bootstrap, job, strip  # noqa: E402

EXP = a0.REPO / "results"
RUNS = {"crnn_s1": (0.0, 1, EXP / "C-05_crnn/dumps"), "crnn_s2": (0.0, 2, EXP / "C-05_crnn/dumps"),
        "la025_s1": (0.25, 1, EXP / "C-11_crnn_la/dumps"), "la025_s2": (0.25, 2, EXP / "C-11_crnn_la/dumps"),
        "la1_s1": (1.0, 1, EXP / "C-11_crnn_la/dumps"), "la1_s2": (1.0, 2, EXP / "C-11_crnn_la/dumps")}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--workers", type=int, default=6)
    args = p.parse_args()
    jobs = [(str(d), tag, tau) for tag, (tau, _, d) in RUNS.items()]
    with ProcessPoolExecutor(args.workers) as pool:
        res = dict(pool.map(job, jobs))
    for tag in res:
        assert res[tag]["test"]["ids"] == res["crnn_s1"]["test"]["ids"], f"test order differs: {tag}"
        assert res[tag]["val"]["ids"] == res["crnn_s1"]["val"]["ids"], f"val order differs: {tag}"
    rare_test = res["crnn_s1"]["test"]["rare"]
    report, selected = {"runs": {}, "judgement": {}}, {}
    for tag, (tau_train, seed, _) in RUNS.items():
        val, test = res[tag]["val"]["rows"], res[tag]["test"]["rows"]
        c0_val_acc = res[f"crnn_s{seed}"]["val"]["rows"][0]["acc"]
        ok = [i for i, r in enumerate(val) if r["acc"] >= c0_val_acc - 0.05]
        if ok:
            k = sorted(ok, key=lambda i: (-val[i]["rare_correct"], abs(val[i]["cfg"]["tau"]) if val[i]["cfg"] else 0.0,
                                          val[i]["cfg"]["gamma"] if val[i]["cfg"] else 0.0))[0]
            feasible = True
        else:
            k, feasible = int(np.argmax([r["acc"] for r in val])), False
        selected[tag] = k
        report["runs"][tag] = {"tau_train": tau_train, "seed": seed, "feasible": feasible,
                               "constraint_val_acc": c0_val_acc - 0.05, "selected_cfg": val[k]["cfg"],
                               "val_selected": strip(val[k]), "test_default": strip(test[0]), "test_selected": strip(test[k]),
                               "val_frontier": [strip(r) for r in val], "test_frontier_descriptive": [strip(r) for r in test]}
    for arm in ("la025", "la1"):
        per_seed = {}
        for seed in (1, 2):
            t, c = res[f"{arm}_s{seed}"]["test"]["rows"], res[f"crnn_s{seed}"]["test"]["rows"]
            vt, vc = t[selected[f"{arm}_s{seed}"]]["vec"], c[selected[f"crnn_s{seed}"]]["vec"]
            fixed, broken = int((vt & ~vc & rare_test).sum()), int((~vt & vc & rare_test).sum())
            per_seed[seed] = {"rare_diff": bootstrap(vt, vc, rare_test), "acc_diff": bootstrap(vt, vc, np.ones_like(rare_test)),
                              "rare_fixed": fixed, "rare_broken": broken, "rare_mcnemar_p": a0.exact_mcnemar(fixed, broken)}
        rd = [per_seed[s]["rare_diff"][0] for s in (1, 2)]
        ad = [per_seed[s]["acc_diff"][0] for s in (1, 2)]
        if all(x >= 0.5 for x in rd) and np.mean(ad) >= -0.1:
            verdict = "training-side adds value"
        elif abs(np.mean(rd)) < 0.5:
            verdict = "equivalent"
        elif all(x <= -0.5 for x in rd):
            verdict = "training-side worse"
        else:
            verdict = "inconclusive"
        report["judgement"][arm] = {"per_seed": per_seed, "verdict": verdict}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "c11_results.json").write_text(json.dumps(report, indent=1, default=float))
    lines = ["| run | tau_train | selected (tau_eff, gamma) | val acc / rare | test default acc / rare | test selected acc / rare |",
             "|---|---:|---|---|---|---|"]
    for tag, r in report["runs"].items():
        cfg = r["selected_cfg"]
        sel = "default" if cfg is None else f"{cfg['tau_eff']}, {cfg['gamma']}"
        lines.append(f"| {tag} | {r['tau_train']} | {sel}{'' if r['feasible'] else ' (infeasible)'} | "
                     f"{r['val_selected']['acc']:.3f} / {r['val_selected']['rare_acc']:.3f} | "
                     f"{r['test_default']['acc']:.3f} / {r['test_default']['rare_acc']:.3f} | "
                     f"{r['test_selected']['acc']:.3f} / {r['test_selected']['rare_acc']:.3f} |")
    lines.append("")
    for arm, j in report["judgement"].items():
        lines.append(f"- {arm} vs crnn (selected vs selected): verdict **{j['verdict']}**")
        for s, v in j["per_seed"].items():
            lines.append(f"  - seed {s}: rare {v['rare_diff'][0]:+.3f} [{v['rare_diff'][1]:+.3f}, {v['rare_diff'][2]:+.3f}], "
                         f"acc {v['acc_diff'][0]:+.3f} [{v['acc_diff'][1]:+.3f}, {v['acc_diff'][2]:+.3f}], "
                         f"rare fixed/broken {v['rare_fixed']}/{v['rare_broken']} (McNemar p={v['rare_mcnemar_p']:.2g})")
    (args.out / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
