"""C-02 evaluation: post-hoc G-LA on top of each continued run; val selection, one test read-out, judgement.

Reuses src/core/prior_correction.py (dump loading, G-LA decoding, metrics) read-only.
Dumps: <dumps>/<run>_{val,test}_{scene,web,document}.npz from dump_ctc_posteriors.py --checkpoint --tag <run>.

    python src/evaluation/evaluate_bctr.py --dumps results/C-02_bctr_la/dumps --ref-dumps results/C-03_analysis/dumps \
        --out results/C-02_bctr_la
Protocol: docs/experiments.md, entries "C-02 预注册" and "C-02 预注册补充".
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import prior_correction as a0  # noqa: E402

RUNS = {"c0_s1": (0.0, 1), "c0_s2": (0.0, 2), "la025_s1": (0.25, 1), "la025_s2": (0.25, 2),
        "la1_s1": (1.0, 1), "la1_s2": (1.0, 2)}
TAU_EFF = (-0.25, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 1.25)
GAMMAS = (0.9, 0.99, 1.01)


def configs(tau_train):
    out = [None]  # default: no post-hoc change
    for t in TAU_EFF:
        post = round(t - tau_train, 4)
        if post == 0:
            continue  # identical to the default
        out += [dict(method="G", tau=post, k=0, gamma=g, tau_eff=t) for g in GAMMAS]
    return out


def evaluate_split(dumps, tag, split, cfgs):
    a0.DUMPS = Path(dumps)
    data = {dom: a0.load(tag, split, dom) for dom in a0.DOMAINS}
    for dom, d in data.items():
        a0.check_baseline(a0.strings(d, a0.decode(d, None)), d["baseline"], (tag, split, dom))
    ids = [r["sample_id"] for dom in a0.DOMAINS for r in data[dom]["rows"]]
    rare = np.concatenate([data[dom]["rare"] for dom in a0.DOMAINS])
    rows = []
    for cfg in cfgs:
        per, pool = a0.evaluate(data, cfg)
        vec = np.concatenate([per[dom]["correct_vector"] for dom in a0.DOMAINS])
        rows.append({"cfg": cfg, "acc": pool["acc"], "rare_acc": pool["rare_acc"], "nonrare_acc": pool["nonrare_acc"],
                     "cer": pool["cer"], "rare_correct": pool["rare_correct"],
                     "domains": {dom: 100 * per[dom]["correct"] / per[dom]["n"] for dom in a0.DOMAINS},
                     "vec": vec})
    return {"ids": ids, "rare": rare, "rows": rows}


def job(args):
    dumps, tag, tau_train = args
    cfgs = configs(tau_train)
    return tag, {split: evaluate_split(dumps, tag, split, cfgs) for split in ("val", "test")}


def bootstrap(va, vb, mask, reps=1000, seed=0):
    """Paired line bootstrap of acc(A) - acc(B) on the lines selected by mask (pp)."""
    a, b = va[mask].astype(float), vb[mask].astype(float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), (reps, len(a)))
    diffs = 100 * (a[idx].mean(1) - b[idx].mean(1))
    return [float(100 * (a.mean() - b.mean())), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]


def strip(row):
    return {k: v for k, v in row.items() if k != "vec"}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dumps", required=True)
    p.add_argument("--ref-dumps", dest="ref_dumps", required=True, help="dir with gtc_{val,test}_*.npz (epoch 100)")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--workers", type=int, default=7)
    args = p.parse_args()
    jobs = [(args.dumps, tag, tau) for tag, (tau, _) in RUNS.items()] + [(args.ref_dumps, "gtc", 0.0)]
    with ProcessPoolExecutor(args.workers) as pool:
        res = dict(pool.map(job, jobs))
    ref_ids = res["c0_s1"]["test"]["ids"]
    for tag in res:
        assert res[tag]["test"]["ids"] == ref_ids, f"test order differs: {tag}"
        assert res[tag]["val"]["ids"] == res["c0_s1"]["val"]["ids"], f"val order differs: {tag}"
    rare_test = res["c0_s1"]["test"]["rare"]
    report = {"runs": {}, "judgement": {}}
    selected = {}
    for tag, (tau_train, seed) in RUNS.items():
        val, test = res[tag]["val"]["rows"], res[tag]["test"]["rows"]
        c0_val_acc = res[f"c0_s{seed}"]["val"]["rows"][0]["acc"]
        ok = [i for i, r in enumerate(val) if r["acc"] >= c0_val_acc - 0.05]
        if ok:
            key = lambda i: (-val[i]["rare_correct"], abs(val[i]["cfg"]["tau"]) if val[i]["cfg"] else 0.0,
                             val[i]["cfg"]["gamma"] if val[i]["cfg"] else 0.0)
            k = sorted(ok, key=key)[0]
            feasible = True
        else:
            k, feasible = int(np.argmax([r["acc"] for r in val])), False
        selected[tag] = k
        report["runs"][tag] = {"tau_train": tau_train, "seed": seed, "feasible": feasible,
                               "constraint_val_acc": c0_val_acc - 0.05,
                               "selected_cfg": val[k]["cfg"], "val_selected": strip(val[k]),
                               "test_default": strip(test[0]), "test_selected": strip(test[k]),
                               "val_frontier": [strip(r) for r in val], "test_frontier_descriptive": [strip(r) for r in test]}
    ref = res["gtc"]
    report["epoch100_reference"] = {"val_default": strip(ref["val"]["rows"][0]), "test_default": strip(ref["test"]["rows"][0])}
    for arm in ("la025", "la1"):
        per_seed = {}
        for seed in (1, 2):
            t, c = res[f"{arm}_s{seed}"]["test"]["rows"], res[f"c0_s{seed}"]["test"]["rows"]
            vt, vc = t[selected[f"{arm}_s{seed}"]]["vec"], c[selected[f"c0_s{seed}"]]["vec"]
            fixed, broken = int((vt & ~vc & rare_test).sum()), int((~vt & vc & rare_test).sum())
            per_seed[seed] = {"rare_diff": bootstrap(vt, vc, rare_test),
                              "acc_diff": bootstrap(vt, vc, np.ones_like(rare_test)),
                              "rare_fixed": fixed, "rare_broken": broken,
                              "rare_mcnemar_p": a0.exact_mcnemar(fixed, broken)}
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
    (args.out / "c2_results.json").write_text(json.dumps(report, indent=1, default=float))
    lines = ["| run | tau_train | selected (tau_eff, gamma) | val acc / rare | test default acc / rare | test selected acc / rare |",
             "|---|---:|---|---|---|---|"]
    for tag, r in report["runs"].items():
        cfg = r["selected_cfg"]
        sel = "default" if cfg is None else f"{cfg['tau_eff']}, {cfg['gamma']}"
        lines.append(f"| {tag} | {r['tau_train']} | {sel}{'' if r['feasible'] else ' (infeasible)'} | "
                     f"{r['val_selected']['acc']:.3f} / {r['val_selected']['rare_acc']:.3f} | "
                     f"{r['test_default']['acc']:.3f} / {r['test_default']['rare_acc']:.3f} | "
                     f"{r['test_selected']['acc']:.3f} / {r['test_selected']['rare_acc']:.3f} |")
    e = report["epoch100_reference"]
    lines.append(f"| epoch100 (reference) | 0 | default | {e['val_default']['acc']:.3f} / {e['val_default']['rare_acc']:.3f} | "
                 f"{e['test_default']['acc']:.3f} / {e['test_default']['rare_acc']:.3f} | — |")
    lines.append("")
    for arm, j in report["judgement"].items():
        lines.append(f"- {arm} vs c0 (selected vs selected): verdict **{j['verdict']}**")
        for s, v in j["per_seed"].items():
            lines.append(f"  - seed {s}: rare {v['rare_diff'][0]:+.3f} [{v['rare_diff'][1]:+.3f}, {v['rare_diff'][2]:+.3f}], "
                         f"acc {v['acc_diff'][0]:+.3f} [{v['acc_diff'][1]:+.3f}, {v['acc_diff'][2]:+.3f}], "
                         f"rare fixed/broken {v['rare_fixed']}/{v['rare_broken']} (McNemar p={v['rare_mcnemar_p']:.2g})")
    (args.out / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
