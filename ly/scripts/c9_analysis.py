"""C-09 analysis: SMTR step-wise correction -- val selection, one test read-out, mechanism (protocol: ly/commit.md,
"C-07 至 C-11 预注册", C-09). Same rules as c4_analysis.py: select on pooled val the config with the most correct
rare lines among those with pooled accuracy >= B0 - 0.05pp (ties: smaller tau, then gamma); report B0, the selected
config and G* (tau 0.2, gamma 0.9) on test; T/E, beta and over-production of B0 / G* on val and test.

    python ly/scripts/c9_analysis.py --preds ly/exp/C-09_smtr/preds --out ly/exp/C-09_smtr --norm official \
        --tags off_T off_S off_B
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from rapidfuzz.distance import Levenshtein

sys.path.insert(0, str(Path(__file__).resolve().parent))
import a0_prior_correction as a0  # noqa: E402
import c3_analysis as c3  # noqa: E402
import c3b_asymmetry as c3b  # noqa: E402
from c4_analysis import official_normalizer  # noqa: E402

GSTAR = "t0.2_g0.9"


def load(preds, tag, split, norm):
    rows = []
    for dom in a0.DOMAINS:
        for line in open(preds / f"{tag}_{split}_{dom}.jsonl", encoding="utf-8"):
            r = json.loads(line)
            r["domain"] = dom
            rows.append(r)
    targets = [norm(r["target"]) for r in rows]
    rare = np.array([any(a0.is_han(ch) and a0.COUNT_OF.get(ch, 0) < a0.RARE for ch in r["target"].replace(" ", "").lower())
                     for r in rows])
    hyps = {k: [norm(r["preds"][k]) for r in rows] for k in rows[0]["preds"]}
    dom = np.array([r["domain"] for r in rows])
    return {"ids": [r["sample_id"] for r in rows], "targets": targets, "rare": rare, "hyps": hyps, "domain": dom}


def score(d, name):
    hyp = d["hyps"][name]
    correct = np.array([h == t for h, t in zip(hyp, d["targets"])])
    edits = sum(Levenshtein.distance(h, t) for h, t in zip(hyp, d["targets"]))
    return {"acc": 100 * correct.mean(), "rare_acc": 100 * correct[d["rare"]].mean(),
            "nonrare_acc": 100 * correct[~d["rare"]].mean(), "rare_correct": int(correct[d["rare"]].sum()),
            "cer": 100 * edits / sum(len(t) for t in d["targets"]),
            "domains": {dom: 100 * correct[d["domain"] == dom].mean() for dom in a0.DOMAINS}, "vec": correct}


def parse(name):
    if name == "B0":
        return 0.0, 1.01
    t, g = name[1:].split("_g")
    return float(t), float(g)


def mechanism(d, names):
    out = {}
    for name in names:
        rows = c3b.pairs_from(d["targets"], d["hyps"][name])
        out[name] = {"asymmetry": {grp: c3b.analyse([r for r in rows if sel(r)])
                                   for grp, sel in (("all", lambda r: True), ("lookalike", lambda r: r["look"]))},
                     "overproduction": c3.overproduction(d["targets"], d["hyps"][name])}
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--preds", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--norm", choices=("official", "strict"), required=True)
    p.add_argument("--tags", nargs="+", required=True)
    args = p.parse_args()
    norm = official_normalizer() if args.norm == "official" else a0.normalize
    report, lines = {"norm": args.norm, "models": {}}, []
    for tag in args.tags:
        val, test = load(args.preds, tag, "val", norm), load(args.preds, tag, "test", norm)
        grid = {name: score(val, name) for name in val["hyps"]}
        base = grid["B0"]
        ok = [n for n in grid if n != "B0" and grid[n]["acc"] >= base["acc"] - 0.05]
        chosen = sorted(ok, key=lambda n: (-grid[n]["rare_correct"], *parse(n)))[0] if ok else None
        t = {}
        for name in ["B0"] + ([chosen] if chosen else []) + [GSTAR]:
            s = score(test, name)
            if name != "B0":
                fixed = int((s["vec"] & ~t["B0"]["vec"] & test["rare"]).sum())
                broken = int((~s["vec"] & t["B0"]["vec"] & test["rare"]).sum())
                s.update(rare_gain=s["rare_acc"] - t["B0"]["rare_acc"], acc_change=s["acc"] - t["B0"]["acc"],
                         rare_fixed=fixed, rare_broken=broken, rare_mcnemar_p=a0.exact_mcnemar(fixed, broken))
            t["selected" if name == chosen else ("G*" if name == GSTAR else "B0")] = s
            if name == chosen == GSTAR:
                t["G*"] = s
        mech = {"val": mechanism(val, ["B0", GSTAR]), "test": mechanism(test, ["B0", GSTAR])}
        sel = t.get("selected")
        crit_i = bool(sel and sel["rare_gain"] >= 0.5 and sel["rare_mcnemar_p"] < 0.05 and sel["acc_change"] >= -0.1)
        crit_ii = all(mech[s]["B0"]["asymmetry"]["all"]["toward_more_frequent"]["ratio"] > 1.5
                      and mech[s]["B0"]["asymmetry"]["all"]["toward_more_frequent"]["z"] > 3
                      and mech[s]["B0"]["asymmetry"]["all"]["beta"]["ci95"][0] > 0 for s in ("val", "test"))
        crit_iii = all(mech[s]["B0"]["overproduction"]["slope_ci95"][0] > 0 for s in ("val", "test"))
        strip = lambda s: {k: v for k, v in s.items() if k != "vec"}  # noqa: E731
        report["models"][tag] = {"selected": chosen,
                                 "val_grid": {n: {k: g[k] for k in ("acc", "rare_acc", "cer")} for n, g in grid.items()},
                                 "test": {k: strip(v) for k, v in t.items()}, "mechanism": mech,
                                 "criteria": {"i_gla": crit_i, "ii_direction": crit_ii, "iii_overproduction": crit_iii}}
        b0 = t["B0"]
        lines.append(f"## {tag} SMTR ({args.norm} normalisation)")
        lines.append(f"- val B0 acc {base['acc']:.3f} rare {base['rare_acc']:.3f}; selected {chosen}")
        lines.append(f"- test B0: acc {b0['acc']:.3f} rare {b0['rare_acc']:.3f} cer {b0['cer']:.3f} | "
                     + ", ".join(f"{k} {v:.2f}" for k, v in b0["domains"].items()))
        for name in ("selected", "G*"):
            if name in t:
                r = t[name]
                lines.append(f"- test {name}: acc {r['acc']:.3f} ({r['acc_change']:+.3f}) rare {r['rare_acc']:.3f} "
                             f"({r['rare_gain']:+.3f}), fixed/broken {r['rare_fixed']}/{r['rare_broken']}, "
                             f"p={r['rare_mcnemar_p']:.2g}")
        for s in ("val", "test"):
            a = mech[s]["B0"]["asymmetry"]["all"]
            tm = a["toward_more_frequent"]
            o, og = mech[s]["B0"]["overproduction"], mech[s][GSTAR]["overproduction"]
            lines.append(f"- {s} direction: pairs {tm['pairs']}, T/E {tm['ratio']:.2f}, z {tm['z']:.1f}, beta "
                         f"{a['beta']['estimate']:.3f} [{a['beta']['ci95'][0]:.3f}, {a['beta']['ci95'][1]:.3f}]; "
                         f"over-production B0 {o['slope']:.4f} [{o['slope_ci95'][0]:.4f}, {o['slope_ci95'][1]:.4f}] "
                         f"-> G* {og['slope']:.4f}")
        lines.append(f"- criteria: (i) {crit_i}  (ii) {crit_ii}  (iii) {crit_iii}")
        lines.append("")
        print("\n".join(lines[-9:]), flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"c9_results_{args.norm}.json").write_text(json.dumps(report, indent=1, default=float, ensure_ascii=False))
    (args.out / f"report_{args.norm}.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
