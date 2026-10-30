"""C-04: external validation on the official BCTR-trained SVTRv2 T/S/B models (protocol: docs/experiments.md, "C-04 预注册"
and its amendment). Reuses core/prior_correction.py (loading, G-LA, metrics), analysis/c03_confusion_analysis.py (over-production) and
analysis/c03_confusion_asymmetry.py (exposure-controlled confusion direction) unchanged; only the text normalisation is switched.

    python src/analysis/c04_external_analysis.py --dumps results/C-04_external/dumps --out results/C-04_external --norm official
    python src/analysis/c04_external_analysis.py ... --norm strict      # secondary protocol (A0's normalisation)
"""
import argparse
import json
from pathlib import Path
import sys
import unicodedata

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import prior_correction as a0  # noqa: E402
from analysis import c03_confusion_analysis as c3  # noqa: E402
from analysis import c03_confusion_asymmetry as c3b  # noqa: E402

TAGS = ("off_T", "off_S", "off_B")
GSTAR = dict(method="G", tau=0.2, k=0, gamma=0.9)


def official_normalizer():
    from opencc import OpenCC
    t2s = OpenCC("t2s").convert
    return lambda text: t2s(unicodedata.normalize("NFKC", text)).lower().replace(" ", "")


def rare_from_raw(d):
    """Rare-line flags from the raw reference (training count < 100), independent of the normalisation."""
    return np.array([any(a0.is_han(ch) and a0.COUNT_OF.get(ch, 0) < a0.RARE for ch in r["target"].replace(" ", "").lower())
                     for r in d["rows"]])


def load_split(tag, split):
    data = {dom: a0.load(tag, split, dom) for dom in a0.DOMAINS}
    for dom, d in data.items():
        d["rare"] = rare_from_raw(d)
        a0.check_baseline(a0.strings(d, a0.decode(d, None)), d["baseline"], (tag, split, dom))
    return data


def select(val):
    _, base = a0.evaluate(val, None)
    grid = []
    for cfg in a0.GRID_G:
        _, pool = a0.evaluate(val, cfg)
        grid.append({**cfg, **{k: v for k, v in pool.items()}})
    ok = [g for g in grid if g["acc"] >= base["acc"] - 0.05]
    best = sorted(ok, key=lambda g: (-g["rare_correct"], g["tau"], g["gamma"]))[0] if ok else None
    return base, grid, None if best is None else {k: best[k] for k in ("method", "tau", "k", "gamma")}


def test_eval(test, configs):
    out, vecs = {}, {}
    rare = np.concatenate([test[dom]["rare"] for dom in a0.DOMAINS])
    for name, cfg in configs.items():
        if cfg is None and name != "B0":
            continue
        per, pool = a0.evaluate(test, cfg)
        vecs[name] = np.concatenate([per[dom]["correct_vector"] for dom in a0.DOMAINS])
        out[name] = {"cfg": cfg, "acc": pool["acc"], "rare_acc": pool["rare_acc"], "nonrare_acc": pool["nonrare_acc"],
                     "cer": pool["cer"], "domains": {dom: 100 * per[dom]["correct"] / per[dom]["n"] for dom in a0.DOMAINS}}
    for name in out:
        if name == "B0":
            continue
        fixed = int((vecs[name] & ~vecs["B0"] & rare).sum())
        broken = int((~vecs[name] & vecs["B0"] & rare).sum())
        out[name].update(rare_gain=out[name]["rare_acc"] - out["B0"]["rare_acc"], acc_change=out[name]["acc"] - out["B0"]["acc"],
                         rare_fixed=fixed, rare_broken=broken, rare_mcnemar_p=a0.exact_mcnemar(fixed, broken))
    return out


def mechanism(data):
    refs = [t for dom in a0.DOMAINS for t in data[dom]["targets"]]
    res = {}
    for name, cfg in (("B0", None), ("G*", GSTAR)):
        hyps = [s for dom in a0.DOMAINS for s in a0.strings(data[dom], a0.decode(data[dom], cfg))]
        rows = c3b.pairs_from(refs, hyps)
        res[name] = {"asymmetry": {grp: c3b.analyse([r for r in rows if sel(r)])
                                   for grp, sel in (("all", lambda r: True), ("lookalike", lambda r: r["look"]))},
                     "overproduction": c3.overproduction(refs, hyps)}
    return res


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dumps", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--norm", choices=("official", "strict"), required=True)
    p.add_argument("--tags", nargs="+", default=list(TAGS))
    args = p.parse_args()
    a0.DUMPS = args.dumps
    if args.norm == "official":
        a0.normalize = official_normalizer()
    report, lines = {"norm": args.norm, "models": {}}, []
    for tag in args.tags:
        val, test = load_split(tag, "val"), load_split(tag, "test")
        base_val, grid, chosen = select(val)
        t = test_eval(test, {"B0": None, "G_selected": chosen, "G*": GSTAR})
        mech = {"val": mechanism(val), "test": mechanism(test)}
        sel = t.get("G_selected")
        crit_i = bool(sel and sel["rare_gain"] >= 0.5 and sel["rare_mcnemar_p"] < 0.05 and sel["acc_change"] >= -0.1)
        crit_ii = all(mech[s]["B0"]["asymmetry"]["all"]["toward_more_frequent"]["ratio"] > 1.5
                      and mech[s]["B0"]["asymmetry"]["all"]["toward_more_frequent"]["z"] > 3
                      and mech[s]["B0"]["asymmetry"]["all"]["beta"]["ci95"][0] > 0 for s in ("val", "test"))
        crit_iii = all(mech[s]["B0"]["overproduction"]["slope_ci95"][0] > 0 for s in ("val", "test"))
        report["models"][tag] = {"val_B0": {k: base_val[k] for k in ("acc", "rare_acc")}, "selected": chosen,
                                 "val_grid": [{k: g[k] for k in ("tau", "gamma", "acc", "rare_acc")} for g in grid],
                                 "test": t, "mechanism": mech,
                                 "criteria": {"i_gla": crit_i, "ii_direction": crit_ii, "iii_overproduction": crit_iii}}
        b0, s = t["B0"], t.get("G_selected")
        lines.append(f"## {tag} ({args.norm} normalisation)")
        lines.append(f"- val B0 acc {base_val['acc']:.3f} rare {base_val['rare_acc']:.3f}; selected {chosen}")
        lines.append(f"- test B0: acc {b0['acc']:.3f} rare {b0['rare_acc']:.3f} | domains "
                     + ", ".join(f"{k} {v:.2f}" for k, v in b0["domains"].items()))
        for name in ("G_selected", "G*"):
            if name in t:
                r = t[name]
                lines.append(f"- test {name}: acc {r['acc']:.3f} ({r['acc_change']:+.3f}) rare {r['rare_acc']:.3f} "
                             f"({r['rare_gain']:+.3f}), fixed/broken {r['rare_fixed']}/{r['rare_broken']}, p={r['rare_mcnemar_p']:.2g}")
        for s in ("val", "test"):
            a = mech[s]["B0"]["asymmetry"]["all"]
            if "beta" in a:
                tm = a["toward_more_frequent"]
                lines.append(f"- {s} direction: pairs {tm['pairs']}, obs {tm['pct_observed']:.1f}% vs sym {tm['pct_expected']:.1f}%, "
                             f"T/E {tm['ratio']:.2f}, z {tm['z']:.1f}, beta {a['beta']['estimate']:.3f} "
                             f"[{a['beta']['ci95'][0]:.3f}, {a['beta']['ci95'][1]:.3f}]")
            o = mech[s]["B0"]["overproduction"]
            og = mech[s]["G*"]["overproduction"]
            lines.append(f"- {s} over-production slope B0 {o['slope']:.4f} [{o['slope_ci95'][0]:.4f}, {o['slope_ci95'][1]:.4f}], "
                         f"G* {og['slope']:.4f}")
        lines.append(f"- criteria: (i) {crit_i}  (ii) {crit_ii}  (iii) {crit_iii}")
        lines.append("")
        print("\n".join(lines[-12:]), flush=True)
    report["passed_all"] = all(all(m["criteria"].values()) for m in report["models"].values())
    lines.append(f"**External validation passed (all models, all criteria): {report['passed_all']}**")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"c4_results_{args.norm}.json").write_text(json.dumps(report, indent=1, default=float, ensure_ascii=False))
    (args.out / f"report_{args.norm}.md").write_text("\n".join(lines) + "\n")
    print(lines[-1])


if __name__ == "__main__":
    main()
