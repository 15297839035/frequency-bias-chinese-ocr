"""C-16 (b, c): CTC models on handwritten text lines that none of them was trained on.

Data: BCTR's extra handwriting collection ("hwdb_ic13"): CASIA HWDB2.x lines as val (handwriting_hwdb_val) and the
ICDAR 2013 text-line competition set as test (handwriting_ic13_test), stored as domain "hwdb"
(<root>/hwdb/hwdb_val, hwdb_test) and dumped with dump_external_models.py / dump_ctc_posteriors.py (lines the CTC label encoder
rejects, e.g. > 25 characters, are excluded as for BCTR). Official normalisation; rare line = a reference character
seen < 100 times in BCTR training (as everywhere). Per model:
  (ii)  direction: T/E > 1.5, z > 3 and beta CI excluding 0, on val and test (C-04 criterion);
  label-free correction on test: G* (tau 0.2, gamma 0.9) with the BCTR training prior, and with the prior estimated
        from the model's own outputs on the unlabelled hwdb val lines (C-06 method, now on a new domain);
  secondary: the A0 selection rule on (labelled) hwdb val, for each prior; every frozen config read on test once.

    python src/inference/run_hwdb_baselines.py --dumps results/C-16_external/hwdb_dumps --out results/C-16_external/hwdb \
        --tags off_T off_S off_B gtc b crnn_s1 crnn_s2
Protocol: docs/experiments.md, entry "C-13 至 C-17 预注册".
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import prior_correction as a0  # noqa: E402
from analysis import c04_external_analysis as c4  # noqa: E402
from analysis import c06_self_prior as c6  # noqa: E402


def keep(t):
    return {k: v for k, v in t.items() if k in ("acc", "rare_acc", "nonrare_acc", "cer", "rare_gain", "acc_change",
                                                "rare_fixed", "rare_broken", "rare_mcnemar_p", "cfg")}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dumps", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--tags", nargs="+", required=True)
    args = p.parse_args()
    a0.DUMPS = args.dumps
    a0.DOMAINS = ("hwdb",)  # c4.load_split / c6.self_prior iterate a0.DOMAINS
    a0.normalize = c4.official_normalizer()
    bctr = c6.BCTR_LOG_PRIOR.copy()
    report, lines = {}, ["# C-16 (b, c): CTC models on handwritten lines (hwdb val / ICDAR 2013 test), official norm.", ""]
    for tag in args.tags:
        a0.LOG_PRIOR = bctr
        val, test = c4.load_split(tag, "val"), c4.load_split(tag, "test")
        meta = {s: json.loads((args.dumps / f"{tag}_{s}_hwdb.meta.json").read_text()) for s in ("val", "test")}
        mech = {"val": c4.mechanism(val), "test": c4.mechanism(test)}
        crit_ii = all(mech[s]["B0"]["asymmetry"]["all"].get("toward_more_frequent", {}).get("ratio", 0) > 1.5
                      and mech[s]["B0"]["asymmetry"]["all"]["toward_more_frequent"]["z"] > 3
                      and mech[s]["B0"]["asymmetry"]["all"]["beta"]["ci95"][0] > 0 for s in ("val", "test"))
        lp_hat, _ = c6.self_prior(val)
        r = {"samples": {s: meta[s]["samples"] for s in meta}, "excluded": {s: meta[s]["excluded"] for s in meta},
             "mechanism": mech, "ii_direction": crit_ii}
        for name, lp in (("bctr_prior", bctr), ("self_prior", lp_hat)):
            a0.LOG_PRIOR = lp
            _, _, chosen = c4.select(val)
            t = c4.test_eval(test, {"B0": None, "G*": c4.GSTAR, "G_selected": chosen})
            r[name] = {"selected": chosen, **{k: keep(v) for k, v in t.items()}}
        a0.LOG_PRIOR = bctr
        report[tag] = r
        b0 = r["bctr_prior"]["B0"]
        lines.append(f"## {tag}: test B0 acc {b0['acc']:.2f}, rare {b0['rare_acc']:.2f}; samples {r['samples']} "
                     f"(excluded {r['excluded']})")
        for s in ("val", "test"):
            a = mech[s]["B0"]["asymmetry"]["all"]
            if "beta" in a:
                tm = a["toward_more_frequent"]
                lines.append(f"- {s} direction: pairs {tm['pairs']}, T/E {tm['ratio']:.2f}, z {tm['z']:.1f}, beta "
                             f"{a['beta']['estimate']:.3f} [{a['beta']['ci95'][0]:.3f}, {a['beta']['ci95'][1]:.3f}]")
            else:
                lines.append(f"- {s} direction: only {a['pairs']} pairs")
        for name in ("bctr_prior", "self_prior"):
            for cfg in ("G*", "G_selected"):
                if cfg in r[name]:
                    v = r[name][cfg]
                    lines.append(f"- {name} {cfg}: rare {v['rare_gain']:+.2f} (fixed/broken {v['rare_fixed']}/"
                                 f"{v['rare_broken']}, p={v['rare_mcnemar_p']:.2g}), overall {v['acc_change']:+.2f}"
                                 + (f", selected {r[name]['selected']}" if cfg == "G_selected" else ""))
        lines += [f"- criterion (ii) direction: {crit_ii}", ""]
        print("\n".join(lines[-9:]), flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "c16_hwdb_results.json").write_text(json.dumps(report, indent=1, default=float, ensure_ascii=False))
    (args.out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
