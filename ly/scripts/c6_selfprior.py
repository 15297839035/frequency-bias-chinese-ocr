"""C-06 (exploratory): G-LA with a self-estimated prior, for models whose training statistics are unknown.

pi_hat = add-one character counts of the model's own greedy transcripts on the (unlabelled) validation set.
Same grid, selection rule and single test read-out as c4_analysis.py; compared with the BCTR training-frequency
prior. Reuses a0_prior_correction / c4_analysis unchanged (only a0.LOG_PRIOR is swapped).

    python ly/scripts/c6_selfprior.py --out ly/exp/C-06_selfprior
"""
import argparse
import collections
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import a0_prior_correction as a0  # noqa: E402
import c4_analysis as c4  # noqa: E402

EXP = a0.REPO / "ly/exp"
MODELS = [("gtc", EXP / "C-03_analysis/dumps", "strict"), ("b", EXP / "C-03_analysis/dumps", "strict"),
          ("off_T", EXP / "C-04_external/dumps", "official"), ("off_S", EXP / "C-04_external/dumps", "official"),
          ("off_B", EXP / "C-04_external/dumps", "official"),
          ("svtrv2_ch", EXP / "C-04b_industrial/dumps", "official"), ("repsvtr_ch", EXP / "C-04b_industrial/dumps", "official"),
          ("crnn_s1", EXP / "C-05_crnn/dumps", "strict"), ("crnn_s2", EXP / "C-05_crnn/dumps", "strict")]
BCTR_LOG_PRIOR = a0.LOG_PRIOR.copy()
CHAR_ID = {c: i for i, c in enumerate(a0.CHARS)}


def self_prior(val):
    counts = np.zeros(len(a0.CHARS))
    for dom in a0.DOMAINS:
        for r in val[dom]["rows"]:
            for ch in r["prediction"]:
                if ch in CHAR_ID:
                    counts[CHAR_ID[ch]] += 1
    lp = np.log((counts + 1.0) / (counts + 1.0)[1:].sum())
    lp[0] = 0.0
    return lp, counts


def run(tag, norm):
    a0.normalize = c4.official_normalizer() if norm == "official" else (lambda t: t.replace(" ", "").lower())
    val, test = c4.load_split(tag, "val"), c4.load_split(tag, "test")
    out = {"norm": norm}
    lp_hat, counts = self_prior(val)
    nz = (counts[1:] > 0)
    out["prior_corr_with_bctr"] = float(np.corrcoef(lp_hat[1:][nz], BCTR_LOG_PRIOR[1:][nz])[0, 1])
    for name, lp in (("bctr_prior", BCTR_LOG_PRIOR), ("self_prior", lp_hat)):
        a0.LOG_PRIOR = lp
        _, _, chosen = c4.select(val)
        t = c4.test_eval(test, {"B0": None, "G_selected": chosen, "G*": c4.GSTAR})
        out[name] = {"selected": chosen,
                     **{k: {kk: t[k][kk] for kk in ("acc", "rare_acc", "rare_gain", "acc_change", "rare_mcnemar_p")
                            if kk in t[k]} for k in t}}
    a0.LOG_PRIOR = BCTR_LOG_PRIOR
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=EXP / "C-06_selfprior")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    res, lines = {}, ["| model | prior | selected (tau, gamma) | test overall change | test rare-line change | G* rare change |",
                      "|---|---|---|---|---|---|"]
    for tag, dumps, norm in MODELS:
        if not (dumps / f"{tag}_val_scene.npz").exists():
            continue
        a0.DUMPS = dumps
        res[tag] = r = run(tag, norm)
        for name in ("bctr_prior", "self_prior"):
            s, g = r[name].get("G_selected"), r[name]["G*"]
            sel = r[name]["selected"]
            lines.append(f"| {tag} | {name} | {None if sel is None else (sel['tau'], sel['gamma'])} | "
                         f"{s['acc_change'] if s else float('nan'):+.3f} | {s['rare_gain'] if s else float('nan'):+.3f} | "
                         f"{g['rare_gain']:+.3f} |")
        print("\n".join(lines[-2:]), f"(corr of log priors {r['prior_corr_with_bctr']:.3f})", flush=True)
    (args.out / "c6_results.json").write_text(json.dumps(res, indent=1, default=float))
    (args.out / "report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
