"""C-17 (with A6): measured tau curves on BCTR for CTC (post-hoc G-LA on the dumps) and SMTR (step-wise decoding),
replacing the schematic curve of Fig. 1(c). Descriptive only: nothing here selects a configuration.

For every model, split and tau (no gate): pooled accuracy, rare-line accuracy, CER and two class-balanced line
metrics -- the macro average over Han characters of the per-character recall (Levenshtein alignment, as C-03 (a)),
over all characters present in the references and over those with >= 5 occurrences. Plateau: the tau range whose
pooled accuracy is within 0.05 points of the maximum. Official normalisation.

    python ly/scripts/c17_curves.py --ctc gtc=ly/exp/C-03_analysis/dumps off_T=ly/exp/C-04_external/dumps ... \
        --smtr gtc=ly/exp/C-09_smtr/preds,ly/exp/C-17_tau_curves/smtr_preds ... --out ly/exp/C-17_tau_curves
Protocol: ly/commit.md, entry "C-13 至 C-17 预注册".
"""
import argparse
import collections
import csv
import json
from pathlib import Path
import sys

import numpy as np
from rapidfuzz.distance import Levenshtein

sys.path.insert(0, str(Path(__file__).resolve().parent))
import a0_prior_correction as a0  # noqa: E402
import c4_analysis as c4  # noqa: E402

TAUS = (0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75, 1.0, 1.25, 1.5)


def smtr_name(tau):
    return "B0" if tau == 0 else f"t{tau}_g1.01"


def line_metrics(refs, hyps, rare):
    correct = np.array([h == r for h, r in zip(hyps, refs)])
    edits = sum(Levenshtein.distance(h, r) for h, r in zip(hyps, refs))
    tot, hit = collections.Counter(), collections.Counter()
    for ref, hyp in zip(refs, hyps):
        wrong = {s for tag, s, _ in Levenshtein.editops(ref, hyp) if tag in ("replace", "delete")}
        for i, ch in enumerate(ref):
            if a0.is_han(ch):
                tot[ch] += 1
                hit[ch] += i not in wrong
    rec = {c: hit[c] / tot[c] for c in tot}
    return {"acc": 100 * correct.mean(), "rare_acc": 100 * correct[rare].mean(),
            "cer": 100 * edits / sum(len(r) for r in refs),
            "char_macro_recall": 100 * float(np.mean(list(rec.values()))),
            "char_macro_recall_min5": 100 * float(np.mean([v for c, v in rec.items() if tot[c] >= 5]))}


def ctc_curve(tag, dumps, split):
    a0.DUMPS = dumps
    data = c4.load_split(tag, split)
    refs = [t for dom in a0.DOMAINS for t in data[dom]["targets"]]
    rare = np.concatenate([data[dom]["rare"] for dom in a0.DOMAINS])
    out = {}
    for tau in TAUS:
        cfg = None if tau == 0 else dict(method="G", tau=tau, k=0, gamma=1.01)
        hyps = [s for dom in a0.DOMAINS for s in a0.strings(data[dom], a0.decode(data[dom], cfg))]
        out[tau] = line_metrics(refs, hyps, rare)
    return out


def smtr_curve(tag, roots, split, norm):
    rows = {}
    for root in roots:
        for dom in a0.DOMAINS:
            path = root / f"{tag}_{split}_{dom}.jsonl"
            if not path.exists():
                continue
            for line in open(path, encoding="utf-8"):
                r = json.loads(line)
                rows.setdefault(r["sample_id"], {"target": r["target"], "preds": {}})["preds"].update(r["preds"])
    ids = sorted(rows)
    refs = [norm(rows[i]["target"]) for i in ids]
    rare = np.array([any(a0.is_han(c) and a0.COUNT_OF.get(c, 0) < a0.RARE for c in rows[i]["target"].replace(" ", "").lower())
                     for i in ids])
    out = {}
    for tau in TAUS:
        name = smtr_name(tau)
        if all(name in rows[i]["preds"] for i in ids):
            out[tau] = line_metrics(refs, [norm(rows[i]["preds"][name]) for i in ids], rare)
    return out


def plateau(curve):
    acc = {t: v["acc"] for t, v in curve.items()}
    best = max(acc.values())
    flat = [t for t, a in acc.items() if a >= best - 0.05]
    return {"tau_max_acc": max(acc, key=acc.get), "plateau": [min(flat), max(flat)],
            "tau_max_rare": max(curve, key=lambda t: curve[t]["rare_acc"]),
            "tau_max_char_macro_recall": max(curve, key=lambda t: curve[t]["char_macro_recall"])}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ctc", nargs="*", default=[], help="tag=dumps_dir")
    p.add_argument("--smtr", nargs="*", default=[], help="tag=preds_dir[,preds_dir...]")
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()
    norm = c4.official_normalizer()
    a0.normalize = norm
    curves, summary = [], {}
    jobs = [("CTC", *x.split("=", 1)) for x in args.ctc] + [("SMTR", *x.split("=", 1)) for x in args.smtr]
    for decoder, tag, where in jobs:
        for split in ("val", "test"):
            cur = (ctc_curve(tag, Path(where), split) if decoder == "CTC"
                   else smtr_curve(tag, [Path(w) for w in where.split(",")], split, norm))
            if not cur:
                continue
            summary[f"{decoder}:{tag}:{split}"] = plateau(cur)
            for tau, v in cur.items():
                curves.append({"decoder": decoder, "model": tag, "split": split, "tau": tau, **v})
            print(decoder, tag, split, json.dumps(summary[f"{decoder}:{tag}:{split}"]), flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "curves.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(curves[0]))
        w.writeheader()
        w.writerows(curves)
    (args.out / "c17_results.json").write_text(json.dumps({"taus": TAUS, "summary": summary}, indent=1))
    lines = ["# C-17 / A6 tau curves (official normalisation; descriptive)", "",
             "| decoder:model:split | tau max acc | plateau (acc within 0.05) | tau max rare | tau max char macro recall |",
             "|---|---:|---|---:|---:|"]
    for k, v in summary.items():
        lines.append(f"| {k} | {v['tau_max_acc']} | {v['plateau']} | {v['tau_max_rare']} | {v['tau_max_char_macro_recall']} |")
    (args.out / "report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
