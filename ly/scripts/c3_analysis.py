"""C-03: mechanism analyses of rare-character errors on BCTR (CTC posteriors of GTC / B seed48).

Reuses ly/scripts/a0_prior_correction.py (dump loading, G-LA decoding, metrics) read-only. Configs:
B0 = greedy decode; G* = G-LA tau=0.2, gamma=0.9 (frozen in A0-03 on GTC val).

    python ly/scripts/c3_analysis.py --dumps ly/exp/C-03_analysis/dumps --out ly/exp/C-03_analysis
Analyses (protocol: ly/commit.md, entry "C-03 预注册"):
  a  character recall by training-frequency bucket, B0 vs G*
  b  Han substitution errors by look-alike (IDS component or glyph top-50) x frequency direction
  c  confusion asymmetry: does the error direction of a confusable pair follow the frequency ratio?
  d  what G* changes: substitutions it removes / introduces, by look-alike and direction
  e  over-production of frequent characters: slope of log(pred/ref count) on log(train count)
  f  per-domain G-LA selection on val vs the pooled selection (test evaluated once per frozen config)
"""
import argparse
import collections
import json
import math
from pathlib import Path
import sys

import numpy as np
from rapidfuzz.distance import Levenshtein

sys.path.insert(0, str(Path(__file__).resolve().parent))
import a0_prior_correction as a0  # noqa: E402

REPO = a0.REPO
GLYPH = REPO / "ly/exp/A0-05_glyph_neighbors/neighbors_k50.npy"
IDS = REPO / "OpenOCR-main/third_party/ucr_author_320ea5e8f141/Pandarenlql-UCR-320ea5e/files/IDS_V7.txt"
IDC = set("⿰⿱⿲⿳⿴⿵⿶⿷⿸⿹⿺⿻")
GSTAR = dict(method="G", tau=0.2, k=0, gamma=0.9)
BUCKETS = ((0, 1, "0"), (1, 10, "1-9"), (10, 100, "10-99"), (100, 1000, "100-999"), (1000, 10 ** 12, ">=1000"))
GRID_DOMAIN = [dict(method="G", tau=t, k=0, gamma=g) for t in (0.1, 0.2, 0.3, 0.5, 0.75, 1.0)
               for g in (0.9, 0.99, 1.01)]
CHAR_ID = {c: i for i, c in enumerate(a0.CHARS)}


def load_ids():
    ids = {}
    for line in open(IDS, encoding="utf-8-sig"):
        ch, sep, seq = line.rstrip("\n").partition(":")
        if sep:
            ids[ch] = set(seq.split()) - IDC
    return ids


IDS_COMP = load_ids()
GLYPH_NB = np.load(GLYPH)


def lookalike(a, b):
    """Shares an IDS component, or either is in the other's rendered-glyph top-50."""
    if a in IDS_COMP and b in IDS_COMP and IDS_COMP[a] & IDS_COMP[b]:
        return True
    ia, ib = CHAR_ID.get(a), CHAR_ID.get(b)
    if ia is None or ib is None:
        return False
    return bool((GLYPH_NB[ia] == ib).any() or (GLYPH_NB[ib] == ia).any())


def count(c):
    return a0.COUNT_OF.get(c, 0)


def bucket(n):
    return next(name for lo, hi, name in BUCKETS if lo <= n < hi)


def direction(a, b):
    ca, cb = count(a), count(b)
    return "to_more_frequent" if cb > ca else ("to_less_frequent" if cb < ca else "equal")


def substitutions(ref, hyp):
    """1:1 substitutions (ref char, hyp char) of a minimal edit script."""
    return [(ref[s], hyp[d]) for tag, s, d in Levenshtein.editops(ref, hyp) if tag == "replace"]


def aligned_recall(refs, hyps):
    tot, hit = collections.Counter(), collections.Counter()
    for ref, hyp in zip(refs, hyps):
        wrong = collections.Counter()
        for tag, s, _ in Levenshtein.editops(ref, hyp):
            if tag in ("replace", "delete"):
                wrong[s] += 1
        for i, ch in enumerate(ref):
            if a0.is_han(ch):
                k = bucket(count(ch))
                tot[k] += 1
                hit[k] += not wrong[i]
    return {k: {"n": tot[k], "recall": 100 * hit[k] / tot[k]} for _, _, k in BUCKETS if tot[k]}


def decompose(pairs):
    out = collections.Counter()
    for a, b in pairs:
        if not (a0.is_han(a) and a0.is_han(b)):
            continue
        grp = "rare_ref" if count(a) < a0.RARE else "common_ref"
        out[(grp, "lookalike" if lookalike(a, b) else "other", direction(a, b))] += 1
    table = {}
    for grp in ("rare_ref", "common_ref"):
        n = sum(v for k, v in out.items() if k[0] == grp)
        table[grp] = {"n": n, **{f"{la}/{d}": 100 * out[(grp, la, d)] / max(1, n)
                                 for la in ("lookalike", "other")
                                 for d in ("to_more_frequent", "to_less_frequent", "equal")}}
    return table


def asymmetry(refs, hyps, min_errors=3):
    """Pairs {a,b} confused >= min_errors times: log rate ratio vs log train-frequency ratio."""
    occ = collections.Counter(ch for ref in refs for ch in ref)
    n = collections.Counter()
    for ref, hyp in zip(refs, hyps):
        for a, b in substitutions(ref, hyp):
            if a0.is_han(a) and a0.is_han(b):
                n[(a, b)] += 1
    rows = []
    for a, b in {tuple(sorted(k)) for k in n}:
        tot = n[(a, b)] + n[(b, a)]
        if tot < min_errors or occ[a] == 0 or occ[b] == 0:
            continue
        rate = math.log((n[(a, b)] + 0.5) / occ[a]) - math.log((n[(b, a)] + 0.5) / occ[b])  # a->b vs b->a
        freq = math.log((count(b) + 1) / (count(a) + 1))  # >0: b more frequent
        rows.append((rate, freq, lookalike(a, b)))
    res = {}
    for name, sel in (("all", lambda r: True), ("lookalike", lambda r: r[2]), ("other", lambda r: not r[2])):
        sub = [r for r in rows if sel(r)]
        if len(sub) < 10:
            res[name] = {"pairs": len(sub)}
            continue
        x, y = np.array([r[1] for r in sub]), np.array([r[0] for r in sub])
        rho = spearman(x, y)
        nonzero = x != 0
        toward = float(np.mean(np.sign(y[nonzero]) == np.sign(x[nonzero]))) * 100
        slope = float(np.polyfit(x, y, 1)[0])
        boot = bootstrap_stat(x, y, lambda u, v: spearman(u, v))
        res[name] = {"pairs": len(sub), "spearman_rho": rho, "rho_ci95": boot, "ols_slope": slope,
                     "pct_errors_flow_to_more_frequent": toward}
    return res


def spearman(x, y):
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def bootstrap_stat(x, y, fn, reps=1000, seed=0):
    rng = np.random.default_rng(seed)
    vals = [fn(x[i], y[i]) for i in (rng.integers(0, len(x), len(x)) for _ in range(reps))]
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def overproduction(refs, hyps, min_total=5):
    """Slope of log((pred+0.5)/(ref+0.5)) on log(train count + 1) over Han characters."""
    r = collections.Counter(ch for t in refs for ch in t if a0.is_han(ch))
    p = collections.Counter(ch for t in hyps for ch in t if a0.is_han(ch))
    chars = [c for c in set(r) | set(p) if r[c] + p[c] >= min_total]
    x = np.array([math.log(count(c) + 1) for c in chars])
    y = np.array([math.log((p[c] + 0.5) / (r[c] + 0.5)) for c in chars])
    slope = float(np.polyfit(x, y, 1)[0])
    return {"chars": len(chars), "slope": slope,
            "slope_ci95": bootstrap_stat(x, y, lambda u, v: float(np.polyfit(u, v, 1)[0]))}


def changes(refs, base, corr):
    """Substitution errors removed / introduced by a correction (multiset difference per line)."""
    removed, introduced = [], []
    for ref, b, c in zip(refs, base, corr):
        if b == c:
            continue
        sb, sc = collections.Counter(substitutions(ref, b)), collections.Counter(substitutions(ref, c))
        removed += list((sb - sc).elements())
        introduced += list((sc - sb).elements())
    return {"removed": decompose(removed), "introduced": decompose(introduced),
            "removed_n": len(removed), "introduced_n": len(introduced)}


def domain_selection(model):
    """Per-domain G-LA choice on val (same rule as A0-03 within the domain), then test once."""
    out = {}
    for dom in a0.DOMAINS:
        val = {dom: a0.load(model, "val", dom)}
        _, base = a0.evaluate(val, None)
        grid = [(cfg, a0.evaluate(val, cfg)[1]) for cfg in GRID_DOMAIN]
        ok = [(c, p) for c, p in grid if p["acc"] >= base["acc"] - 0.05]
        best = sorted(ok, key=lambda cp: (-cp[1]["rare_correct"], cp[0]["tau"], cp[0]["gamma"]))[0][0] if ok else None
        test = {dom: a0.load(model, "test", dom)}
        res = {}
        for name, cfg in (("B0", None), ("G*_pooled", GSTAR), ("G_domain", best)):
            if name == "G_domain" and cfg is None:
                continue
            _, pool = a0.evaluate(test, cfg)
            res[name] = {"cfg": cfg, "acc": pool["acc"], "rare_acc": pool["rare_acc"], "nonrare_acc": pool["nonrare_acc"]}
        out[dom] = {"val_selected": best, "val_frontier": [{"tau": c["tau"], "gamma": c["gamma"], "acc": p["acc"],
                                                            "rare_acc": p["rare_acc"]} for c, p in grid],
                    "val_B0": {"acc": base["acc"], "rare_acc": base["rare_acc"]}, "test": res}
    return out


def run(model, split):
    data = {dom: a0.load(model, split, dom) for dom in a0.DOMAINS}
    for dom, d in data.items():
        assert a0.strings(d, a0.decode(d, None)) == d["baseline"], (model, split, dom)
    refs = [t for dom in a0.DOMAINS for t in data[dom]["targets"]]
    base = [s for dom in a0.DOMAINS for s in a0.strings(data[dom], a0.decode(data[dom], None))]
    corr = [s for dom in a0.DOMAINS for s in a0.strings(data[dom], a0.decode(data[dom], GSTAR))]
    subs_base = [p for r, h in zip(refs, base) for p in substitutions(r, h)]
    return {"lines": len(refs),
            "a_recall": {"B0": aligned_recall(refs, base), "G*": aligned_recall(refs, corr)},
            "b_decomposition_B0": decompose(subs_base),
            "c_asymmetry_B0": asymmetry(refs, base),
            "d_gstar_changes": changes(refs, base, corr),
            "e_overproduction": {"B0": overproduction(refs, base), "G*": overproduction(refs, corr),
                                 "reference_check": overproduction(refs, refs)}}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dumps", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--models", nargs="+", default=["gtc", "b"])
    args = p.parse_args()
    a0.DUMPS = args.dumps
    args.out.mkdir(parents=True, exist_ok=True)
    result = {"configs": {"B0": None, "G*": GSTAR}}
    for model in args.models:
        for split in ("val", "test"):
            result[f"{model}_{split}"] = run(model, split)
            print("done", model, split, flush=True)
        result[f"{model}_f_domain_selection"] = domain_selection(model)
    (args.out / "c3_results.json").write_text(json.dumps(result, indent=1, ensure_ascii=False))
    print(json.dumps({k: v for k, v in result.items() if k.endswith("_val")}, indent=1, ensure_ascii=False)[:6000])


if __name__ == "__main__":
    main()
