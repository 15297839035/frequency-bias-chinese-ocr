"""C-08: ablations of the CTC prior correction, sensitivity analyses and extra metrics, on the existing dumps (CPU).

Main protocol: BCTR's official normalisation for every model (strict = secondary, --norm strict). Per model:
  variants  each selected on pooled val by A0's rule (pooled acc >= B0 - 0.05pp, then most correct rare lines;
            ties: smaller tau, then gamma) and read out once on test:
              seg      segment-level G-LA, gamma in {0.9, 0.99, 1.01}            (the method of the paper)
              seg_nogate  segment-level, gamma = 1.01 only
              seg_top2 / seg_top5  candidates restricted to the segment's top-2 / top-5 (gamma 0.9)
              frame    frame-level LA: every frame's non-blank top-16 log p - tau log pi, blank untouched,
                       then greedy + CTC collapse (ASR-style prior normalisation)
            plus tau = 1 (balanced strength) at gamma 0.9 / 1.01 without selection (descriptive)
  sensitivity (G* = tau 0.2, gamma 0.9, no reselection): rare threshold {10, 50, 100, 500}; look-alike definition;
            beta with min errors {1, 3, 5, 10}, with an intercept and with a complexity covariate (IDS components)
  metrics   CER, NED, per-bucket character recall / precision, precision of output rare characters, relative
            rare-line error reduction, recoverable ceiling (strict text only: share of B0 rare-character
            substitutions whose reference is among the segment's candidates, and share of those G* repairs)
Protocol: docs/experiments.md, entry "C-07 至 C-11 预注册" (C-08).

    python src/analysis/c08_correction_ablation.py --out results/C-08_ablation --norm official --workers 6
"""
import argparse
import collections
from concurrent.futures import ProcessPoolExecutor
import json
import math
from pathlib import Path
import sys

import numpy as np
from rapidfuzz.distance import Levenshtein

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import prior_correction as a0  # noqa: E402
from analysis import c03_confusion_analysis as c3  # noqa: E402
from analysis import c03_confusion_asymmetry as c3b  # noqa: E402
from analysis.c04_external_analysis import official_normalizer  # noqa: E402

EXP = a0.REPO / "results"
MODELS = [("gtc", EXP / "C-03_analysis/dumps"), ("b", EXP / "C-03_analysis/dumps"),
          ("off_T", EXP / "C-04_external/dumps"), ("off_S", EXP / "C-04_external/dumps"),
          ("off_B", EXP / "C-04_external/dumps"),
          ("svtrv2_ch", EXP / "C-04b_industrial/dumps"), ("repsvtr_ch", EXP / "C-04b_industrial/dumps"),
          ("crnn_s1", EXP / "C-05_crnn/dumps"), ("crnn_s2", EXP / "C-05_crnn/dumps")]
TAUS = (0.1, 0.2, 0.3, 0.5, 0.75, 1.0)
GSTAR = dict(tau=0.2, gamma=0.9)
IDC = set("⿰⿱⿲⿳⿴⿵⿶⿷⿸⿹⿺⿻")
FAMILIES = {
    "seg": [dict(kind="seg", tau=t, gamma=g, topk=0) for t in TAUS for g in (0.9, 0.99, 1.01)],
    "seg_nogate": [dict(kind="seg", tau=t, gamma=1.01, topk=0) for t in TAUS],
    "seg_top2": [dict(kind="seg", tau=t, gamma=0.9, topk=2) for t in TAUS],
    "seg_top5": [dict(kind="seg", tau=t, gamma=0.9, topk=5) for t in TAUS],
    "frame": [dict(kind="frame", tau=t, gamma=1.01, topk=0) for t in (0.05,) + TAUS],
}


# ----------------------------------------------------------------------------- decoding
def seg_rank(d):
    """Rank of each candidate within its segment by mean log p (0 = best)."""
    order = np.lexsort((-d["cand_score"], d["cand_seg"]))
    seg_sorted = d["cand_seg"][order]
    start = np.r_[0, np.flatnonzero(np.diff(seg_sorted)) + 1]
    sizes = np.diff(np.r_[start, len(order)])
    rank = np.empty(len(order), np.int64)
    rank[order] = np.arange(len(order)) - np.repeat(start, sizes)
    return rank


def seg_decode(d, cfg):
    score = d["cand_score"] - cfg["tau"] * a0.LOG_PRIOR[d["cand_cls"]]
    allowed = np.ones(len(score), bool)
    if cfg["topk"]:
        allowed &= d["cand_rank"] < cfg["topk"]
    allowed |= d["cand_is_star"]
    gated = d["seg_star_p"][d["cand_seg"]] >= cfg["gamma"]
    allowed &= ~gated | d["cand_is_star"]
    score = np.where(allowed, score, -np.inf)
    best = np.full(len(d["seg_star"]), -np.inf)
    np.maximum.at(best, d["cand_seg"], score)
    win = np.flatnonzero(score == best[d["cand_seg"]])
    first = np.unique(d["cand_seg"][win], return_index=True)[1]
    chosen = d["seg_star"].copy()
    chosen[d["cand_seg"][win[first]]] = d["cand_cls"][win[first]]
    return a0.strings(d, chosen)


def frame_decode(raw, tau):
    idx, logp, blank, offsets = raw
    adj = logp - np.float32(tau) * np.where(idx == 0, 0.0, a0.LOG_PRIOR[idx]).astype(np.float32)
    best = adj.argmax(1)
    lab = idx[np.arange(len(idx)), best]
    lab = np.where(blank > adj[np.arange(len(idx)), best], 0, lab)
    out = []
    for a, b in zip(offsets[:-1], offsets[1:]):
        seq = lab[a:b]
        keep = seq[(seq != 0) & np.r_[True, seq[1:] != seq[:-1]]]
        out.append(a0.normalize("".join(a0.CHARS[c] for c in keep)))
    return out


def hyps(data, raws, cfg):
    if cfg is None:
        return {dom: data[dom]["baseline"] for dom in a0.DOMAINS}
    if cfg["kind"] == "frame":
        return {dom: frame_decode(raws[dom], cfg["tau"]) for dom in a0.DOMAINS}
    return {dom: seg_decode(data[dom], cfg) for dom in a0.DOMAINS}


# ----------------------------------------------------------------------------- loading / metrics
def raw_rare(rows, threshold):
    return np.array([any(a0.is_han(ch) and a0.COUNT_OF.get(ch, 0) < threshold for ch in r["target"].replace(" ", "").lower())
                     for r in rows])


def load(dumps, tag, split):
    a0.DUMPS = dumps
    data, raws = {}, {}
    for dom in a0.DOMAINS:
        d = a0.load(tag, split, dom)
        d["rare"] = raw_rare(d["rows"], a0.RARE)
        d["cand_rank"] = seg_rank(d)
        data[dom] = d
        z = np.load(dumps / f"{tag}_{split}_{dom}.npz")
        raws[dom] = (z["topk_idx"].astype(np.int64), z["topk_logp"].astype(np.float32),
                     z["blank_logp"].astype(np.float32), z["offsets"])
    return data, raws


def flat(data, key):
    return [x for dom in a0.DOMAINS for x in data[dom][key]]


def line_metrics(data, h):
    refs, hyp = flat(data, "targets"), [x for dom in a0.DOMAINS for x in h[dom]]
    rare = np.concatenate([data[dom]["rare"] for dom in a0.DOMAINS])
    correct = np.array([a == b for a, b in zip(hyp, refs)])
    eds = np.array([Levenshtein.distance(a, b) for a, b in zip(hyp, refs)])
    lens = np.array([max(len(a), len(b), 1) for a, b in zip(hyp, refs)])
    return {"acc": 100 * correct.mean(), "rare_acc": 100 * correct[rare].mean(), "rare_correct": int(correct[rare].sum()),
            "nonrare_acc": 100 * correct[~rare].mean(), "cer": 100 * eds.sum() / sum(len(r) for r in refs),
            "ned": 100 * (1 - (eds / lens).mean()), "vec": correct, "rare": rare}


def select(val_data, val_raws, family):
    base = line_metrics(val_data, hyps(val_data, val_raws, None))
    grid = []
    for cfg in FAMILIES[family]:
        m = line_metrics(val_data, hyps(val_data, val_raws, cfg))
        grid.append({**cfg, **{k: m[k] for k in ("acc", "rare_acc", "rare_correct")}})
    ok = [g for g in grid if g["acc"] >= base["acc"] - 0.05]
    best = sorted(ok, key=lambda g: (-g["rare_correct"], g["tau"], g["gamma"]))[0] if ok else None
    return None if best is None else {k: best[k] for k in ("kind", "tau", "gamma", "topk")}, grid


def char_pr(refs, hyps_):
    """Per training-count bucket: recall of reference characters and precision of output characters (aligned)."""
    ref_n, hyp_n, match = collections.Counter(), collections.Counter(), collections.Counter()
    rare_out, rare_ok = 0, 0
    for r, h in zip(refs, hyps_):
        for ch in r:
            if a0.is_han(ch):
                ref_n[c3.bucket(c3.count(ch))] += 1
        for ch in h:
            if a0.is_han(ch):
                hyp_n[c3.bucket(c3.count(ch))] += 1
                rare_out += c3.count(ch) < a0.RARE
        for op in Levenshtein.opcodes(r, h):
            if op.tag != "equal":
                continue
            for ch in r[op.src_start:op.src_end]:
                if a0.is_han(ch):
                    match[c3.bucket(c3.count(ch))] += 1
                    rare_ok += c3.count(ch) < a0.RARE
    names = [b[2] for b in c3.BUCKETS]
    return {"recall": {b: 100 * match[b] / ref_n[b] if ref_n[b] else None for b in names},
            "precision": {b: 100 * match[b] / hyp_n[b] if hyp_n[b] else None for b in names},
            "rare_output_precision": 100 * rare_ok / max(rare_out, 1), "rare_outputs": rare_out}


# ----------------------------------------------------------------------------- mechanism sensitivity
def ids_components():
    out = {}
    for line in open(c3.IDS, encoding="utf-8-sig"):
        ch, sep, seq = line.rstrip("\n").partition(":")
        if sep:
            out[ch] = sum(t not in IDC for t in seq.split())
    return out


def fit(rows, extra=None, intercept=False):
    """Conditional binomial MLE: logit P(a->b) = logit p0 + beta x (+ eta * extra) (+ alpha); b = more frequent."""
    from scipy.optimize import minimize
    o = c3b.oriented(rows)
    k, N, p0, x = c3b.arrays(o)
    cols = [x]
    if extra is not None:
        cols.append(np.array([extra(r["a"]) - extra(r["b"]) for r in o], float))
    if intercept:
        cols.append(np.ones(len(o)))
    X = np.stack(cols, 1)
    keep = np.isfinite(X).all(1)
    X, k, N, off = X[keep], k[keep], N[keep], np.log(p0[keep]) - np.log1p(-p0[keep])

    def nll(w):
        z = off + X @ w
        return float(np.sum(k * np.logaddexp(0, -z) + (N - k) * np.logaddexp(0, z)))
    est = minimize(nll, np.zeros(X.shape[1]), method="BFGS").x
    rng = np.random.default_rng(0)
    boots = []
    for _ in range(300):
        i = rng.integers(0, len(k), len(k))

        def nll_b(w, i=i):
            z = off[i] + X[i] @ w
            return float(np.sum(k[i] * np.logaddexp(0, -z) + (N[i] - k[i]) * np.logaddexp(0, z)))
        boots.append(minimize(nll_b, est, method="BFGS").x)
    boots = np.array(boots)
    return {"pairs": int(len(k)), "estimate": est.tolist(),
            "ci95": [[float(np.percentile(boots[:, j], 2.5)), float(np.percentile(boots[:, j], 97.5))]
                     for j in range(X.shape[1])]}


def lookalike_shares(refs, hyps_, seed=0):
    """Rare-reference Han substitutions: share that is look-alike under several definitions, and the same for
    random substitutes drawn from the output character distribution (base rate)."""
    subs = [(a, b) for r, h in zip(refs, hyps_) for a, b in c3.substitutions(r, h)
            if a0.is_han(a) and a0.is_han(b) and c3.count(a) < a0.RARE]
    out_chars = [ch for h in hyps_ for ch in h if a0.is_han(ch)]
    rng = np.random.default_rng(seed)
    rand = [(a, out_chars[i]) for (a, _), i in zip(subs, rng.integers(0, len(out_chars), len(subs)))]

    def glyph(a, b, k):
        ia, ib = c3.CHAR_ID.get(a), c3.CHAR_ID.get(b)
        return ia is not None and ib is not None and bool((c3.GLYPH_NB[ia][:k] == ib).any() or (c3.GLYPH_NB[ib][:k] == ia).any())

    defs = {"ids_shared_component": lambda a, b: bool(c3.IDS_COMP.get(a, set()) & c3.IDS_COMP.get(b, set())),
            "glyph_top10": lambda a, b: glyph(a, b, 10), "glyph_top50": lambda a, b: glyph(a, b, 50),
            "union_used_in_paper": c3.lookalike}
    return {"substitutions": len(subs),
            **{name: {"observed": 100 * np.mean([f(a, b) for a, b in subs]),
                      "random_substitute": 100 * np.mean([f(a, b) for a, b in rand])} for name, f in defs.items()}}


def ceiling(tag, dumps, split):
    """Strict text: B0 rare-character substitutions whose reference is among the segment's candidates; G* repairs."""
    a0.DUMPS = dumps
    saved = a0.normalize
    a0.normalize = lambda t: t.replace(" ", "").lower()
    total = recoverable = repaired = 0
    try:
        for dom in a0.DOMAINS:
            d = a0.load(tag, split, dom)
            gstar = a0.decode(d, dict(method="G", k=0, **GSTAR))
            seg_of_line = collections.defaultdict(list)
            for s, line in enumerate(d["seg_line"]):
                seg_of_line[int(line)].append(s)
            cands = collections.defaultdict(set)
            for s, c in zip(d["cand_seg"], d["cand_cls"]):
                cands[int(s)].add(a0.CHARS[int(c)])
            for line, (ref, hyp) in enumerate(zip(d["targets"], d["baseline"])):
                segs = seg_of_line[line]
                if len(segs) != len(hyp):
                    continue
                for tag_, i, j in Levenshtein.editops(ref, hyp):
                    if tag_ != "replace" or not a0.is_han(ref[i]) or c3.count(ref[i]) >= a0.RARE:
                        continue
                    total += 1
                    if ref[i] in cands[segs[j]]:
                        recoverable += 1
                        repaired += a0.CHARS[int(gstar[segs[j]])] == ref[i]
    finally:
        a0.normalize = saved
    return {"rare_substitutions": total, "recoverable_pct": 100 * recoverable / max(total, 1),
            "repaired_by_gstar_pct_of_recoverable": 100 * repaired / max(recoverable, 1)}


# ----------------------------------------------------------------------------- per model
def run_model(job):
    tag, dumps, norm = job
    if norm == "official":
        a0.normalize = official_normalizer()
    val, val_raw = load(dumps, tag, "val")
    test, test_raw = load(dumps, tag, "test")
    for dom in a0.DOMAINS:  # B0 sanity: frame-level decoding at tau 0 equals the greedy baseline
        a0.check_baseline(frame_decode(test_raw[dom], 0.0), test[dom]["baseline"], (tag, dom))
    res = {"tag": tag, "norm": norm, "variants": {}}
    b0 = line_metrics(test, hyps(test, test_raw, None))
    res["B0"] = {k: v for k, v in b0.items() if k not in ("vec", "rare")}
    for fam in FAMILIES:
        chosen, grid = select(val, val_raw, fam)
        entry = {"selected": chosen, "val_grid": grid}
        if chosen:
            m = line_metrics(test, hyps(test, test_raw, chosen))
            fixed = int((m["vec"] & ~b0["vec"] & b0["rare"]).sum())
            broken = int((~m["vec"] & b0["vec"] & b0["rare"]).sum())
            entry["test"] = {**{k: v for k, v in m.items() if k not in ("vec", "rare")},
                             "acc_change": m["acc"] - b0["acc"], "rare_gain": m["rare_acc"] - b0["rare_acc"],
                             "rare_fixed": fixed, "rare_broken": broken, "rare_mcnemar_p": a0.exact_mcnemar(fixed, broken),
                             "rare_rel_error_reduction": 100 * (m["rare_acc"] - b0["rare_acc"]) / (100 - b0["rare_acc"])}
        res["variants"][fam] = entry
        print(tag, fam, chosen, {k: round(v, 3) for k, v in entry.get("test", {}).items()
                                 if k in ("acc_change", "rare_gain")}, flush=True)
    for g in (0.9, 1.01):  # balanced strength, no selection
        m = line_metrics(test, hyps(test, test_raw, dict(kind="seg", tau=1.0, gamma=g, topk=0)))
        res["variants"][f"tau1_g{g}"] = {"test": {"acc_change": m["acc"] - b0["acc"], "rare_gain": m["rare_acc"] - b0["rare_acc"]}}
    # G* sensitivity and metrics (test)
    gcfg = dict(kind="seg", topk=0, **GSTAR)
    hg = hyps(test, test_raw, gcfg)
    refs = flat(test, "targets")
    hb = [x for dom in a0.DOMAINS for x in test[dom]["baseline"]]
    hgs = [x for dom in a0.DOMAINS for x in hg[dom]]
    gm = line_metrics(test, hg)
    res["gstar"] = {"acc_change": gm["acc"] - b0["acc"], "rare_gain": gm["rare_acc"] - b0["rare_acc"],
                    "cer": [b0["cer"], gm["cer"]], "ned": [b0["ned"], gm["ned"]],
                    "char_pr_B0": char_pr(refs, hb), "char_pr_Gstar": char_pr(refs, hgs)}
    rows_all = [r for dom in a0.DOMAINS for r in test[dom]["rows"]]
    res["rare_threshold"] = {}
    for th in (10, 50, 100, 500):
        rare = raw_rare(rows_all, th)
        cb = np.array([a == b for a, b in zip(hb, refs)])
        cg = np.array([a == b for a, b in zip(hgs, refs)])
        res["rare_threshold"][th] = {"lines": int(rare.sum()), "B0": 100 * cb[rare].mean(), "Gstar": 100 * cg[rare].mean(),
                                     "gain": 100 * (cg[rare].mean() - cb[rare].mean())}
    res["lookalike"] = lookalike_shares(refs, hb)
    comp = ids_components()
    rows = {m: c3b.pairs_from(refs, hb, min_errors=m) for m in (1, 3, 5, 10)}
    res["beta_min_errors"] = {}
    for m, r in rows.items():
        a = c3b.analyse(r)
        res["beta_min_errors"][m] = {"pairs": a["pairs"]} if "beta" not in a else {
            "pairs": a["pairs"], "T/E": a["toward_more_frequent"]["ratio"], "beta": a["beta"]["estimate"],
            "ci95": a["beta"]["ci95"]}
    res["beta_intercept"] = fit(rows[3], intercept=True)
    res["beta_complexity"] = fit(rows[3], extra=lambda ch: comp.get(ch, np.nan))
    res["ceiling"] = ceiling(tag, dumps, "test")
    print(tag, "done", flush=True)
    return tag, res


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--norm", choices=("official", "strict"), default="official")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--tags", nargs="*")
    args = p.parse_args()
    jobs = [(t, d, args.norm) for t, d in MODELS if (d / f"{t}_val_scene.npz").exists() and (not args.tags or t in args.tags)]
    with ProcessPoolExecutor(args.workers) as ex:
        results = dict(ex.map(run_model, jobs))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"c8_results_{args.norm}.json").write_text(json.dumps(results, indent=1, default=float, ensure_ascii=False))
    L = [f"# C-08 ablations ({args.norm} normalisation; test, each variant selected on val)", "",
         "| model | B0 acc / rare | " + " | ".join(FAMILIES) + " | tau=1 (g0.9) |", "|---|---|" + "---|" * (len(FAMILIES) + 1)]
    for tag, r in results.items():
        cells = []
        for fam in FAMILIES:
            t = r["variants"][fam].get("test")
            s = r["variants"][fam]["selected"]
            cells.append("infeasible" if t is None else
                         f"{t['rare_gain']:+.2f} / {t['acc_change']:+.2f} (τ{s['tau']}, γ{s['gamma']})")
        t1 = r["variants"]["tau1_g0.9"]["test"]
        L.append(f"| {tag} | {r['B0']['acc']:.2f} / {r['B0']['rare_acc']:.2f} | " + " | ".join(cells)
                 + f" | {t1['rare_gain']:+.2f} / {t1['acc_change']:+.2f} |")
    L += ["", "Cells: rare-line gain / overall change (pp) of the val-selected config.", ""]
    for tag, r in results.items():
        L.append(f"## {tag}")
        g = r["gstar"]
        L.append(f"- G*: CER {g['cer'][0]:.2f} -> {g['cer'][1]:.2f}, NED {g['ned'][0]:.2f} -> {g['ned'][1]:.2f}; "
                 f"rare-output precision {g['char_pr_B0']['rare_output_precision']:.1f} -> "
                 f"{g['char_pr_Gstar']['rare_output_precision']:.1f}")
        L.append("- recall by bucket B0 -> G*: " + ", ".join(
            f"{b} {g['char_pr_B0']['recall'][b] or 0:.1f}->{g['char_pr_Gstar']['recall'][b] or 0:.1f}" for b in g["char_pr_B0"]["recall"]))
        L.append("- precision by bucket B0 -> G*: " + ", ".join(
            f"{b} {g['char_pr_B0']['precision'][b] or 0:.1f}->{g['char_pr_Gstar']['precision'][b] or 0:.1f}"
            for b in g["char_pr_B0"]["precision"]))
        L.append("- rare threshold (lines, B0 -> G*): " + ", ".join(
            f"<{th}: {v['lines']}, {v['B0']:.2f}->{v['Gstar']:.2f}" for th, v in r["rare_threshold"].items()))
        la = r["lookalike"]
        L.append(f"- look-alike share of {la['substitutions']} rare substitutions (observed / random substitute): " + ", ".join(
            f"{k} {v['observed']:.1f}/{v['random_substitute']:.1f}" for k, v in la.items() if k != "substitutions"))
        L.append("- beta by min errors: " + ", ".join(
            f"N>={m}: {v['pairs']} pairs" + (f", T/E {v['T/E']:.2f}, beta {v['beta']:.2f}" if "beta" in v else "")
            for m, v in r["beta_min_errors"].items()))
        bi, bc = r["beta_intercept"], r["beta_complexity"]
        L.append(f"- beta with intercept {bi['estimate'][0]:.2f} {bi['ci95'][0]} (alpha {bi['estimate'][1]:.2f} {bi['ci95'][1]}); "
                 f"with IDS-complexity {bc['estimate'][0]:.2f} {bc['ci95'][0]} (eta {bc['estimate'][1]:.3f} {bc['ci95'][1]})")
        c = r["ceiling"]
        L.append(f"- ceiling: {c['rare_substitutions']} rare substitutions, {c['recoverable_pct']:.1f}% recoverable, "
                 f"G* repairs {c['repaired_by_gstar_pct_of_recoverable']:.1f}% of those")
        L.append("")
    (args.out / f"report_{args.norm}.md").write_text("\n".join(L) + "\n")
    print("\n".join(L[:len(results) + 6]))


if __name__ == "__main__":
    main()
