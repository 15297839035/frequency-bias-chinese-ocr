"""Plan v4, CPU analyses V4A-1 to V4A-5 and V4A-7 (plan: ly/commit.md, "规划 v4", section A). Descriptive; nothing
here selects a parameter that is later evaluated on test.

`lines` -- BCTR test, uncorrected output (B0), official normalisation, per model:
  A4  T/E (errors toward the more frequent member of a confusable pair over the symmetric expectation) with a
      bootstrap CI over pairs, and beta (C-03b regression) with its pair-bootstrap CI;
  A2  Bayes reference: each pair is an equal-variance Gaussian two-class problem decided with the training prior
      ratio (Bayes rule for the training distribution); d' is calibrated so that the pair's expected number of errors
      equals the observed one; the same regression on its expected errors gives beta_Bayes and T/E_Bayes;
      beta - beta_Bayes with a paired pair-bootstrap CI (also beta_Bayes refitted on simulated binomial counts);
  A3  free coefficient on logit p0: logit P = alpha * logit p0 + beta * x;
  A5  look-alike definitions: share of rare-character substitutions that are look-alikes (IDS component shared,
      IDS Jaccard >= 0.5, glyph top-10 / top-50, the union used in the paper) vs random substitutes; T/E by group;
  A1  (CTC only) rare-character substitutions split into: correct character not among the segment's candidates /
      a candidate and the substitute more frequent / a candidate otherwise; share repaired by the segment-level
      correction at tau in {0.2, 0.5, 1, 2} (exact arg max over the candidates). Strict text for the alignment, as
      C-08; pairs that are the same character under the official normalisation (e.g. traditional/simplified) skipped.
  Models: --ctc tag=dumpdir (a0 dumps), --smtr tag=predsdir (C-09 format), --attn tag=predsdir@countsdir (C-16:
  one model per domain; its own domain's training counts define frequency and rarity).
`hwdb` -- A2 to A4 for the HWDB classifiers (ICDAR 2013 test, default predictions; frequency = the run's own training
  images), groups given as name=run_dir,run_dir,...
`hwdb_tau` -- A7: test distributions q_c proportional to (training images + 1)^lambda (lambda 0 = class-balanced,
  1 = the training distribution) and to (BCTR frequency + 1)^lambda; per model the accuracy-maximising effective
  strength, its plateau (within 0.05 points), the strength a calibrated model would need (1 - lambda for the first
  family), and the val-selected strength read on test with a class-stratified bootstrap CI against the model's
  uncorrected decision (tau_eff = tau_train).

    python ly/scripts/v4a_direction.py lines --out ly/exp/V4A_cpu/lines --ctc gtc=ly/exp/C-03_analysis/dumps ...
    python ly/scripts/v4a_direction.py hwdb --out ly/exp/V4A_cpu/hwdb --data ly/data/hwdb \
        --group nat=ly/exp/C-01b_hwdb_seeds/dumps_all/ce_s0,... --group rev=...
    python ly/scripts/v4a_direction.py hwdb_tau --out ly/exp/V4A_cpu/hwdb_tau \
        --lt ly/exp/C-01b_hwdb_seeds/dumps_all --nat ly/exp/C-01n_hwdb_nat/dumps
"""
import argparse
import collections
import json
import math
from multiprocessing import Pool
from pathlib import Path
import sys

import numpy as np
from rapidfuzz.distance import Levenshtein
from scipy.optimize import minimize
from scipy.stats import norm as gauss

sys.path.insert(0, str(Path(__file__).resolve().parent))
import a0_prior_correction as a0  # noqa: E402
import c3_analysis as c3  # noqa: E402
import c3b_asymmetry as c3b  # noqa: E402
from c4_analysis import official_normalizer  # noqa: E402

DGRID = np.geomspace(0.02, 25.0, 1000)
REPAIR_TAUS = (0.2, 0.5, 1.0, 2.0)
RARITY = ((0, 1, "0"), (1, 10, "1-9"), (10, 100, "10-99"))
OFF = None  # official normaliser, created per process


def off():
    global OFF
    if OFF is None:
        OFF = official_normalizer()
    return OFF


# ----------------------------------------------------------------------------- pair statistics
def pairs_from(refs, hyps, count, min_errors=3):
    """As c3b.pairs_from, with the frequency (training count) function as an argument."""
    occ = collections.Counter(ch for ref in refs for ch in ref)
    n = collections.Counter()
    for ref, hyp in zip(refs, hyps):
        for a, b in c3.substitutions(ref, hyp):
            if a0.is_han(a) and a0.is_han(b):
                n[(a, b)] += 1
    rows = []
    for a, b in {tuple(sorted(k)) for k in n}:
        N = n[(a, b)] + n[(b, a)]
        if N < min_errors or occ[a] == 0 or occ[b] == 0:
            continue
        rows.append(dict(a=a, b=b, n_ab=n[(a, b)], n_ba=n[(b, a)], occ_a=occ[a], occ_b=occ[b],
                         cnt_a=count(a), cnt_b=count(b), look=c3.lookalike(a, b)))
    return rows


def calibrate(oa, ob, N, L):
    """Per pair: a at -d/2, b at +d/2 (unit variance), decide b iff d*s + L > 0 with L = log prior ratio (b more
    frequent, L > 0). Returns d', P(a->b), P(b->a) with oa*P(a->b) + ob*P(b->a) = N for the largest such d'
    (grid, 0.7% steps); if no d' gives N errors, the d' with the most errors."""
    d = DGRID[None, :]
    pab = gauss.cdf(-d / 2 + L[:, None] / d)
    pba = gauss.cdf(-d / 2 - L[:, None] / d)
    E = oa[:, None] * pab + ob[:, None] * pba
    ok = E >= N[:, None]
    j = np.where(ok.any(1), ok.shape[1] - 1 - np.argmax(ok[:, ::-1], 1), np.argmax(E, 1))
    r = np.arange(len(j))
    return DGRID[j], pab[r, j], pba[r, j], ok.any(1)


def fit2(k, N, p0, x):
    off_ = np.log(p0) - np.log1p(-p0)

    def nll(w):
        z = w[0] * off_ + w[1] * x
        return float(np.sum(k * np.logaddexp(0, -z) + (N - k) * np.logaddexp(0, z)))
    return minimize(nll, np.array([1.0, 1.0]), method="BFGS").x


def ci(v):
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def direction(rows, reps=1000, seed=0):
    o = c3b.oriented(rows)
    res = {"pairs": len(o)}
    if len(o) < 10:
        return res
    k, N, p0, x = c3b.arrays(o)
    oa = np.array([r["occ_a"] for r in o], float)
    ob = np.array([r["occ_b"] for r in o], float)
    d, pab, pba, reached = calibrate(oa, ob, N, x)
    pB = oa * pab / np.maximum(oa * pab + ob * pba, 1e-300)
    E, KB = N * p0, N * pB
    beta, beta_b = c3b.fit_beta(k, N, p0, x), c3b.fit_beta(KB, N, p0, x)
    a2 = fit2(k, N, p0, x)
    rng = np.random.default_rng(seed)
    sim = [c3b.fit_beta(rng.binomial(N.astype(np.int64), pB).astype(float), N, p0, x) for _ in range(200)]
    bt = []
    for _ in range(reps):
        i = rng.integers(0, len(o), len(o))
        b1, b2 = c3b.fit_beta(k[i], N[i], p0[i], x[i]), c3b.fit_beta(KB[i], N[i], p0[i], x[i])
        bt.append((k[i].sum() / E[i].sum(), KB[i].sum() / E[i].sum(), b1, b2, b1 - b2,
                   k[i].sum() / KB[i].sum()))
    bt = np.array(bt)
    a2b = np.array([fit2(k[i], N[i], p0[i], x[i]) for i in (rng.integers(0, len(o), len(o)) for _ in range(200))])
    res.update({
        "errors": float(N.sum()),
        "T/E": float(k.sum() / E.sum()), "T/E_ci95": ci(bt[:, 0]),
        "pct_toward_frequent": float(100 * k.sum() / N.sum()), "pct_symmetric": float(100 * E.sum() / N.sum()),
        "beta": beta, "beta_ci95": ci(bt[:, 2]),
        "bayes": {"T/E": float(KB.sum() / E.sum()), "T/E_ci95": ci(bt[:, 1]),
                  "pct_toward_frequent": float(100 * KB.sum() / N.sum()),
                  "beta": beta_b, "beta_ci95": ci(bt[:, 3]), "beta_simulated_median": float(np.median(sim)),
                  "beta_simulated_ci95": ci(sim),
                  "beta_minus_bayes": beta - beta_b, "beta_minus_bayes_ci95": ci(bt[:, 4]),
                  "observed_over_bayes_toward_frequent": float(k.sum() / KB.sum()),
                  "observed_over_bayes_ci95": ci(bt[:, 5]),
                  "dprime_median": float(np.median(d)), "dprime_iqr": [float(np.percentile(d, 25)),
                                                                     float(np.percentile(d, 75))],
                  "pairs_calibrated_exactly": int(reached.sum())},
        "free_alpha": {"alpha": float(a2[0]), "beta": float(a2[1]), "alpha_ci95": ci(a2b[:, 0]),
                       "beta_ci95": ci(a2b[:, 1])}})
    return res


# ----------------------------------------------------------------------------- look-alike definitions
def jaccard(a, b):
    A, B = c3.IDS_COMP.get(a), c3.IDS_COMP.get(b)
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)


def glyph(a, b, k):
    ia, ib = c3.CHAR_ID.get(a), c3.CHAR_ID.get(b)
    return ia is not None and ib is not None and bool((c3.GLYPH_NB[ia][:k] == ib).any()
                                                      or (c3.GLYPH_NB[ib][:k] == ia).any())


DEFS = {"ids_any": lambda a, b: bool(c3.IDS_COMP.get(a, set()) & c3.IDS_COMP.get(b, set())),
        "ids_jaccard50": lambda a, b: jaccard(a, b) >= 0.5,
        "glyph10": lambda a, b: glyph(a, b, 10), "glyph50": lambda a, b: glyph(a, b, 50),
        "union_paper": c3.lookalike,
        "strict": lambda a, b: jaccard(a, b) >= 0.5 or glyph(a, b, 10)}


def lookalike_block(refs, hyps, count, rows, seed=0):
    subs = [(a, b) for r, h in zip(refs, hyps) for a, b in c3.substitutions(r, h)
            if a0.is_han(a) and a0.is_han(b) and count(a) < a0.RARE]
    out_chars = [ch for h in hyps for ch in h if a0.is_han(ch)]
    rng = np.random.default_rng(seed)
    rand = [(a, out_chars[i]) for (a, _), i in zip(subs, rng.integers(0, len(out_chars), len(subs)))]
    res = {"rare_substitutions": len(subs),
           "share": {name: {"observed": float(100 * np.mean([f(a, b) for a, b in subs])),
                            "random_substitute": float(100 * np.mean([f(a, b) for a, b in rand]))}
                     for name, f in DEFS.items()}}
    res["direction_by_group"] = {}
    for name in ("union_paper", "strict"):
        f = DEFS[name]
        for grp, keep in (("lookalike", True), ("other", False)):
            sel = [r for r in rows if f(r["a"], r["b"]) == keep]
            o = c3b.oriented(sel)
            if len(o) < 10:
                res["direction_by_group"][f"{name}/{grp}"] = {"pairs": len(o)}
                continue
            k, N, p0, x = c3b.arrays(o)
            res["direction_by_group"][f"{name}/{grp}"] = {"pairs": len(o), "errors": float(N.sum()),
                                                          "T/E": float(k.sum() / (N * p0).sum()),
                                                          "beta": c3b.fit_beta(k, N, p0, x)}
    return res


# ----------------------------------------------------------------------------- A1 decomposition (CTC)
def rarity(n):
    return next(name for lo, hi, name in RARITY if lo <= n < hi)


def decompose(data):
    """data: domain -> a0.load() result with strict normalisation."""
    norm = off()
    tally = collections.Counter()
    for dom, d in data.items():
        nseg = len(d["seg_star"])
        starts = np.searchsorted(d["cand_seg"], np.arange(nseg + 1))
        line_starts = np.searchsorted(d["seg_line"], np.arange(len(d["rows"]) + 1))
        for line, (ref, hyp) in enumerate(zip(d["targets"], d["baseline"])):
            if ref == hyp:
                continue
            segs = range(line_starts[line], line_starts[line + 1])
            if len(segs) != len(hyp):
                tally["lines_skipped_alignment"] += 1
                continue
            for tag, i, j in Levenshtein.editops(ref, hyp):
                if tag != "replace":
                    continue
                a, b = ref[i], hyp[j]
                if not (a0.is_han(a) and a0.is_han(b)) or c3.count(a) >= a0.RARE:
                    continue
                if norm(a) == norm(b):
                    tally["skipped_same_under_official_norm"] += 1
                    continue
                s = segs[j]
                cls = d["cand_cls"][starts[s]:starts[s + 1]]
                sc = d["cand_score"][starts[s]:starts[s + 1]]
                ia = c3.CHAR_ID.get(a)
                grp = rarity(c3.count(a))
                tally[("all", grp)] += 1
                if ia is None or not (cls == ia).any():
                    tally[("not_candidate", grp)] += 1
                    continue
                kind = "candidate_freq_favoured" if c3.count(b) > c3.count(a) else "candidate_other"
                tally[(kind, grp)] += 1
                lp = a0.LOG_PRIOR[cls]
                for t in REPAIR_TAUS:
                    if cls[int(np.argmax(sc - t * lp))] == ia:
                        tally[(f"repaired_tau{t}", kind, grp)] += 1
    groups = [g for _, _, g in RARITY] + ["total"]

    def get(*key):
        if key[-1] == "total":
            return sum(tally[key[:-1] + (g,)] for _, _, g in RARITY)
        return tally[key]
    out = {"skipped_same_under_official_norm": tally["skipped_same_under_official_norm"],
           "lines_skipped_alignment": tally["lines_skipped_alignment"], "by_rarity": {}}
    for g in groups:
        n = get("all", g)
        e = {"rare_substitutions": n}
        for kind in ("not_candidate", "candidate_freq_favoured", "candidate_other"):
            e[f"pct_{kind}"] = 100 * get(kind, g) / max(n, 1)
        for kind in ("candidate_freq_favoured", "candidate_other"):
            m = get(kind, g)
            e[f"repaired_pct_of_{kind}"] = {str(t): 100 * get(f"repaired_tau{t}", kind, g) / max(m, 1)
                                            for t in REPAIR_TAUS}
        e["repaired_pct_of_all"] = {str(t): 100 * (get(f"repaired_tau{t}", "candidate_freq_favoured", g)
                                                   + get(f"repaired_tau{t}", "candidate_other", g)) / max(n, 1)
                                    for t in REPAIR_TAUS}
        out["by_rarity"][g] = e
    return out


# ----------------------------------------------------------------------------- per model
def run_line_model(job):
    kind, tag, src = job
    norm = off()
    if kind == "ctc":
        a0.DUMPS = Path(src)
        a0.normalize = lambda t: t.replace(" ", "").lower()  # strict, for the A1 alignment
        data = {dom: a0.load(tag, "test", dom) for dom in a0.DOMAINS}
        refs = [norm(r["target"]) for dom in a0.DOMAINS for r in data[dom]["rows"]]
        hyps = [norm(r["prediction"]) for dom in a0.DOMAINS for r in data[dom]["rows"]]
        rows = pairs_from(refs, hyps, c3.count)
        res = {"kind": kind, "lines": len(refs), "direction": direction(rows),
               "lookalike": lookalike_block(refs, hyps, c3.count, rows), "decomposition": decompose(data)}
    elif kind == "smtr":
        refs, hyps = [], []
        for dom in a0.DOMAINS:
            for line in open(Path(src) / f"{tag}_test_{dom}.jsonl", encoding="utf-8"):
                r = json.loads(line)
                refs.append(norm(r["target"]))
                hyps.append(norm(r["preds"]["B0"]))
        rows = pairs_from(refs, hyps, c3.count)
        res = {"kind": kind, "lines": len(refs), "direction": direction(rows),
               "lookalike": lookalike_block(refs, hyps, c3.count, rows)}
    else:  # attn: one model per domain, own domain's counts
        preds, counts = src.split("@")
        rows, refs_all, hyps_all, look = [], [], [], {}
        for dom in a0.DOMAINS:
            cnt = json.loads((Path(counts) / f"train_counts_{dom}.json").read_text(encoding="utf-8"))
            count = lambda ch, cnt=cnt: cnt.get(ch, 0)  # noqa: E731
            refs, hyps = [], []
            for line in open(Path(preds) / f"{tag}_test_{dom}.jsonl", encoding="utf-8"):
                r = json.loads(line)
                refs.append(norm(r["target"]))
                hyps.append(norm(r["preds"]["B0"]))
            dom_rows = pairs_from(refs, hyps, count)
            rows += dom_rows
            look[dom] = lookalike_block(refs, hyps, count, dom_rows)
            refs_all += refs
            hyps_all += hyps
        res = {"kind": kind, "lines": len(refs_all), "direction": direction(rows), "lookalike_by_domain": look}
    print("done", kind, tag, flush=True)
    return f"{kind}:{tag}", res


def lines(args):
    jobs = [("ctc", *s.split("=", 1)) for s in args.ctc] + [("smtr", *s.split("=", 1)) for s in args.smtr] \
        + [("attn", *s.split("=", 1)) for s in args.attn]
    with Pool(min(args.procs, len(jobs))) as pool:
        results = dict(pool.map(run_line_model, jobs, chunksize=1))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "v4a_lines.json").write_text(json.dumps(results, indent=1, default=float, ensure_ascii=False))
    rep = ["# V4A lines: BCTR test, B0, official normalisation", "",
           "| model | pairs | T/E [CI] | T/E Bayes | beta [CI] | beta Bayes (sim) | beta - Bayes [CI] | free alpha, beta |",
           "|---|---:|---|---:|---|---|---|---|"]
    for name, r in results.items():
        d = r["direction"]
        if "beta" not in d:
            rep.append(f"| {name} | {d['pairs']} | | | | | | |")
            continue
        b = d["bayes"]
        rep.append(f"| {name} | {d['pairs']} | {d['T/E']:.2f} [{d['T/E_ci95'][0]:.2f}, {d['T/E_ci95'][1]:.2f}] | "
                   f"{b['T/E']:.2f} | {d['beta']:.2f} [{d['beta_ci95'][0]:.2f}, {d['beta_ci95'][1]:.2f}] | "
                   f"{b['beta']:.2f} ({b['beta_simulated_median']:.2f}) | {b['beta_minus_bayes']:+.2f} "
                   f"[{b['beta_minus_bayes_ci95'][0]:+.2f}, {b['beta_minus_bayes_ci95'][1]:+.2f}] | "
                   f"{d['free_alpha']['alpha']:.2f}, {d['free_alpha']['beta']:.2f} |")
    rep += ["", "## A1 decomposition (CTC; rare-character substitutions, all rarity bins)", "",
            "| model | n | not cand. % | cand.+freq % | cand.+other % | repaired at tau 0.2 / 0.5 / 1 / 2 (% of all) |",
            "|---|---:|---:|---:|---:|---|"]
    for name, r in results.items():
        if "decomposition" in r:
            e = r["decomposition"]["by_rarity"]["total"]
            rep.append(f"| {name} | {e['rare_substitutions']} | {e['pct_not_candidate']:.1f} | "
                       f"{e['pct_candidate_freq_favoured']:.1f} | {e['pct_candidate_other']:.1f} | "
                       + " / ".join(f"{v:.1f}" for v in e["repaired_pct_of_all"].values()) + " |")
    rep += ["", "## A5 look-alike share of rare-character substitutions (observed / random substitute, %)", ""]
    for name, r in results.items():
        blocks = [r["lookalike"]] if "lookalike" in r else list(r.get("lookalike_by_domain", {}).values())
        for blk in blocks:
            rep.append(f"- {name}: " + "; ".join(f"{k} {v['observed']:.1f}/{v['random_substitute']:.1f}"
                                                for k, v in blk["share"].items())
                       + " | T/E by group: " + "; ".join(f"{k} {v.get('T/E', float('nan')):.2f} ({v['pairs']})"
                                                         for k, v in blk["direction_by_group"].items()))
    (args.out / "report.md").write_text("\n".join(rep) + "\n", encoding="utf-8")
    print("\n".join(rep))


# ----------------------------------------------------------------------------- HWDB
def hwdb(args):
    import c7_flip_direction as c7
    classes = (Path(args.data) / "classes.txt").read_text(encoding="utf-8")
    results, rep = {}, ["# V4A hwdb: ICDAR 2013 test, default predictions; frequency = the run's training images", "",
                        "| group/run | pairs | T/E [CI] | T/E Bayes | beta [CI] | beta Bayes (sim) | beta - Bayes [CI] |",
                        "|---|---:|---|---:|---|---|---|"]
    for spec in args.group:
        name, runs = spec.split("=", 1)
        pooled = []
        for run in map(Path, runs.split(",")):
            pred, labels, counts = c7.load_run(run, "test")
            occ = np.bincount(labels, minlength=len(classes))
            rows = c7.rows_from(c7.confusion(pred, labels), occ, counts, classes)
            pooled += rows
            results[f"{name}/{run.name}"] = direction(rows)
        results[f"{name}/pooled"] = direction(pooled)
        print("done", name, flush=True)
    for key, d in results.items():
        if "beta" not in d:
            continue
        b = d["bayes"]
        rep.append(f"| {key} | {d['pairs']} | {d['T/E']:.2f} [{d['T/E_ci95'][0]:.2f}, {d['T/E_ci95'][1]:.2f}] | "
                   f"{b['T/E']:.2f} | {d['beta']:.2f} [{d['beta_ci95'][0]:.2f}, {d['beta_ci95'][1]:.2f}] | "
                   f"{b['beta']:.2f} ({b['beta_simulated_median']:.2f}) | {b['beta_minus_bayes']:+.2f} "
                   f"[{b['beta_minus_bayes_ci95'][0]:+.2f}, {b['beta_minus_bayes_ci95'][1]:+.2f}] |")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "v4a_hwdb.json").write_text(json.dumps(results, indent=1, default=float))
    (args.out / "report.md").write_text("\n".join(rep) + "\n")
    print("\n".join(rep))


def hwdb_tau(args):
    from c1_reeval_hwdb import TAUS, bootstrap_diff, per_class_acc
    w = np.load(args.weights)
    lambdas = np.round(np.arange(0, 1.5001, 0.125), 3)
    out, rep = {}, ["# V4A-7: optimal effective strength vs test distribution (HWDB, ICDAR 2013)", ""]
    for split_name, root in (("lt", args.lt), ("nat", args.nat)):
        root = Path(root)
        counts = np.load(root / "lt_counts.npy")
        fam = {"train": counts + 1.0, "bctr": w["nat"] + 1.0}
        for run in sorted(p for p in root.iterdir() if (p / "meta.json").exists() and p.name.split("_s")[0] in args.losses):
            meta = json.loads((run / "meta.json").read_text())
            k0 = int(np.argmin(np.abs(TAUS - meta["tau_train"])))  # the model's own (uncorrected) decision
            acc, preds = {}, {}
            for s in ("val", "test"):
                d = np.load(run / f"{s}.npz")
                labels = d["labels"].astype(np.int64)
                acc[s] = per_class_acc(d["grid"], labels, len(counts))
                preds[s] = (d["grid"], labels)
            res = {}
            for fname, base in fam.items():
                for lam in lambdas:
                    q = base ** lam
                    q /= q.sum()
                    cv, ct = 100 * q @ acc["val"], 100 * q @ acc["test"]
                    kv, kt = int(np.argmax(cv)), int(np.argmax(ct))
                    plat = TAUS[ct >= ct.max() - 0.05]
                    grid, labels = preds["test"]
                    point, lo, hi = bootstrap_diff(grid[:, kv], grid[:, k0], labels, q, reps=500)
                    res[f"{fname}^{lam}"] = {"tau_star_test": float(TAUS[kt]), "plateau_test": [float(plat.min()),
                                                                                              float(plat.max())],
                                             "tau_selected_val": float(TAUS[kv]), "test_at_selected": float(ct[kv]),
                                             "test_at_default": float(ct[k0]), "gain_selected_vs_default": point,
                                             "gain_ci95": [lo, hi],
                                             "calibrated_tau": float(1 - lam) if fname == "train" else None}
            out[f"{split_name}/{run.name}"] = {"loss": meta["loss"], "tau_train": meta["tau_train"], "curves": res}
            print("done", split_name, run.name, flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "v4a_hwdb_tau.json").write_text(json.dumps({"lambdas": lambdas.tolist(), "runs": out}, indent=1))
    for split_name in ("lt", "nat"):
        for fname in ("train", "bctr"):
            rep += [f"## {split_name}, q ~ ({fname} + 1)^lambda (mean over runs of each loss)", "",
                    "| loss | lambda | tau* test | plateau | tau selected (val) | gain vs default [CI of first seed] | 1 - lambda |",
                    "|---|---:|---:|---|---:|---|---:|"]
            for loss in args.losses:
                runs = [v for k, v in out.items() if k.startswith(split_name + "/") and v["loss"] == loss]
                if not runs:
                    continue
                for lam in lambdas:
                    e = [r["curves"][f"{fname}^{lam}"] for r in runs]
                    rep.append(f"| {loss} | {lam:g} | {np.mean([x['tau_star_test'] for x in e]):.2f} | "
                               f"[{np.mean([x['plateau_test'][0] for x in e]):.2f}, "
                               f"{np.mean([x['plateau_test'][1] for x in e]):.2f}] | "
                               f"{np.mean([x['tau_selected_val'] for x in e]):.2f} | "
                               f"{np.mean([x['gain_selected_vs_default'] for x in e]):+.2f} "
                               f"[{e[0]['gain_ci95'][0]:+.2f}, {e[0]['gain_ci95'][1]:+.2f}] | "
                               + (f"{1 - lam:.2f} |" if fname == "train" else "- |"))
            rep.append("")
    (args.out / "report.md").write_text("\n".join(rep) + "\n")
    print("\n".join(rep))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("lines")
    q.add_argument("--out", type=Path, required=True)
    q.add_argument("--ctc", nargs="*", default=[])
    q.add_argument("--smtr", nargs="*", default=[])
    q.add_argument("--attn", nargs="*", default=[])
    q.add_argument("--procs", type=int, default=6)
    q = sub.add_parser("hwdb")
    q.add_argument("--out", type=Path, required=True)
    q.add_argument("--data", required=True)
    q.add_argument("--group", nargs="+", required=True)
    q = sub.add_parser("hwdb_tau")
    q.add_argument("--out", type=Path, required=True)
    q.add_argument("--lt", required=True)
    q.add_argument("--nat", required=True)
    q.add_argument("--weights", default="ly/exp/C-01_bal_vs_nat/weights.npz")
    q.add_argument("--losses", nargs="+", default=["ce", "la"])
    args = p.parse_args()
    {"lines": lines, "hwdb": hwdb, "hwdb_tau": hwdb_tau}[args.cmd](args)
