"""C-13: frequency-flip intervention on BCTR text lines -- pair selection, line weights, sampler self-test, analysis.

  pairs    choose confusable pairs from the C-05 CRNN *val* predictions (both seeds, official normalisation): both
           characters Han, unchanged by the normalisation, 300 <= n_low < n_high <= 5000 training occurrences,
           2 <= n_high / n_low <= 8, >= 3 val confusions; greedy by val confusions per training occurrence (the most
           confusions for the occurrence budget; amended 2026-10-16 21:47 before any training, see docs/experiments.md), no
           character in two pairs, at most 200 pairs and 150,000 training occurrences in total.
  weights  per-line sampling weights for the arms rev (exposures of the two members swapped) and bal (both at the
           geometric mean), by iterative proportional fitting of multiplicative per-character factors on the lines
           that contain a target character (line weight capped to [1/16, 16]), with the per-bucket renormalisation the
           sampler applies. Writes weights_<arm>.npz (one array per training LMDB) and exposure_<arm>.json (expected
           per-epoch exposure of every character).
  selftest CPU checks of the sampler (unit weights == Scheme51RatioSampler; resampled multiplicities) and the fitting.
  analyze  direction of the confusions of the selected pairs on test (and val, selection-biased, descriptive) for the
           natural arm (C-05 crnn_s1/s2) and the C-13 arms: D = errors toward the originally more frequent member /
           expectation under symmetric per-occurrence confusion; pair-cluster bootstrap; criteria F1-F3.

    python src/data/line_swap.py pairs --dumps results/C-05_crnn/dumps --out results/C-13_crnn_flip
    python src/data/line_swap.py weights --out results/C-13_crnn_flip --dataset-root /root/autodl-tmp/bctr
    python src/data/line_swap.py selftest
    python src/data/line_swap.py analyze --nat-dumps results/C-05_crnn/dumps --dumps results/C-13_crnn_flip/dumps \
        --out results/C-13_crnn_flip
Protocol: docs/experiments.md, entry "C-13 至 C-17 预注册".
"""
import argparse
import collections
import io
import json
import math
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import prior_correction as a0  # noqa: E402

TRAIN_DOMAINS = ("document", "scene", "web")  # order of the training data_dir_list
MAX_LEN = 25  # CTCLabelEncode max_text_length of the C-05 config
N_MIN, LOW, HIGH, RATIO, MAX_PAIRS, BUDGET = 3, 300, 5000, (2.0, 8.0), 200, 150_000
W_CAP = 16.0
ARMS = ("rev", "bal")
NAT_TAGS = ("crnn_s1", "crnn_s2")
ARM_TAGS = {"rev": ("rev_s1", "rev_s2"), "bal": ("bal_s1", "bal_s2")}


def official():
    from analysis.c04_external_analysis import official_normalizer
    return official_normalizer()


# ----------------------------------------------------------------------------- pairs
def pairs(args):
    from analysis import c03_confusion_analysis as c3
    norm = official()
    charset = set(a0.CHARS[1:])

    def eligible(c):
        return a0.is_han(c) and c in charset and norm(c) == c

    n = collections.Counter()
    for tag in args.tags:
        for dom in a0.DOMAINS:
            for line in open(args.dumps / f"{tag}_val_{dom}.jsonl", encoding="utf-8"):
                r = json.loads(line)
                for a, b in c3.substitutions(norm(r["target"]), norm(r["prediction"])):
                    if eligible(a) and eligible(b):
                        n[(a, b)] += 1
    cand = []
    for a, b in {tuple(sorted(k)) for k in n}:
        ca, cb = a0.COUNT_OF.get(a, 0), a0.COUNT_OF.get(b, 0)
        N = n[(a, b)] + n[(b, a)]
        if ca == cb:
            continue
        h, l = (a, b) if ca > cb else (b, a)
        nh, nl = max(ca, cb), min(ca, cb)
        if N >= N_MIN and nl >= LOW and nh <= HIGH and RATIO[0] <= nh / nl <= RATIO[1]:
            cand.append({"h": h, "l": l, "n_h": int(nh), "n_l": int(nl), "val_errors": N, "val_to_h": n[(l, h)],
                         "val_to_l": n[(h, l)], "lookalike": bool(c3.lookalike(h, l))})
    cand.sort(key=lambda r: (-r["val_errors"] / (r["n_h"] + r["n_l"]), -r["val_errors"], r["h"], r["l"]))
    used, chosen, total = set(), [], 0
    for r in cand:
        if len(chosen) == MAX_PAIRS:
            break
        if r["h"] in used or r["l"] in used or total + r["n_h"] + r["n_l"] > BUDGET:
            continue
        chosen.append(r)
        used |= {r["h"], r["l"]}
        total += r["n_h"] + r["n_l"]
    summary = {"candidates": len(cand), "pairs": len(chosen), "train_occurrences": total,
               "val_errors": sum(r["val_errors"] for r in chosen),
               "val_share_to_high": sum(r["val_to_h"] for r in chosen) / max(1, sum(r["val_errors"] for r in chosen)),
               "lookalike_share": float(np.mean([r["lookalike"] for r in chosen])) if chosen else None,
               "median_ratio": float(np.median([r["n_h"] / r["n_l"] for r in chosen])) if chosen else None}
    rules = {"source": [f"{t} val" for t in args.tags], "normalisation": "official", "N_min": N_MIN,
             "n_low_min": LOW, "n_high_max": HIGH, "ratio": RATIO, "max_pairs": MAX_PAIRS, "occurrence_budget": BUDGET,
             "order": "val confusions / (n_high + n_low), descending"}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "pairs.json").write_text(json.dumps({"rules": rules, "summary": summary, "pairs": chosen},
                                                    indent=1, ensure_ascii=False))
    print(json.dumps(summary, ensure_ascii=False))


# ----------------------------------------------------------------------------- weights
def read_train(root):
    """Per training LMDB: CTC-encodable label (dictionary characters only, 1..25 of them, else None) and the
    aspect-ratio bucket of the training dataset (round(w / h) clipped to [1, 12]); index 0 unused."""
    import lmdb
    from PIL import Image
    charset = set(a0.CHARS[1:])
    out = {}
    for dom in TRAIN_DOMAINS:
        name = f"{dom}_train"
        env = lmdb.open(str(root / dom / name), readonly=True, lock=False, readahead=False, meminit=False)
        with env.begin() as txn:
            n = int(txn.get(b"num-samples"))
            labels, ratio, no_wh = [None] * (n + 1), np.ones(n + 1), 0
            for i in range(1, n + 1):
                raw = txn.get(b"label-%09d" % i)
                kept = "".join(c for c in ("" if raw is None else raw.decode("utf-8")) if c in charset)
                labels[i] = kept if 0 < len(kept) <= MAX_LEN else None
                wh = txn.get(b"wh-%09d" % i)
                if wh is None:
                    no_wh += 1
                    w, h = Image.open(io.BytesIO(txn.get(b"image-%09d" % i))).size
                else:
                    w, h = map(float, wh.decode("utf-8").split("_"))
                ratio[i] = float(w) / float(h)
                if i % 200000 == 0:
                    print(f"{name}: {i}/{n}", flush=True)
        env.close()
        bucket = np.clip(np.around(ratio), 1, 12).astype(np.int64)
        bucket[0] = 0
        out[name] = {"labels": labels, "bucket": bucket, "no_wh_key": no_wh,
                     "rejected": sum(lab is None for lab in labels[1:])}
    return out


def fit_weights(train, targets, iters=1000, tol=1e-3):
    """Multiplicative per-target-character factors r_c; line weight = clip(prod over the distinct target characters of
    the line of r_c, 1/W_CAP, W_CAP); the sampler rescales each bucket to its size. Fits expected exposures to targets."""
    tc = sorted(targets)
    tid = {c: j for j, c in enumerate(tc)}
    T = np.array([targets[c] for c in tc], float)
    rows = []  # (lmdb name, sample number, bucket, target ids, target counts)
    n_b = collections.Counter()
    for name, d in train.items():
        for i in range(1, len(d["labels"])):
            n_b[int(d["bucket"][i])] += 1
            lab = d["labels"][i]
            if lab is None:
                continue
            cnt = collections.Counter(c for c in lab if c in tid)
            if cnt:
                rows.append((name, i, int(d["bucket"][i]), [tid[c] for c in cnt], list(cnt.values())))
    L = len(rows)
    li = np.repeat(np.arange(L), [len(r[3]) for r in rows])
    cj = np.concatenate([r[3] for r in rows]).astype(np.int64)
    cv = np.concatenate([r[4] for r in rows]).astype(float)
    lb = np.array([r[2] for r in rows])
    buckets = sorted(n_b)
    bidx = {b: k for k, b in enumerate(buckets)}
    lbk = np.array([bidx[b] for b in lb])
    nb = np.array([n_b[b] for b in buckets], float)
    mb = np.bincount(lbk, minlength=len(buckets)).astype(float)
    log_r = np.zeros(len(tc))
    hist = []
    for it in range(iters):
        w = np.clip(np.exp(np.bincount(li, weights=log_r[cj], minlength=L)), 1 / W_CAP, W_CAP)
        f = nb / (nb - mb + np.bincount(lbk, weights=w, minlength=len(buckets)))
        E = np.bincount(cj, weights=(f[lbk] * w)[li] * cv, minlength=len(tc))
        err = np.log(E / T)
        hist.append(float(np.abs(err).max()))
        if hist[-1] < tol:
            break
        # bounded, so that a factor stuck at the cap cannot drift to +-inf (inf - inf on a line holding two targets)
        log_r = np.clip(log_r - err, -3 * math.log(W_CAP), 3 * math.log(W_CAP))
    return {"rows": rows, "w": w, "f": dict(zip(buckets, f.tolist())), "E": dict(zip(tc, E.tolist())),
            "iterations": it + 1, "max_abs_log_error": hist[-1]}


def all_exposure(train, fit):
    """Expected per-epoch exposure of every character: sum over buckets of f_b * (base counts + (w - 1) * counts of
    the reweighted lines)."""
    f = fit["f"]
    E = collections.Counter()
    for d in train.values():
        for i in range(1, len(d["labels"])):
            lab = d["labels"][i]
            if lab is not None:
                fb = f[int(d["bucket"][i])]
                for c in lab:
                    E[c] += fb
    for (name, i, b, _, _), w in zip(fit["rows"], fit["w"]):
        for c in train[name]["labels"][i]:
            E[c] += f[b] * (w - 1.0)
    return E


def weights(args):
    pairs_ = json.loads((args.out / "pairs.json").read_text(encoding="utf-8"))["pairs"]
    train = read_train(Path(args.dataset_root))
    own = collections.Counter(c for d in train.values() for lab in d["labels"][1:] if lab for c in lab)
    report = {"lmdbs": {k: {"lines": len(d["labels"]) - 1, "rejected_by_label_filter": d["rejected"],
                            "no_wh_key": d["no_wh_key"]} for k, d in train.items()},
              "counts_corr_with_frequency_file": float(np.corrcoef(
                  np.log1p([own[c] for c in a0.CHARS[1:]]), np.log1p([a0.COUNT_OF[c] for c in a0.CHARS[1:]]))[0, 1])}
    for arm in ARMS:
        targets = {}
        for p in pairs_:
            nh, nl = own[p["h"]], own[p["l"]]
            targets[p["h"]], targets[p["l"]] = (nl, nh) if arm == "rev" else (math.sqrt(nh * nl),) * 2
        fit = fit_weights(train, targets)
        arrays = {name: np.ones(len(d["labels"])) for name, d in train.items()}
        for (name, i, _, _, _), w in zip(fit["rows"], fit["w"]):
            arrays[name][i] = w
        np.savez(args.out / f"weights_{arm}.npz", **arrays)
        E = all_exposure(train, fit)
        err = {c: math.log(fit["E"][c] / targets[c]) for c in targets}
        coll = [abs(math.log(E[c] / own[c])) for c in own if own[c] >= 100 and c not in targets and a0.is_han(c)]
        within = [abs(err[p["h"]]) <= math.log(1.25) and abs(err[p["l"]]) <= math.log(1.25) for p in pairs_]
        report[arm] = {"iterations": fit["iterations"], "max_abs_log_error": fit["max_abs_log_error"],
                       "target_chars_within_10pct": float(np.mean([abs(v) <= math.log(1.1) for v in err.values()])),
                       "pairs_within_25pct": int(sum(within)), "reweighted_lines": len(fit["rows"]),
                       "weight_min": float(fit["w"].min()), "weight_max": float(fit["w"].max()),
                       "bucket_factors": fit["f"],
                       "collateral_abs_log_change_median": float(np.median(coll)),
                       "collateral_abs_log_change_p95": float(np.percentile(coll, 95))}
        (args.out / f"exposure_{arm}.json").write_text(json.dumps(
            {"targets": targets, "achieved": fit["E"], "own_counts": {c: own[c] for c in targets},
             "pairs_within_25pct": [p["h"] + p["l"] for p, ok in zip(pairs_, within) if ok],
             "exposure": {c: E[c] for c in E}}, ensure_ascii=False))
        print(arm, json.dumps(report[arm]), flush=True)
    ok = all(report[a]["target_chars_within_10pct"] >= 0.9 for a in ARMS)
    report["check_90pct_within_10pct"] = ok
    (args.out / "weights_report.json").write_text(json.dumps(report, indent=1))
    print("WEIGHTS_OK" if ok else "WEIGHTS_CHECK_FAILED")


# ----------------------------------------------------------------------------- selftest
class FakeDataset:
    """The attributes RatioSampler / line_weights read from RatioDataSetTVResize."""

    def __init__(self, n, seed):
        rng = np.random.default_rng(seed)
        self.ds_width, self.seed = True, 1
        self.wh_ratio = np.clip(np.around(rng.gamma(2.0, 2.0, n)), 1, 12)
        self.wh_ratio_sort = np.argsort(self.wh_ratio)
        self.data_idx_order_list = np.column_stack([np.zeros(n), rng.permutation(n) + 1]).astype(np.float64)
        self.lmdb_sets = {0: {"dirpath": "/tmp/x/document_train/", "num_samples": n}}

    def __len__(self):
        return len(self.wh_ratio)


def selftest(args):
    import random
    import tempfile
    sys.path.insert(0, str(a0.REPO / "OpenOCR-main"))
    from openrec.scheme51 import Scheme51RatioSampler
    from training.line_swap_sampler import C13WeightedRatioSampler
    kw = dict(scales=[[128, 32]], first_bs=64, fix_bs=False, divided_factor=[4, 16], is_training=True, max_ratio=12,
              max_bs=256)
    tmp = Path(tempfile.mkdtemp())
    ds = FakeDataset(20000, 0)
    n = len(ds)
    np.savez(tmp / "unit.npz", document_train=np.ones(n + 1))
    rng = np.random.default_rng(1)
    w = np.where(rng.random(n + 1) < 0.1, rng.choice([1 / 8, 0.5, 2.0, 8.0], n + 1), 1.0)
    np.savez(tmp / "rand.npz", document_train=w)
    ref = Scheme51RatioSampler(ds, **kw)
    unit = C13WeightedRatioSampler(ds, weights_path=str(tmp / "unit.npz"), **kw)
    for _ in range(2):  # two epochs, identical batches
        assert list(iter(ref)) == list(iter(unit)), "unit weights must reproduce Scheme51RatioSampler"
    rs = C13WeightedRatioSampler(ds, weights_path=str(tmp / "rand.npz"), **kw)
    assert len(rs) == len(ref)
    random.seed(5)
    rs.batch_list, ref.batch_list = rs.create_batch(), ref.create_batch()
    assert [len(b) for b in rs.batch_list] == [len(b) for b in ref.batch_list], "batch sizes must not change"
    # multiplicities: per bucket, line i appears floor or ceil of n_b * w_i / sum_b w (ignoring the wrap-around fill)
    order = ds.data_idx_order_list[:, 1].astype(int)
    lw = w[order]
    seen = collections.Counter()
    for ratio in np.unique(ds.wh_ratio):
        idx = np.flatnonzero(ds.wh_ratio == ratio)
        if np.all(lw[idx] == 1):
            continue
        random.seed(11)
        got = collections.Counter(__import__("training.line_swap_sampler", fromlist=["resample"]).resample(idx, lw[idx], 3).tolist())
        exp = len(idx) * lw[idx] / lw[idx].sum()
        for i, e in zip(idx, exp):
            assert math.floor(e - 1e-9) <= got[i] <= math.ceil(e + 1e-9), (i, e, got[i])
        seen[ratio] = len(idx)
    assert seen, "no reweighted bucket tested"
    # fitting: a tiny corpus with two pairs, targets swapped
    labels = [None] + ["ab", "a", "a", "a", "c", "cd", "d", "bx", "y", "y", "dd", "x"] * 50
    bucket = np.array([0] + [1, 2] * (6 * 50))
    train = {"document_train": {"labels": labels, "bucket": bucket}}
    own = collections.Counter(c for lab in labels[1:] for c in lab)
    targets = {"a": own["b"], "b": own["a"], "c": own["d"], "d": own["c"]}
    fit = fit_weights(train, targets)
    assert fit["max_abs_log_error"] < 1e-3, fit["max_abs_log_error"]
    E = all_exposure(train, fit)
    assert all(abs(E[c] - fit["E"][c]) < 1e-6 * max(1, E[c]) for c in targets)
    assert abs(sum(E.values()) - sum(own.values())) / sum(own.values()) < 0.5
    print("SELFTEST OK", {"buckets_checked": len(seen), "fit_iterations": fit["iterations"]})


# ----------------------------------------------------------------------------- analyze
def load_model(dumps, tag, split, norm):
    refs, hyps, raw = [], [], []
    for dom in a0.DOMAINS:
        for line in open(dumps / f"{tag}_{split}_{dom}.jsonl", encoding="utf-8"):
            r = json.loads(line)
            refs.append(norm(r["target"]))
            hyps.append(norm(r["prediction"]))
            raw.append(r["target"])
    return refs, hyps, raw


def pair_counts(refs, hyps, pairs_):
    from analysis import c03_confusion_analysis as c3
    members = {c for p in pairs_ for c in (p["h"], p["l"])}
    n = collections.Counter()
    for ref, hyp in zip(refs, hyps):
        for a, b in c3.substitutions(ref, hyp):
            if a in members and b in members:
                n[(a, b)] += 1
    k = np.array([n[(p["l"], p["h"])] for p in pairs_], float)  # toward the originally more frequent member
    N = k + np.array([n[(p["h"], p["l"])] for p in pairs_], float)
    return k, N


def recall(refs, hyps, chars):
    from rapidfuzz.distance import Levenshtein
    tot, hit = collections.Counter(), collections.Counter()
    for ref, hyp in zip(refs, hyps):
        wrong = {s for tag, s, _ in Levenshtein.editops(ref, hyp) if tag in ("replace", "delete")}
        for i, ch in enumerate(ref):
            if ch in chars:
                tot[ch] += 1
                hit[ch] += i not in wrong
    return tot, hit


def direction(K, N, q, idx=None):
    """Errors toward the originally more frequent member / their expectation under symmetric confusion (floored at
    1e-6 so that logs stay finite in sparse bootstrap draws)."""
    if idx is not None:
        K, N, q = K[idx], N[idx], q[idx]
    return max(1e-6, K.sum()) / max(1e-6, (N * q).sum())


def analyze(args):
    from analysis import c03_confusion_asymmetry as c3b
    norm = official()
    pairs_ = json.loads((args.out / "pairs.json").read_text(encoding="utf-8"))["pairs"]
    expo = {arm: json.loads((args.out / f"exposure_{arm}.json").read_text(encoding="utf-8")) for arm in ARMS}
    keep = [all(p["h"] + p["l"] in expo[a]["pairs_within_25pct"] for a in ARMS) for p in pairs_]
    pairs_ = [p for p, ok in zip(pairs_, keep) if ok]
    own = expo["rev"]["own_counts"]
    xs = {"nat": np.array([math.log((own[p["h"]] + 1) / (own[p["l"]] + 1)) for p in pairs_])}
    for arm in ARMS:
        e = expo[arm]["exposure"]
        xs[arm] = np.array([math.log((e[p["h"]] + 1) / (e[p["l"]] + 1)) for p in pairs_])
    models = {"nat": [(args.nat_dumps, t) for t in NAT_TAGS],
              **{arm: [(args.dumps, t) for t in ARM_TAGS[arm]] for arm in ARMS}}
    members = {c for p in pairs_ for c in (p["h"], p["l"])}
    res = {"pairs_analysed": len(pairs_), "pairs_dropped_exposure": int(len(keep) - sum(keep)),
           "x_mean": {k: float(v.mean()) for k, v in xs.items()}}
    rng = np.random.default_rng(0)
    for split in ("test", "val"):
        occ = None
        per, pooled = {}, {}
        for arm, runs in models.items():
            K, N = np.zeros(len(pairs_)), np.zeros(len(pairs_))
            for dumps, tag in runs:
                refs, hyps, raw = load_model(dumps, tag, split, norm)
                o = collections.Counter(ch for r in refs for ch in r if ch in members)
                occ = o if occ is None else occ
                assert o == occ, "all models must share the same references"
                k, n = pair_counts(refs, hyps, pairs_)
                K, N = K + k, N + n
                correct = np.array([h == r for h, r in zip(hyps, refs)])
                rare = np.array([any(a0.is_han(c) and a0.COUNT_OF.get(c, 0) < a0.RARE for c in t.replace(" ", "").lower())
                                 for t in raw])
                target_line = np.array([any(c in members for c in r) for r in refs])
                tot, hit = recall(refs, hyps, members)
                hs, ls_ = [p["h"] for p in pairs_], [p["l"] for p in pairs_]
                per[tag] = {"arm": arm, "acc": 100 * correct.mean(), "rare_acc": 100 * correct[rare].mean(),
                            "target_line_acc": 100 * correct[target_line].mean(), "target_lines": int(target_line.sum()),
                            "recall_high": 100 * sum(hit[c] for c in hs) / max(1, sum(tot[c] for c in hs)),
                            "recall_low": 100 * sum(hit[c] for c in ls_) / max(1, sum(tot[c] for c in ls_)),
                            "pair_errors": float(n.sum()), "to_high": float(k.sum())}
            q = np.array([occ[p["l"]] / max(1, occ[p["l"]] + occ[p["h"]]) for p in pairs_])
            pooled[arm] = {"K": K, "N": N}
        q = np.array([occ[p["l"]] / max(1, occ[p["l"]] + occ[p["h"]]) for p in pairs_])
        D = {arm: direction(v["K"], v["N"], q) for arm, v in pooled.items()}
        boots = collections.defaultdict(list)
        for _ in range(args.reps):
            i = rng.integers(0, len(pairs_), len(pairs_))
            d = {arm: direction(v["K"], v["N"], q, i) for arm, v in pooled.items()}
            for arm in d:
                boots[arm].append(math.log(d[arm]))
            boots["nat-rev"].append(math.log(d["nat"]) - math.log(d["rev"]))
            boots["nat-bal"].append(math.log(d["nat"]) - math.log(d["bal"]))
        ci = {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in boots.items()}
        # joint conditional-binomial fit with each arm's own expected exposure (C-03b model), bootstrap over pairs
        k_all = np.concatenate([pooled[a]["K"] for a in pooled])
        n_all = np.concatenate([pooled[a]["N"] for a in pooled])
        q_all = np.clip(np.concatenate([q] * len(pooled)), 1e-6, 1 - 1e-6)
        x_all = np.concatenate([xs[a] for a in pooled])
        pid = np.concatenate([np.arange(len(pairs_))] * len(pooled))
        m = n_all > 0
        beta = c3b.fit_beta(k_all[m], n_all[m], q_all[m], x_all[m])
        bb = []
        for _ in range(args.reps):
            pick = rng.integers(0, len(pairs_), len(pairs_))
            sel = np.concatenate([np.flatnonzero(pid == j) for j in pick])
            sel = sel[n_all[sel] > 0]
            bb.append(c3b.fit_beta(k_all[sel], n_all[sel], q_all[sel], x_all[sel]))
        # secondary: C-07-style share on pairs with >= 3 errors under both nat and rev
        both = (pooled["nat"]["N"] >= 3) & (pooled["rev"]["N"] >= 3)
        res[split] = {"per_model": per, "occurrences_total": int(sum(occ.values())),
                      "pooled": {a: {"errors": float(v["N"].sum()), "to_high": float(v["K"].sum()),
                                     "expected_to_high": float((v["N"] * q).sum()), "D": float(D[a]),
                                     "log_D_ci95": ci[a]} for a, v in pooled.items()},
                      "log_D_nat_minus_rev_ci95": ci["nat-rev"], "log_D_nat_minus_bal_ci95": ci["nat-bal"],
                      "beta_own_exposure": {"estimate": beta, "ci95": [float(np.percentile(bb, 2.5)),
                                                                       float(np.percentile(bb, 97.5))]},
                      "secondary_both_ge3": {"pairs": int(both.sum()),
                                             **{a: float(pooled[a]["K"][both].sum() / max(1, pooled[a]["N"][both].sum()))
                                                for a in pooled}}}
        print(split, json.dumps({a: round(D[a], 3) for a in D}), "beta", round(beta, 3), flush=True)
    t = res["test"]
    Dn, Db, Dr = (t["pooled"][a]["D"] for a in ("nat", "bal", "rev"))
    res["criteria"] = {
        "F1_flip": bool(Dn > 1 and Dr < 1 and t["log_D_nat_minus_rev_ci95"][0] > 0),
        "F2_dose": bool(Dn > Db > Dr and abs(math.log(Db)) <= 0.5 * abs(math.log(Dn))),
        "F3_beta_own_exposure": bool(t["beta_own_exposure"]["ci95"][0] > 0)}
    (args.out / "c13_results.json").write_text(json.dumps(res, indent=1, ensure_ascii=False, default=float))
    lines = ["# C-13 frequency flip on BCTR text lines (official normalisation)", "",
             f"Pairs analysed: {res['pairs_analysed']} (dropped for exposure misfit: {res['pairs_dropped_exposure']})", ""]
    for split in ("test", "val"):
        s = res[split]
        lines += [f"## {split}" + (" (pairs were selected on val: descriptive only)" if split == "val" else ""), "",
                  "| arm | errors | toward high | expected | D [95% CI] |", "|---|---:|---:|---:|---|"]
        for a, v in s["pooled"].items():
            lo, hi = (math.exp(x) for x in v["log_D_ci95"])
            lines.append(f"| {a} | {v['errors']:.0f} | {v['to_high']:.0f} | {v['expected_to_high']:.1f} | "
                         f"{v['D']:.2f} [{lo:.2f}, {hi:.2f}] |")
        b = s["beta_own_exposure"]
        lines += ["", f"- beta (own exposure, joint over arms): {b['estimate']:.3f} [{b['ci95'][0]:.3f}, {b['ci95'][1]:.3f}]",
                  f"- secondary (pairs with >= 3 errors under nat and rev): {s['secondary_both_ge3']}", "",
                  "| model | arm | acc | rare-line acc | target-line acc | recall high / low |", "|---|---|---:|---:|---:|---|"]
        for tag, v in s["per_model"].items():
            lines.append(f"| {tag} | {v['arm']} | {v['acc']:.2f} | {v['rare_acc']:.2f} | {v['target_line_acc']:.2f} | "
                         f"{v['recall_high']:.1f} / {v['recall_low']:.1f} |")
        lines.append("")
    lines.append(f"**Criteria (test): {res['criteria']}**")
    (args.out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("CRITERIA", json.dumps(res["criteria"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pairs")
    p.add_argument("--dumps", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--tags", nargs="+", default=list(NAT_TAGS))
    p = sub.add_parser("weights")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--dataset-root", dest="dataset_root", default="/root/autodl-tmp/bctr")
    sub.add_parser("selftest")
    p = sub.add_parser("analyze")
    p.add_argument("--nat-dumps", dest="nat_dumps", type=Path, required=True)
    p.add_argument("--dumps", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--reps", type=int, default=1000)
    args = parser.parse_args()
    {"pairs": pairs, "weights": weights, "selftest": selftest, "analyze": analyze}[args.cmd](args)
