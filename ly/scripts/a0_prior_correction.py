"""A0: look-alike-local frequency-prior correction of CTC outputs (protocol: ly/commit.md, A0-00).

Two stages, kept separate on purpose:
  select  -- grid search on the three val splits of the GTC model only; writes selection.json
  test    -- one-shot evaluation of B0 / selected G-LA / selected L-LA on test for GTC and B;
             refuses to run twice.
Usage (repo root):
    ly/.venv/bin/python ly/scripts/a0_prior_correction.py select
    ly/.venv/bin/python ly/scripts/a0_prior_correction.py test
"""
from array import array
import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
from rapidfuzz.distance import Levenshtein

REPO = Path(__file__).resolve().parents[2]
DUMPS = REPO / "ly/exp/A0-01_ctc_dumps/dumps"
LOOKALIKE = REPO / "ly/exp/A0-02_lookalike/lookalike_k50.npz"
FREQ = REPO / "OpenOCR-main/output/rec/bctr_scheme51_ab_bs64/train_character_frequency.json"
OUT = REPO / "ly/exp/A0-03_prior_correction"
DOMAINS = ("scene", "web", "document")
RARE = 100  # a line is "rare" if its reference contains a Han char seen < 100 times in training

# Pre-registered grids (A0-00). gamma = 1.01 means "never gate".
GRID_G = [dict(method="G", tau=t, k=0, gamma=g) for t in (0.1, 0.2, 0.3, 0.5, 0.75, 1.0) for g in (0.9, 0.99, 1.01)]
GRID_L = [dict(method="L", tau=t, k=k, gamma=g) for t in (0.25, 0.5, 0.75, 1.0, 1.5)
          for k in (5, 10, 20, 50) for g in (0.7, 0.9, 0.99, 1.01)]

freq = json.loads(FREQ.read_text(encoding="utf-8"))
CHARS = freq["characters"]
COUNTS = np.asarray(freq["counts"], dtype=np.float64)
LOG_PRIOR = np.log((COUNTS + 1.0) / (COUNTS + 1.0)[1:].sum())
NEIGHBORS = np.load(LOOKALIKE)["neighbors"].astype(np.int64)
COUNT_OF = dict(zip(CHARS, freq["counts"]))


def normalize(text):  # RecMetric settings of the frozen protocol: ignore spaces, lower-case, no filter
    return text.replace(" ", "").lower()


def is_han(c):
    return "一" <= c <= "鿿" or "㐀" <= c <= "䶿"


def neighbor_rank_lookup():
    """Sorted keys star * C + neighbour -> rank, so ranks can be looked up without a dense matrix."""
    C = len(CHARS)
    stars, ranks = np.nonzero(NEIGHBORS >= 0)
    keys = stars * C + NEIGHBORS[stars, ranks]
    order = np.argsort(keys)
    return keys[order], ranks[order].astype(np.int8)


RANK_KEYS, RANK_VALUES = neighbor_rank_lookup()


def load(model, split, domain):
    """Flatten one dump into compact segment / candidate arrays."""
    data = np.load(DUMPS / f"{model}_{split}_{domain}.npz")
    rows = [json.loads(line) for line in open(DUMPS / f"{model}_{split}_{domain}.jsonl", encoding="utf-8")]
    idx, logp, offsets = data["topk_idx"].astype(np.int64), data["topk_logp"].astype(np.float32), data["offsets"]
    K = idx.shape[1]
    seg_line, seg_star, seg_star_score = array("i"), array("i"), array("f")
    cand_seg, cand_cls, cand_score = array("i"), array("i"), array("f")
    for line, (a, b) in enumerate(zip(offsets[:-1], offsets[1:])):
        top1 = idx[a:b, 0]
        t = 0
        while t < len(top1):
            c = top1[t]
            if c == 0:
                t += 1
                continue
            u = t
            while u + 1 < len(top1) and top1[u + 1] == c:
                u += 1
            sums, floor = {}, 0.0
            for f in range(a + t, a + u + 1):
                kth = logp[f, K - 1]
                floor += kth
                for j in range(K):
                    cls = idx[f, j]
                    if cls:
                        sums[cls] = sums.get(cls, 0.0) + logp[f, j] - kth
            n = u - t + 1
            seg = len(seg_line)
            seg_line.append(line)
            seg_star.append(int(c))
            seg_star_score.append((floor + sums[c]) / n)
            for cls, extra in sums.items():
                cand_seg.append(seg)
                cand_cls.append(int(cls))
                cand_score.append((floor + extra) / n)  # mean log p; absent frames use that frame's K-th value
            t = u + 1
    targets = [normalize(r["target"]) for r in rows]
    baseline = [normalize(r["prediction"]) for r in rows]
    rare = np.array([any(is_han(ch) and COUNT_OF.get(ch, 0) < RARE for ch in t) for t in targets])
    seg_star = np.frombuffer(seg_star, dtype=np.int32).astype(np.int64)
    cand_seg = np.frombuffer(cand_seg, dtype=np.int32).astype(np.int64)
    cand_cls = np.frombuffer(cand_cls, dtype=np.int32).astype(np.int64)
    keys = seg_star[cand_seg] * len(CHARS) + cand_cls
    pos = np.minimum(np.searchsorted(RANK_KEYS, keys), len(RANK_KEYS) - 1)
    rank = np.where(RANK_KEYS[pos] == keys, RANK_VALUES[pos], 50).astype(np.int8)  # 50 = not a neighbour
    return {"rows": rows, "targets": targets, "baseline": baseline, "rare": rare,
            "seg_line": np.frombuffer(seg_line, dtype=np.int32), "seg_star": seg_star,
            "seg_star_p": np.exp(np.frombuffer(seg_star_score, dtype=np.float32)),
            "cand_seg": cand_seg, "cand_cls": cand_cls,
            "cand_score": np.frombuffer(cand_score, dtype=np.float32),
            "cand_rank": rank, "cand_is_star": cand_cls == seg_star[cand_seg]}


def decode(d, cfg):
    """Chosen class per segment under a configuration; B0 when cfg is None."""
    if cfg is None:
        return d["seg_star"]
    score = d["cand_score"] - cfg["tau"] * LOG_PRIOR[d["cand_cls"]]
    allowed = np.ones(len(score), bool) if cfg["method"] == "G" else (d["cand_is_star"] | (d["cand_rank"] < cfg["k"]))
    gated = d["seg_star_p"][d["cand_seg"]] >= cfg["gamma"]
    allowed &= ~gated | d["cand_is_star"]
    score = np.where(allowed, score, -np.inf)
    nseg = len(d["seg_star"])
    best = np.full(nseg, -np.inf)
    np.maximum.at(best, d["cand_seg"], score)
    win = np.flatnonzero(score == best[d["cand_seg"]])
    first = np.unique(d["cand_seg"][win], return_index=True)[1]
    chosen = d["seg_star"].copy()
    chosen[d["cand_seg"][win[first]]] = d["cand_cls"][win[first]]
    return chosen


def strings(d, chosen):
    out = [[] for _ in d["rows"]]
    for line, cls in zip(d["seg_line"], chosen):
        out[line].append(CHARS[cls])
    return [normalize("".join(chars)) for chars in out]


def metrics(d, preds):
    correct = np.array([p == t for p, t in zip(preds, d["targets"])])
    edits = sum(Levenshtein.distance(p, t) for p, t in zip(preds, d["targets"]))
    chars = sum(len(t) for t in d["targets"])
    return {"n": len(preds), "correct": int(correct.sum()), "edits": int(edits), "ref_chars": int(chars),
            "rare_n": int(d["rare"].sum()), "rare_correct": int(correct[d["rare"]].sum()),
            "correct_vector": correct}


def pooled(results):
    keys = ("n", "correct", "edits", "ref_chars", "rare_n", "rare_correct")
    tot = {k: sum(r[k] for r in results.values()) for k in keys}
    return {"acc": 100 * tot["correct"] / tot["n"], "cer": 100 * tot["edits"] / tot["ref_chars"],
            "rare_acc": 100 * tot["rare_correct"] / tot["rare_n"],
            "nonrare_acc": 100 * (tot["correct"] - tot["rare_correct"]) / (tot["n"] - tot["rare_n"]), **tot}


def evaluate(data, cfg):
    per = {dom: metrics(d, strings(d, decode(d, cfg))) for dom, d in data.items()}
    return per, pooled(per)



def check_baseline(decoded, baseline, what, max_ties=None):
    """Sanity check: greedy decoding of the dumped top-k equals the model's own greedy output. The dump's top-1 comes
    from torch.topk and the model's output from the post-process argmax; on an exact probability tie the two can pick
    different classes (C-05 crnn_s2: 1 of 127k test lines). Allow a few such ties (<= 1e-4 of the lines, at least 2) and
    report them; anything more is a real mismatch."""
    bad = [i for i, (x, y) in enumerate(zip(decoded, baseline)) if x != y]
    limit = max(2, int(1e-4 * len(baseline))) if max_ties is None else max_ties
    assert len(decoded) == len(baseline) and len(bad) <= limit, (what, len(bad))
    for i in bad:
        print(f"[tie] {what} line {i}: dump top-1 {decoded[i]!r} vs model {baseline[i]!r}", flush=True)
    return bad


def exact_mcnemar(fixed, broken):
    n = fixed + broken
    if n == 0:
        return 1.0
    k = min(fixed, broken)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def label(cfg):
    return "B0" if cfg is None else (f"G-LA(tau={cfg['tau']},gamma={cfg['gamma']})" if cfg["method"] == "G"
                                     else f"L-LA(tau={cfg['tau']},k={cfg['k']},gamma={cfg['gamma']})")


def select():
    data = {dom: load("gtc", "val", dom) for dom in DOMAINS}
    for dom, d in data.items():  # B0 must reproduce the model's own greedy decode
        assert strings(d, decode(d, None)) == d["baseline"], dom
    _, base = evaluate(data, None)
    grid = []
    for cfg in GRID_G + GRID_L:
        _, pool = evaluate(data, cfg)
        grid.append({**cfg, **{k: v for k, v in pool.items()}})
        print(label(cfg), f"acc={pool['acc']:.3f} rare={pool['rare_acc']:.3f}", flush=True)
    chosen = {}
    for method in ("G", "L"):
        ok = [g for g in grid if g["method"] == method and g["acc"] >= base["acc"] - 0.05]
        best = sorted(ok, key=lambda g: (-g["rare_correct"], g["tau"], g["k"], g["gamma"]))[0] if ok else None
        chosen[method] = None if best is None else {k: best[k] for k in ("method", "tau", "k", "gamma")}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "val_grid.json").write_text(json.dumps({"baseline": base, "grid": grid}, indent=1) + "\n")
    (OUT / "selection.json").write_text(json.dumps({"rule": "max pooled val rare-line acc s.t. pooled acc >= B0 - 0.05pp; "
                                                    "ties: smaller tau, k, gamma", "model": "gtc", "chosen": chosen},
                                                   indent=2) + "\n")
    print("B0 val:", {k: round(v, 3) for k, v in base.items() if isinstance(v, float)})
    print("selected:", chosen)


def test():
    target = OUT / "test_results.json"
    if target.exists():
        sys.exit("test already evaluated once; see " + str(target))
    chosen = json.loads((OUT / "selection.json").read_text())["chosen"]
    configs = [None, chosen["G"], chosen["L"]]
    report, lines = {}, ["| 模型 | 方法 | 合并 Acc | 合并 CER | 生僻行 Acc | 非生僻行 Acc | Scene | Web | Document |",
                         "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for model in ("gtc", "b"):
        data = {dom: load(model, "test", dom) for dom in DOMAINS}
        for dom, d in data.items():
            assert strings(d, decode(d, None)) == d["baseline"], (model, dom)
        base_vec = None
        for cfg in configs:
            per, pool = evaluate(data, cfg)
            vec = np.concatenate([per[dom]["correct_vector"] for dom in DOMAINS])
            rare = np.concatenate([data[dom]["rare"] for dom in DOMAINS])
            entry = {k: v for k, v in pool.items()}
            entry["domains"] = {dom: 100 * per[dom]["correct"] / per[dom]["n"] for dom in DOMAINS}
            if cfg is None:
                base_vec = vec
            else:
                fixed = int((vec & ~base_vec & rare).sum())
                broken = int((~vec & base_vec & rare).sum())
                entry["rare_fixed"], entry["rare_broken"] = fixed, broken
                entry["rare_mcnemar_p"] = exact_mcnemar(fixed, broken)
                entry["all_fixed"] = int((vec & ~base_vec).sum())
                entry["all_broken"] = int((~vec & base_vec).sum())
            report[f"{model}:{label(cfg)}"] = entry
            lines.append(f"| {model.upper()} | {label(cfg)} | {entry['acc']:.3f} | {entry['cer']:.3f} | "
                         f"{entry['rare_acc']:.3f} | {entry['nonrare_acc']:.3f} | "
                         + " | ".join(f"{entry['domains'][dom]:.3f}" for dom in DOMAINS) + " |")
    target.write_text(json.dumps(report, indent=1) + "\n")
    (OUT / "test_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    for key, entry in report.items():
        if "rare_fixed" in entry:
            print(key, "rare fixed/broken", entry["rare_fixed"], entry["rare_broken"],
                  "p=%.2g" % entry["rare_mcnemar_p"], "all fixed/broken", entry["all_fixed"], entry["all_broken"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("select", "test"))
    {"select": select, "test": test}[parser.parse_args().stage]()
