"""A0 diagnostic (val only): for rare-target substitution errors, is the correct character
present in the CTC posterior (segment top-16 union), how highly is it ranked, and is it in the
IDS look-alike neighbourhood of the wrong prediction?  Lines with equal-length B0 output only.
Usage: ly/.venv/bin/python ly/scripts/a0_posterior_ceiling.py
"""
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from a0_prior_correction import CHARS, COUNT_OF, DOMAINS, LOG_PRIOR, RARE, is_han, load  # noqa: E402

index_of = {c: i for i, c in enumerate(CHARS)}
stats = Counter()
for dom in DOMAINS:
    d = load("gtc", "val", dom)
    starts = np.searchsorted(d["cand_seg"], np.arange(len(d["seg_star"]) + 1))
    line_segs = {}
    for s, line in enumerate(d["seg_line"]):
        line_segs.setdefault(int(line), []).append(s)
    for line, (ref, hyp) in enumerate(zip(d["targets"], d["baseline"])):
        segs = line_segs.get(line, [])
        if len(ref) != len(hyp) or len(segs) != len(hyp) or ref == hyp:
            continue
        for pos, (r, h) in enumerate(zip(ref, hyp)):
            if r == h or not is_han(r) or COUNT_OF.get(r, 0) >= RARE or r not in index_of:
                continue
            stats["rare_sub"] += 1
            s = segs[pos]
            a, b = starts[s], starts[s + 1]
            cls, score, rank = d["cand_cls"][a:b], d["cand_score"][a:b], d["cand_rank"][a:b]
            gt = index_of[r]
            hit = np.flatnonzero(cls == gt)
            if len(hit) == 0:
                continue
            stats["in_top16"] += 1
            order = np.argsort(-score)
            pos_gt = int(np.flatnonzero(cls[order] == gt)[0])
            stats["gt_rank2"] += pos_gt == 1
            stats["gt_rank<=5"] += pos_gt <= 4
            stats["in_N50"] += rank[hit[0]] < 50
            stats["in_N10"] += rank[hit[0]] < 10
            gap = score[cls == d["seg_star"][s]][0] - score[hit[0]]
            prior_gap = LOG_PRIOR[d["seg_star"][s]] - LOG_PRIOR[gt]
            stats["fixable_tau<=1"] += gap <= prior_gap  # a tau<=1 prior shift would flip it
n = stats["rare_sub"]
print({k: (v, f"{100*v/n:.1f}%") for k, v in stats.items()})
