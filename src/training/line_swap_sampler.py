"""C-13: Scheme51RatioSampler with per-line exposure weights (frequency-flip intervention on BCTR text lines).

Every epoch each aspect-ratio bucket keeps its size, but its members are drawn by systematic resampling with the line
weights: line i appears floor or ceil of n_bucket * w_i / sum_bucket(w) times. Number of batches, batch sizes and the
aspect-ratio mix are therefore those of Scheme51RatioSampler. A bucket whose weights are all 1 takes exactly the
parent's code path (same calls to `random`), so unit weights reproduce Scheme51RatioSampler batch for batch.
Weights come from `line_swap.py weights`: one array per training LMDB, indexed by the LMDB sample number.
OpenOCR's Trainer builds the sampler from the config, so train_crnn_line_swap.py registers this module under
tools.data.SAMPLER_MODULES. Protocol: docs/experiments.md, entry "C-13 至 C-17 预注册".
"""
import os
import random

import numpy as np

from openrec.scheme51 import Scheme51RatioSampler


def line_weights(data_source, weights_path):
    """Weight of every dataset index, looked up through the dataset's (lmdb, sample number) order list."""
    saved = np.load(weights_path)
    out = np.ones(len(data_source), np.float64)
    order = np.asarray(data_source.data_idx_order_list)
    for lmdb_idx, source in data_source.lmdb_sets.items():
        name = os.path.basename(os.path.normpath(source["dirpath"]))
        w = saved[name]
        if len(w) != source["num_samples"] + 1:  # index 0 unused, LMDB sample numbers start at 1
            raise ValueError(f"{weights_path}: {name} has {len(w) - 1} weights, LMDB has {source['num_samples']}")
        sel = order[:, 0] == lmdb_idx
        out[sel] = w[order[sel, 1].astype(np.int64)]
    if not (np.isfinite(out).all() and (out >= 0).all()):
        raise ValueError("weights must be finite and non-negative")
    return out


def resample(indices, w, seed):
    """Systematic resampling of len(indices) draws with probabilities ~ w (random order, random offset)."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(indices))
    cw = np.cumsum(w[perm])
    cw = cw / cw[-1]
    u = (np.arange(len(indices)) + rng.random()) / len(indices)
    pick = np.minimum(np.searchsorted(cw, u, side="right"), len(indices) - 1)
    return indices[perm[pick]]


class C13WeightedRatioSampler(Scheme51RatioSampler):
    def __init__(self, data_source, weights_path, **kwargs):
        self.weights_path = weights_path
        self.line_w = line_weights(data_source, weights_path)  # needed by create_batch() inside the parent __init__
        super().__init__(data_source, **kwargs)

    def create_batch(self):
        """Scheme51RatioSampler.create_batch with the bucket members resampled by weight."""
        batches = []
        for ratio in self.indices_rank_i_ratio_unique:
            indices = self.indices_rank_i_ori[np.where(self.indices_rank_i_ratio == ratio)[0]].copy()
            w = self.line_w[indices]
            if self.is_training and not np.all(w == 1.0):
                indices = resample(indices, w, random.getrandbits(64))
            if self.shuffle:
                random.shuffle(indices)
            size = self.base_batch_size if ratio < 5 else min(
                self.max_bs, int(max(1, self.base_elements / (self.base_im_h * ratio * self.base_im_h))))
            for start in range(0, len(indices), size):
                chunk = indices[start:start + size]
                if self.is_training and len(indices) > size and len(chunk) < size:
                    chunk = np.concatenate((indices[:size - len(chunk)], chunk))
                entries = np.stack((np.full_like(chunk, ratio * self.base_im_h),
                                    np.full_like(chunk, self.base_im_h), chunk,
                                    np.full_like(chunk, ratio)), axis=1)
                batches.append(entries.tolist())
        return batches
