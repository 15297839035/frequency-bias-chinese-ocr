"""A1: long-tailed handwritten character recognition with structure-aware soft supervision.

Losses (--loss):
  ce        plain cross-entropy
  ls        uniform label smoothing, eps
  csls      eps spread uniformly over the top-k neighbours from a *data-driven* similarity
            (cosine of class-mean features of a trained CE model; --neighbors from `prototypes`)
  sals      eps spread uniformly over the top-k *IDS structural* neighbours (data-free)
  sals_f    sals with frequency-coupled eps: head classes give away more mass (eps_min..eps_max)
  gls       eps over the top-k *rendered-glyph appearance* neighbours (data-free; --glyph-neighbors)
  ugls      union: IDS top-k/2 + glyph top-k/2, de-duplicated, filled up in glyph order (main arm)
  ugls_f    ugls with frequency-coupled eps
  la        logit-adjusted cross-entropy (Menon et al., tau=1)
  la_sals_f la + sals_f
  la_ls, la_sals, la_ugls, la_csls   la + the named smoothing (A1-03, exploratory)
  C-10 long-tail baselines (docs/experiments.md, "C-07 至 C-11 预注册"; earlier losses unchanged):
  cb        class-balanced re-weighting (Cui et al. 2019), effective number with --cb-beta, sum_c w_c = C
  focal     focal loss (Lin et al. 2017), --focal-gamma
  rs_sqrt   plain CE with square-root re-sampling: class c drawn with probability ~ n_c^q (--rs-q 0.5)
  rad       CE + structure-aware auxiliary head: multi-label BCE on the class's IDS components (--rad-weight);
            the head is training-only, model.pt holds the plain ResNet-18 classifier
  C-15 control (docs/experiments.md, "C-13 至 C-17 预注册"):
  rgls      ugls with random neighbours (--random-neighbors from `randnb`): each ugls neighbour replaced by a random
            class of nearly the same training count that is not look-alike to the class
Uniform weights over the k neighbours on purpose: csls and sals differ only in *which* classes
are neighbours (the key comparison of the paper).

Data: arrays written by prepare_hwdb.py (64x64 uint8, white background).
    python src/training/train_hwdb_classifier.py train --data data/hwdb --split lt100 --loss sals --seed 0 --out results/A1-01_hwdb_lt/sals_s0
    python src/training/train_hwdb_classifier.py prototypes --data data/hwdb --split lt100 --ckpt <ce run>/model.pt --out <ce run>
    python src/training/train_hwdb_classifier.py smoke     # CPU self-test on synthetic data
"""
import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import torchvision

REPO = Path(__file__).resolve().parents[2]
IDS_PATH = REPO / "OpenOCR-main/third_party/ucr_author_320ea5e8f141/Pandarenlql-UCR-320ea5e/files/IDS_V7.txt"
IDC = set("⿰⿱⿲⿳⿴⿵⿶⿷⿸⿹⿺⿻")


# ----------------------------------------------------------------------------- neighbours
def load_ids():
    ids = {}
    for line in open(IDS_PATH, encoding="utf-8-sig"):
        ch, sep, seq = line.rstrip("\n").partition(":")
        if sep:
            ids[ch] = seq.split()
    return ids


def ids_neighbors(classes, k):
    """Top-k classes sharing >= 1 IDS component, ranked by IDS edit distance (-1 padded)."""
    from rapidfuzz.distance import Levenshtein
    from rapidfuzz.process import cdist
    ids = load_ids()
    tokens = sorted({t for s in ids.values() for t in s})
    code = {t: chr(0xF0000 + i) for i, t in enumerate(tokens)}
    has = [c in ids for c in classes]
    strings = ["".join(code[t] for t in ids[c]) if c in ids else "" for c in classes]
    comps = [set(ids[c]) - IDC if c in ids else set() for c in classes]
    dist = cdist(strings, strings, scorer=Levenshtein.distance, dtype=np.int32, workers=-1)
    out = np.full((len(classes), k), -1, np.int64)
    for i, ci in enumerate(comps):
        if not has[i] or not ci:
            continue
        cand = [j for j in range(len(classes)) if j != i and has[j] and comps[j] & ci]
        cand.sort(key=lambda j: (dist[i, j], j))
        out[i, :len(cand[:k])] = cand[:k]
    return out


def ids_distance_fn(classes):
    from rapidfuzz.distance import Levenshtein
    ids = load_ids()

    def dist(a, b):
        ca, cb = classes[a], classes[b]
        if ca not in ids or cb not in ids:
            return None
        return Levenshtein.distance(ids[ca], ids[cb]), bool((set(ids[ca]) & set(ids[cb])) - IDC)
    return dist


def union_neighbors(ids_nb, glyph_nb, k):
    """IDS top-k/2 + glyph top-k/2, de-duplicated, then filled up in glyph order (-1 padded)."""
    out = np.full((len(ids_nb), k), -1, np.int64)
    for c in range(len(ids_nb)):
        merged = []
        for j in list(ids_nb[c, :k // 2]) + list(glyph_nb[c, :k // 2]) + list(glyph_nb[c]):
            if j >= 0 and j != c and j not in merged:
                merged.append(int(j))
            if len(merged) == k:
                break
        out[c, :len(merged)] = merged
    return out


# ----------------------------------------------------------------------------- model / data
def build_model(num_classes):
    """ResNet-18 for 64x64 grey glyphs: 3x3 stride-1 stem, the stock 2x max-pool kept
    (feature maps 32-16-8-4), i.e. CIFAR-ResNet-18 cost (~0.56 GMACs) at 64x64 input."""
    model = torchvision.models.resnet18(num_classes=num_classes)
    model.conv1 = nn.Conv2d(1, 64, 3, 1, 1, bias=False)
    return model


class AuxNet(nn.Module):
    """rad: ResNet-18 classifier plus a linear multi-label head on the pooled features (training only)."""

    def __init__(self, num_classes, num_aux, pos_rate=0.01):
        super().__init__()
        self.net = build_model(num_classes)
        self.aux = nn.Linear(self.net.fc.in_features, num_aux)
        nn.init.constant_(self.aux.bias, math.log(pos_rate / (1 - pos_rate)))  # prior init: small initial BCE
        self.aux_out = None

    def forward(self, x):
        n = self.net
        x = n.maxpool(n.relu(n.bn1(n.conv1(x))))
        f = torch.flatten(n.avgpool(n.layer4(n.layer3(n.layer2(n.layer1(x))))), 1)
        self.aux_out = self.aux(f)
        return n.fc(f)


def component_targets(classes):
    """Multi-hot IDS components (non-IDC tokens of the decomposition) per class; classes without IDS get zeros."""
    ids = load_ids()
    comps = [sorted(set(ids[c]) - IDC) if c in ids else [] for c in classes]
    vocab = sorted({t for cs in comps for t in cs})
    index = {t: i for i, t in enumerate(vocab)}
    target = np.zeros((len(classes), len(vocab)), np.float32)
    for c, cs in enumerate(comps):
        target[c, [index[t] for t in cs]] = 1
    return target


def augment(x):
    """Random affine on a GPU batch (x: N,1,H,W in [0,1], background 0)."""
    n = x.shape[0]
    angle = (torch.rand(n, device=x.device) - 0.5) * math.radians(10)
    scale = 0.9 + 0.2 * torch.rand(n, device=x.device)
    shear = (torch.rand(n, device=x.device) - 0.5) * 0.2
    shift = (torch.rand(n, 2, device=x.device) - 0.5) * 0.1
    cos, sin = torch.cos(angle) / scale, torch.sin(angle) / scale
    theta = torch.stack([torch.stack([cos, -sin + shear, shift[:, 0]], 1),
                         torch.stack([sin, cos, shift[:, 1]], 1)], 1)
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    return F.grid_sample(x, grid, align_corners=False, padding_mode="zeros")


def to_input(batch_uint8, device):
    return (255 - torch.from_numpy(batch_uint8).to(device, non_blocking=True).float()).div_(255).unsqueeze(1)


class Split:
    def __init__(self, data, name, indices=None):
        self.images = np.load(Path(data) / f"{name}_images.npy", mmap_mode="r")
        labels = np.load(Path(data) / f"{name}_labels.npy")
        idx = np.arange(len(labels)) if indices is None else indices
        self.idx = idx[labels[idx] >= 0]
        self.labels = labels

    def batches(self, size, shuffle, rng=None, order=None):
        if order is None:
            order = rng.permutation(self.idx) if shuffle else self.idx
        for s in range(0, len(order), size):
            chunk = np.sort(order[s:s + size])
            yield self.images[chunk], self.labels[chunk]


# ----------------------------------------------------------------------------- losses
class SoftLoss:
    def __init__(self, args, counts, neighbors):
        self.kind, self.eps = args.loss, args.eps
        C = len(counts)
        self.log_prior = torch.log(torch.as_tensor((counts + 1) / (counts + 1).sum(), dtype=torch.float32))
        logn = np.log(counts.clip(min=1))
        frac = (logn - logn.min()) / max(1e-8, logn.max() - logn.min())
        self.eps_c = torch.as_tensor(args.eps_min + (args.eps_max - args.eps_min) * frac, dtype=torch.float32)
        self.nbr = None
        if neighbors is not None:
            nbr = torch.as_tensor(neighbors[:, :args.k])
            self.valid = nbr >= 0
            self.nbr = nbr.clamp(min=0)
        self.C = C
        beta = getattr(args, "cb_beta", 0.9999)
        eff = (1 - beta ** counts) / (1 - beta)  # cb: effective number of samples
        w = 1 / np.maximum(eff, 1.0)
        self.cb_w = torch.as_tensor(w / w.sum() * C, dtype=torch.float32)
        self.focal_gamma, self.rad_weight = getattr(args, "focal_gamma", 2.0), getattr(args, "rad_weight", 0.5)
        self.comp = None

    def to(self, device):
        for name in ("log_prior", "eps_c", "nbr", "valid", "cb_w", "comp"):
            if getattr(self, name, None) is not None:
                setattr(self, name, getattr(self, name).to(device))
        return self

    def __call__(self, logits, y):
        kind = self.kind
        if kind.startswith("la"):
            logits = logits + self.log_prior
        logp = F.log_softmax(logits.float(), dim=1)
        nll = -logp.gather(1, y[:, None]).squeeze(1)
        if kind in ("ce", "la", "rs_sqrt"):
            return nll.mean()
        if kind == "cb":
            w = self.cb_w[y]
            return (w * nll).sum() / w.sum()
        if kind == "focal":
            return ((1 - torch.exp(-nll)) ** self.focal_gamma * nll).mean()
        if kind == "rad":
            aux = F.binary_cross_entropy_with_logits(self.aux_out.float(), self.comp[y], reduction="none").sum(1)
            return nll.mean() + self.rad_weight * aux.mean()
        if kind in ("ls", "la_ls"):
            return ((1 - self.eps) * nll - self.eps * logp.mean(1)).mean()
        eps = self.eps_c[y] if kind.endswith("_f") else torch.full_like(nll, self.eps)
        valid = self.valid[y]
        nbr_logp = logp.gather(1, self.nbr[y])
        n_valid = valid.sum(1)
        smooth = torch.where(n_valid > 0, -(nbr_logp * valid).sum(1) / n_valid.clamp(min=1), -logp.mean(1))
        return ((1 - eps) * nll + eps * smooth).mean()


# ----------------------------------------------------------------------------- evaluation
@torch.no_grad()
def evaluate(model, split, device, counts, classes, bs=1024):
    model.eval()
    preds, confs, labels = [], [], []
    for x, y in split.batches(bs, False):
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(to_input(x, device))
        p = logits.float().softmax(1)
        c, k = p.max(1)
        preds.append(k.cpu().numpy())
        confs.append(c.cpu().numpy())
        labels.append(y)
    pred, conf, y = map(np.concatenate, (preds, confs, labels))
    correct = pred == y
    group = np.where(counts[y] > 100, "many", np.where(counts[y] >= 20, "medium", "few"))
    res = {"acc": 100 * correct.mean(), "n": int(len(y))}
    for g in ("many", "medium", "few"):
        m = group == g
        res[f"acc_{g}"] = 100 * correct[m].mean() if m.any() else None
    bins = np.linspace(0, 1, 16)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    res["ece"] = 100 * ece
    dist = ids_distance_fn(classes)
    eds, look = [], []
    for a, b in zip(pred[~correct], y[~correct]):
        d = dist(int(a), int(b))
        if d is not None:
            eds.append(d[0])
            look.append(d[1])
    res["error_ids_ed_mean"] = float(np.mean(eds)) if eds else None
    res["error_lookalike_share"] = 100 * float(np.mean(look)) if look else None
    model.train()
    return res


# ----------------------------------------------------------------------------- commands
def train(args):
    device = torch.device(args.device)
    torch.backends.cudnn.benchmark = True  # fixed 64x64 input
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    data = Path(args.data)
    classes = (data / "classes.txt").read_text(encoding="utf-8")
    indices = np.load(data / f"train_{args.split}_indices.npy") if args.split != "full" else None
    tr = Split(data, "train", indices)
    counts = np.bincount(tr.labels[tr.idx], minlength=len(classes)).astype(np.float64)
    val_idx = np.random.default_rng(123).permutation(len(np.load(data / "val_labels.npy")))[:args.val_size]
    va, te = Split(data, "val", np.sort(val_idx)), Split(data, "test")
    neighbors = None
    if args.loss in ("sals", "sals_f", "la_sals_f", "la_sals"):
        neighbors = ids_neighbors(classes, args.k)
    elif args.loss in ("csls", "la_csls"):
        neighbors = np.load(args.neighbors)
    elif args.loss == "gls":
        neighbors = np.load(args.glyph_neighbors)[:, :args.k]
    elif args.loss in ("ugls", "ugls_f", "la_ugls"):
        neighbors = union_neighbors(ids_neighbors(classes, args.k), np.load(args.glyph_neighbors), args.k)
    elif args.loss == "rgls":
        neighbors = np.load(args.random_neighbors)[:, :args.k]
    crit = SoftLoss(args, counts, neighbors)
    if args.loss == "rad":
        crit.comp = torch.as_tensor(component_targets(classes))
        model = AuxNet(len(classes), crit.comp.shape[1], pos_rate=float(crit.comp.mean()))
    else:
        model = build_model(len(classes))
    crit = crit.to(device)
    model = model.to(device).to(memory_format=torch.channels_last)
    probs = None
    if args.loss == "rs_sqrt":  # per-sample probability so that class c is drawn ~ n_c^q
        n_lab = counts[tr.labels[tr.idx]]
        probs = n_lab ** (getattr(args, "rs_q", 0.5) - 1)
        probs = probs / probs.sum()
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=True, weight_decay=args.wd)
    steps = args.epochs * math.ceil(len(tr.idx) / args.bs)
    warm = math.ceil(len(tr.idx) / args.bs)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=2))
    log = []
    start = time.time()
    for epoch in range(args.epochs):
        order = None if probs is None else rng.choice(tr.idx, size=len(tr.idx), replace=True, p=probs)
        for x, y in tr.batches(args.bs, True, rng, order=order):
            xb = augment(to_input(x, device)).contiguous(memory_format=torch.channels_last)
            yb = torch.from_numpy(y.astype(np.int64)).to(device)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model(xb)
            if args.loss == "rad":
                crit.aux_out = model.aux_out
            loss = crit(logits, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
        val = evaluate(model, va, device, counts, classes)
        log.append({"epoch": epoch + 1, "loss": float(loss.detach()), "val": val, "minutes": (time.time() - start) / 60})
        print(json.dumps(log[-1]), flush=True)
    torch.save((model.net if args.loss == "rad" else model).state_dict(), out / "model.pt")
    result = {"val_final": log[-1]["val"], "test": evaluate(model, te, device, counts, classes),
              "train_samples": int(len(tr.idx)), "minutes": (time.time() - start) / 60}
    (out / "log.json").write_text(json.dumps(log, indent=1))
    (out / "result.json").write_text(json.dumps(result, indent=2))
    print("RESULT", json.dumps(result), flush=True)


@torch.no_grad()
def prototypes(args):
    """Data-driven neighbours for csls: cosine of class-mean penultimate features of a CE model."""
    device = torch.device(args.device)
    data = Path(args.data)
    classes = (data / "classes.txt").read_text(encoding="utf-8")
    indices = np.load(data / f"train_{args.split}_indices.npy") if args.split != "full" else None
    tr = Split(data, "train", indices)
    model = build_model(len(classes))
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    model.fc = nn.Identity()
    model = model.to(device).eval()
    sums = torch.zeros(len(classes), 512, device=device)
    n = torch.zeros(len(classes), device=device)
    for x, y in tr.batches(1024, False):
        f = model(to_input(x, device)).float()
        yb = torch.from_numpy(y.astype(np.int64)).to(device)
        sums.index_add_(0, yb, f)
        n.index_add_(0, yb, torch.ones_like(yb, dtype=torch.float))
    proto = F.normalize(sums / n.clamp(min=1)[:, None], dim=1)
    sim = proto @ proto.T
    sim.fill_diagonal_(-2)
    nbr = sim.topk(args.k, dim=1).indices.cpu().numpy()
    np.save(Path(args.out) / f"csls_neighbors_k{args.k}.npy", nbr)
    print("saved data-driven neighbours", nbr.shape)


def random_neighbors(args):
    """C-15: replace each ugls neighbour j of class c (same k, same slot order) by a random class within +-window ranks
    of j in the split's training-count order (so its count is nearly n_j), that is not look-alike to c (no shared IDS
    component, not in either class's glyph top-50) and not already used for c; the window doubles if nothing is left."""
    data = Path(args.data)
    classes = (data / "classes.txt").read_text(encoding="utf-8")
    labels = np.load(data / "train_labels.npy")
    lab = labels[np.load(data / f"train_{args.split}_indices.npy")]
    C = len(classes)
    counts = np.bincount(lab[lab >= 0], minlength=C)
    glyph = np.load(args.glyph_neighbors)
    ugls = union_neighbors(ids_neighbors(classes, args.k), glyph, args.k)
    ids = load_ids()
    comps = [set(ids[c]) - IDC if c in ids else set() for c in classes]
    gsets = [set(glyph[c].tolist()) for c in range(C)]

    def look(a, b):
        return bool(comps[a] & comps[b]) or b in gsets[a] or a in gsets[b]
    order = sorted(range(C), key=lambda c: (-counts[c], ord(classes[c])))
    rank = np.empty(C, np.int64)
    rank[order] = np.arange(C)
    rng = np.random.default_rng(args.seed)
    out = np.full((C, args.k), -1, np.int64)
    widened = 0
    for c in range(C):
        used = {c}
        for s, j in enumerate(ugls[c]):
            if j < 0:
                continue
            w = args.window
            while True:
                cand = [x for x in order[max(0, rank[j] - w):rank[j] + w + 1] if x not in used and not look(c, x)]
                if cand or w > C:
                    break
                w *= 2
                widened += 1
            pick = int(rng.choice(cand))
            out[c, s] = pick
            used.add(pick)
    np.save(args.out, out)
    valid = ugls >= 0
    logn = np.log(counts + 1.0)
    rows = np.repeat(np.arange(C)[:, None], args.k, 1)
    summary = {"k": args.k, "window": args.window, "seed": args.seed, "widened": widened,
               "slots": int(valid.sum()), "slots_filled": int((out >= 0).sum()),
               "mean_abs_log_count_diff": float(np.abs(logn[out[valid]] - logn[ugls[valid]]).mean()),
               "mean_log_count_ugls": float(logn[ugls[valid]].mean()), "mean_log_count_rgls": float(logn[out[valid]].mean()),
               "lookalike_share_ugls": float(np.mean([look(a, b) for a, b in zip(rows[valid], ugls[valid])])),
               "lookalike_share_rgls": float(np.mean([look(a, b) for a, b in zip(rows[valid], out[valid])])),
               "neighbour_rarer_than_class_ugls": float((counts[ugls[valid]] < counts[rows[valid]]).mean()),
               "neighbour_rarer_than_class_rgls": float((counts[out[valid]] < counts[rows[valid]]).mean())}
    Path(args.out).with_name(Path(args.out).stem + "_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary))


def smoke(args):
    """Tiny synthetic dataset on CPU: checks the whole pipeline end to end."""
    import tempfile
    tmp = Path(tempfile.mkdtemp())
    ids = load_ids()
    classes = "".join(sorted(c for c in ids if "一" <= c <= "鿿")[:40])
    (tmp / "classes.txt").write_text(classes, encoding="utf-8")
    rng = np.random.default_rng(0)
    for name, n in (("train", 800), ("val", 200), ("test", 200)):
        np.save(tmp / f"{name}_images.npy", rng.integers(0, 256, (n, 64, 64), dtype=np.uint8))
        np.save(tmp / f"{name}_labels.npy", rng.integers(0, len(classes), n).astype(np.int32))
    np.save(tmp / "train_lt100_indices.npy", np.arange(0, 800, 2))
    glyph = np.stack([rng.permutation([j for j in range(len(classes)) if j != c])[:20] for c in range(len(classes))])
    np.save(tmp / "glyph.npy", glyph)
    for loss in args.losses:
        ns = argparse.Namespace(data=str(tmp), split="lt100", loss=loss, seed=0, out=str(tmp / loss), device="cpu",
                                epochs=1, bs=64, lr=0.05, wd=5e-4, eps=0.1, eps_min=0.05, eps_max=0.2, k=10,
                                neighbors=None, glyph_neighbors=str(tmp / "glyph.npy"), val_size=100)
        train(ns)
    if "ce" not in args.losses:
        print("SMOKE OK (subset)", tmp)
        return
    prototypes(argparse.Namespace(data=str(tmp), split="lt100", ckpt=str(tmp / "ce/model.pt"), out=str(tmp / "ce"),
                                  device="cpu", k=10))
    for loss in ("csls", "la_csls"):
        train(argparse.Namespace(data=str(tmp), split="lt100", loss=loss, seed=0, out=str(tmp / loss), device="cpu",
                                 epochs=1, bs=64, lr=0.05, wd=5e-4, eps=0.1, eps_min=0.05, eps_max=0.2, k=10,
                                 neighbors=str(tmp / "ce/csls_neighbors_k10.npy"), glyph_neighbors=None, val_size=100))
    print("SMOKE OK", tmp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("train")
    p.add_argument("--data", required=True)
    p.add_argument("--split", default="lt100", help="lt100 | full")
    p.add_argument("--loss", choices=("ce", "ls", "csls", "sals", "sals_f", "gls", "ugls", "ugls_f", "la", "la_sals_f",
                                      "la_ls", "la_sals", "la_ugls", "la_csls", "cb", "focal", "rs_sqrt", "rad", "rgls"),
                   required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--bs", type=int, default=512)
    p.add_argument("--lr", type=float, default=0.2)
    p.add_argument("--wd", type=float, default=5e-4)
    p.add_argument("--eps", type=float, default=0.1)
    p.add_argument("--eps-min", dest="eps_min", type=float, default=0.05)
    p.add_argument("--eps-max", dest="eps_max", type=float, default=0.2)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--neighbors", help="csls: .npy from the prototypes command")
    p.add_argument("--glyph-neighbors", dest="glyph_neighbors", help="gls/ugls: .npy from glyph_neighbors.py")
    p.add_argument("--random-neighbors", dest="random_neighbors", help="rgls: .npy from the randnb command")
    p.add_argument("--val-size", dest="val_size", type=int, default=50000)
    p.add_argument("--cb-beta", dest="cb_beta", type=float, default=0.9999)
    p.add_argument("--focal-gamma", dest="focal_gamma", type=float, default=2.0)
    p.add_argument("--rad-weight", dest="rad_weight", type=float, default=0.5)
    p.add_argument("--rs-q", dest="rs_q", type=float, default=0.5)
    p = sub.add_parser("prototypes")
    p.add_argument("--data", required=True)
    p.add_argument("--split", default="lt100")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--k", type=int, default=10)
    p = sub.add_parser("randnb")
    p.add_argument("--data", required=True)
    p.add_argument("--split", default="lt100")
    p.add_argument("--glyph-neighbors", dest="glyph_neighbors", required=True)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--window", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True, help=".npy path")
    p = sub.add_parser("smoke")
    p.add_argument("--losses", nargs="+", default=["ce", "ls", "sals", "sals_f", "gls", "ugls", "ugls_f", "la", "la_sals_f",
                                                   "la_ls", "la_sals", "la_ugls"])
    args = parser.parse_args()
    {"train": train, "prototypes": prototypes, "randnb": random_neighbors, "smoke": smoke}[args.cmd](args)
