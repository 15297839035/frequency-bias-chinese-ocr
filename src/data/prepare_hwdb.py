"""A1: convert CASIA-HWDB offline .gnt files into 64x64 uint8 arrays and build HWDB-LT.

Inputs: directories of .gnt files (one file per writer), e.g. HWDB1.0trn, HWDB1.1trn (train),
HWDB1.0tst, HWDB1.1tst (validation), ICDAR2013 competition offline set (test).
Classes: the 3,755 GB2312 level-1 characters, taken as the label set of the test directory.

    python src/data/prepare_hwdb.py convert --name train --gnt-dirs HWDB1.0trn HWDB1.1trn --out data/hwdb
    python src/data/prepare_hwdb.py convert --name val   --gnt-dirs HWDB1.0tst HWDB1.1tst --out data/hwdb
    python src/data/prepare_hwdb.py convert --name test  --gnt-dirs competition_gnt        --out data/hwdb
    python src/data/prepare_hwdb.py classes --out data/hwdb          # class list = test labels
    python src/data/prepare_hwdb.py longtail --out data/hwdb --imbalance 100 --nmax 500
    python src/data/prepare_hwdb.py dose --out data/hwdb --alpha 0.25      # C-14 (also flip / natural)

HWDB-LT (pre-registered in docs/experiments.md, A1-00): class c keeps
n_c = round(nmax * imbalance ** (-r_c / (C - 1))) training samples, where r_c is the rank of c
by real-text frequency (BCTR training character counts; ties by code point), so the long tail
follows how often characters occur in real text. Samples are drawn without replacement, seed 0.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

SIZE = 64
INNER = 56  # longer side after resize; the rest is white padding
REPO = Path(__file__).resolve().parents[2]
FREQ = REPO / "OpenOCR-main/output/rec/bctr_scheme51_ab_bs64/train_character_frequency.json"
if not FREQ.exists():  # artifact repository layout
    FREQ = REPO / "data/bctr/train_character_frequency.json"


def read_gnt(path):
    """Yield (character, uint8 image) from one CASIA .gnt file."""
    data = Path(path).read_bytes()
    pos = 0
    while pos + 10 <= len(data):
        size = int.from_bytes(data[pos:pos + 4], "little")
        tag = data[pos + 4:pos + 6]
        # Symbols are stored as one ASCII byte + 0x00 ("!\x00"); decoding both bytes would yield two code
        # points and shift every later label in *_chars.txt against its image. Keep exactly one per record.
        char = chr(tag[0]) if tag[1] == 0 else tag.decode("gbk", errors="replace")
        if len(char) != 1:
            char = "�"
        width = int.from_bytes(data[pos + 6:pos + 8], "little")
        height = int.from_bytes(data[pos + 8:pos + 10], "little")
        if size != 10 + width * height:
            raise ValueError(f"{path}: corrupt record at byte {pos}")
        image = np.frombuffer(data, np.uint8, width * height, pos + 10).reshape(height, width)
        yield char, image
        pos += size


def normalise(image):
    """Keep aspect ratio, longer side -> INNER, centre on a white SIZE x SIZE canvas."""
    h, w = image.shape
    scale = INNER / max(h, w)
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    resized = np.asarray(Image.fromarray(image).resize((nw, nh), Image.BILINEAR))
    canvas = np.full((SIZE, SIZE), 255, np.uint8)
    top, left = (SIZE - nh) // 2, (SIZE - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    return canvas


def convert(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(p for d in args.gnt_dirs for p in Path(d).rglob("*.gnt"))
    if not files:
        raise SystemExit("no .gnt files found")
    count = sum(1 for f in files for _ in read_gnt(f))
    images = np.lib.format.open_memmap(out / f"{args.name}_images.npy", "w+", np.uint8, (count, SIZE, SIZE))
    chars, writers = [], []
    i = 0
    for writer, f in enumerate(files):
        for char, image in read_gnt(f):
            images[i] = normalise(image)
            chars.append(char)
            writers.append(writer)
            i += 1
        print(f"{args.name}: {writer + 1}/{len(files)} files, {i}/{count} samples", flush=True)
    images.flush()
    assert i == count and all(len(c) == 1 for c in chars)
    (out / f"{args.name}_chars.txt").write_text("".join(chars), encoding="utf-8")
    np.save(out / f"{args.name}_writers.npy", np.asarray(writers, np.int32))
    (out / f"{args.name}_files.json").write_text(json.dumps([str(f) for f in files], ensure_ascii=False))


def classes(args):
    out = Path(args.out)
    test = (out / "test_chars.txt").read_text(encoding="utf-8")
    classes = sorted(set(test))
    (out / "classes.txt").write_text("".join(classes), encoding="utf-8")
    reference = REPO / "results/A1-00_prep/gb2312_l1_classes.txt"
    if reference.is_file() and reference.read_text(encoding="utf-8") != "".join(classes):
        raise SystemExit("class list differs from the GB2312 level-1 list the glyph neighbours were built for")
    index = {c: i for i, c in enumerate(classes)}
    for name in ("train", "val", "test"):
        chars = (out / f"{name}_chars.txt").read_text(encoding="utf-8")
        labels = np.asarray([index.get(c, -1) for c in chars], np.int32)
        np.save(out / f"{name}_labels.npy", labels)
        print(name, "samples", len(labels), "outside the class set (dropped later):", int((labels < 0).sum()))
    print("classes", len(classes))


def longtail(args):
    out = Path(args.out)
    classes = (out / "classes.txt").read_text(encoding="utf-8")
    labels = np.load(out / "train_labels.npy")
    freq = json.loads(FREQ.read_text(encoding="utf-8"))
    real = dict(zip(freq["characters"], freq["counts"]))
    order = sorted(range(len(classes)), key=lambda c: (-real.get(classes[c], 0), ord(classes[c])))
    rank = np.empty(len(classes), np.int64)
    rank[order] = np.arange(len(classes))
    C = len(classes)
    target = np.round(args.nmax * args.imbalance ** (-rank / (C - 1))).astype(np.int64)
    rng = np.random.default_rng(0)
    keep = []
    for c in range(C):
        pool = np.flatnonzero(labels == c)
        if len(pool) < target[c]:
            raise SystemExit(f"class {classes[c]} has only {len(pool)} samples < {target[c]}; lower --nmax")
        keep.append(np.sort(rng.choice(pool, target[c], replace=False)))
    keep = np.concatenate(keep)
    name = f"lt{args.imbalance}"
    np.save(out / f"train_{name}_indices.npy", keep)
    counts = np.bincount(labels[keep], minlength=C)
    summary = {"imbalance": args.imbalance, "nmax": args.nmax, "classes": C, "samples": int(len(keep)),
               "many(>100)": int((counts > 100).sum()), "medium(20-100)": int(((counts >= 20) & (counts <= 100)).sum()),
               "few(<20)": int((counts < 20).sum()), "min_count": int(counts.min()),
               "ordering": "BCTR real-text frequency rank; ties by code point", "seed": 0}
    (out / f"train_{name}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary))


def natural(args):
    """C-01n: per-class counts proportional to BCTR training frequency, clipped to [nmin, min(nmax, available)],
    scaled (bisection on s) so the total matches --total (default: the LT100 split, 403,741)."""
    out = Path(args.out)
    classes = (out / "classes.txt").read_text(encoding="utf-8")
    labels = np.load(out / "train_labels.npy")
    freq = json.loads(FREQ.read_text(encoding="utf-8"))
    real = dict(zip(freq["characters"], freq["counts"]))
    C = len(classes)
    f = np.array([real.get(c, 0) for c in classes], np.float64)
    avail = np.bincount(labels[labels >= 0], minlength=C)
    hi_cap = np.minimum(args.nmax, avail)

    def counts(s):
        return np.clip(np.round(s * f), args.nmin, hi_cap).astype(np.int64)
    lo, hi = 0.0, 1.0
    while counts(hi).sum() < args.total:
        hi *= 2
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if counts(mid).sum() < args.total else (lo, mid)
    target = counts(hi)
    rng = np.random.default_rng(0)
    keep = np.concatenate([np.sort(rng.choice(np.flatnonzero(labels == c), target[c], replace=False)) for c in range(C)])
    np.save(out / "train_nat_indices.npy", keep)
    n = np.bincount(labels[keep], minlength=C)
    summary = {"profile": "BCTR frequency, clipped", "scale_s": hi, "nmin": args.nmin, "nmax": args.nmax,
               "total_target": args.total, "samples": int(len(keep)), "classes": C,
               "at_cap": int((n == hi_cap).sum()), "at_floor": int((n == args.nmin).sum()),
               "many(>100)": int((n > 100).sum()), "medium(20-100)": int(((n >= 20) & (n <= 100)).sum()),
               "few(<20)": int((n < 20).sum()), "min_count": int(n.min()), "max_count": int(n.max()), "seed": 0}
    (out / "train_nat_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary))


def flip(args):
    """C-07: the HWDB-LT count profile (same imbalance / nmax) assigned in another class order.
    rev: BCTR frequency rank reversed (rarest character gets nmax); perm: a random order (--perm-seed).
    A class with fewer images than its target keeps all of them (reported as capped)."""
    out = Path(args.out)
    classes = (out / "classes.txt").read_text(encoding="utf-8")
    labels = np.load(out / "train_labels.npy")
    freq = json.loads(FREQ.read_text(encoding="utf-8"))
    real = dict(zip(freq["characters"], freq["counts"]))
    C = len(classes)
    order = sorted(range(C), key=lambda c: (-real.get(classes[c], 0), ord(classes[c])))
    rank = np.empty(C, np.int64)
    rank[order] = np.arange(C)
    rank = C - 1 - rank if args.order == "rev" else np.random.default_rng(args.perm_seed).permutation(C)[rank]
    target = np.round(args.nmax * args.imbalance ** (-rank / (C - 1))).astype(np.int64)
    avail = np.bincount(labels[labels >= 0], minlength=C)
    capped = target > avail
    target = np.minimum(target, avail)
    rng = np.random.default_rng(0)
    keep = np.concatenate([np.sort(rng.choice(np.flatnonzero(labels == c), target[c], replace=False)) for c in range(C)])
    name = f"lt{args.imbalance}{args.order}"
    np.save(out / f"train_{name}_indices.npy", keep)
    counts = np.bincount(labels[keep], minlength=C)
    nat_rank = np.empty(C, np.int64)
    nat_rank[order] = np.arange(C)
    summary = {"order": args.order, "perm_seed": args.perm_seed if args.order == "perm" else None,
               "imbalance": args.imbalance, "nmax": args.nmax, "classes": C, "samples": int(len(keep)),
               "capped_classes": int(capped.sum()), "many(>100)": int((counts > 100).sum()),
               "medium(20-100)": int(((counts >= 20) & (counts <= 100)).sum()), "few(<20)": int((counts < 20).sum()),
               "min_count": int(counts.min()), "max_count": int(counts.max()),
               "spearman_vs_natural_rank": float(np.corrcoef(np.argsort(np.argsort(-counts, kind="stable")),
                                                             nat_rank)[0, 1]), "seed": 0}
    (out / f"train_{name}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary))


def dose(args):
    """C-14: dose-response between HWDB-LT (alpha = 0, natural BCTR order) and lt100rev (alpha = 1): class c gets
    n_c ~ nat_c^(1 - alpha) * rev_c^alpha of the two LT profiles (alpha = 0.5: every class equal), scaled (bisection,
    as in `natural`) so the total matches --total (the LT100 split, 403,741), capped at the available images."""
    out = Path(args.out)
    classes = (out / "classes.txt").read_text(encoding="utf-8")
    labels = np.load(out / "train_labels.npy")
    freq = json.loads(FREQ.read_text(encoding="utf-8"))
    real = dict(zip(freq["characters"], freq["counts"]))
    C = len(classes)
    order = sorted(range(C), key=lambda c: (-real.get(classes[c], 0), ord(classes[c])))
    rank = np.empty(C, np.int64)
    rank[order] = np.arange(C)
    nat = args.nmax * float(args.imbalance) ** (-rank / (C - 1))
    rev = args.nmax * float(args.imbalance) ** (-(C - 1 - rank) / (C - 1))
    profile = nat ** (1 - args.alpha) * rev ** args.alpha
    avail = np.bincount(labels[labels >= 0], minlength=C)

    def counts(s):
        return np.clip(np.round(s * profile), 1, avail).astype(np.int64)
    lo, hi = 0.0, 1.0
    while counts(hi).sum() < args.total:
        hi *= 2
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if counts(mid).sum() < args.total else (lo, mid)
    target = counts(hi)
    rng = np.random.default_rng(0)
    keep = np.concatenate([np.sort(rng.choice(np.flatnonzero(labels == c), target[c], replace=False)) for c in range(C)])
    name = f"lt{args.imbalance}dose{round(100 * args.alpha):03d}"
    np.save(out / f"train_{name}_indices.npy", keep)
    n = np.bincount(labels[keep], minlength=C)
    summary = {"alpha": args.alpha, "imbalance": args.imbalance, "nmax": args.nmax, "scale_s": hi,
               "total_target": args.total, "samples": int(len(keep)), "classes": C,
               "capped_classes": int((n == avail).sum()), "min_count": int(n.min()), "max_count": int(n.max()),
               "spearman_vs_natural_rank": float(np.corrcoef(np.argsort(np.argsort(-n, kind="stable")), rank)[0, 1]),
               "log_imbalance_max_over_min": float(np.log(n.max() / n.min())), "seed": 0}
    (out / f"train_{name}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("dose")
    p.add_argument("--out", required=True)
    p.add_argument("--alpha", type=float, required=True)
    p.add_argument("--imbalance", type=int, default=100)
    p.add_argument("--nmax", type=int, default=500)
    p.add_argument("--total", type=int, default=403741)
    p = sub.add_parser("flip")
    p.add_argument("--out", required=True)
    p.add_argument("--order", choices=("rev", "perm"), required=True)
    p.add_argument("--imbalance", type=int, default=100)
    p.add_argument("--nmax", type=int, default=500)
    p.add_argument("--perm-seed", dest="perm_seed", type=int, default=0)
    p = sub.add_parser("convert")
    p.add_argument("--name", choices=("train", "val", "test"), required=True)
    p.add_argument("--gnt-dirs", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("classes")
    p.add_argument("--out", required=True)
    p = sub.add_parser("longtail")
    p.add_argument("--out", required=True)
    p.add_argument("--imbalance", type=int, default=100)
    p.add_argument("--nmax", type=int, default=500)
    p = sub.add_parser("natural")
    p.add_argument("--out", required=True)
    p.add_argument("--nmin", type=int, default=2)
    p.add_argument("--nmax", type=int, default=500)
    p.add_argument("--total", type=int, default=403741)
    args = parser.parse_args()
    {"convert": convert, "classes": classes, "longtail": longtail, "natural": natural, "flip": flip,
     "dose": dose}[args.cmd](args)
