"""C-08 / W-03: label-noise audit sample. GTC, BCTR test, official normalisation; rare-reference Han substitutions of
B0 that are not traditional/simplified (or NFKC) variants; one random substitution per line, n lines (seed 0).
Writes grid sheets (3 x 3 cells, numbered) and an index with reference, B0 output, the substituted pair and G*.

    python ly/scripts/c8_audit.py --out ly/exp/C-08_ablation/audit --n 54
"""
import argparse
import io
import json
from pathlib import Path
import sys
import unicodedata

import lmdb
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
import a0_prior_correction as a0  # noqa: E402
import c3_analysis as c3  # noqa: E402
from c4_analysis import official_normalizer  # noqa: E402

GSTAR = dict(method="G", tau=0.2, k=0, gamma=0.9)
ROOT = Path("/root/autodl-tmp/bctr")


def load_image(sample_id, envs):
    split, idx = sample_id.split(":")
    if split not in envs:
        envs[split] = lmdb.open(str(ROOT / split.split("_")[0] / split), readonly=True, lock=False, readahead=False)
    with envs[split].begin() as txn:
        return Image.open(io.BytesIO(txn.get(f"image-{int(idx):09d}".encode()))).convert("RGB")


def grid(images, out, cell=360, cols=3):
    rows = (len(images) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * cell, rows * (cell + 20)), "white")
    draw = ImageDraw.Draw(canvas)
    for i, im in enumerate(images):
        s = min((cell - 10) / im.width, (cell - 10) / im.height)
        im = im.resize((max(1, int(im.width * s)), max(1, int(im.height * s))))
        x, y = (i % cols) * cell, (i // cols) * (cell + 20)
        draw.text((x + 5, y + 2), str(i), fill="red")
        canvas.paste(im, (x + 5, y + 18))
    canvas.save(out)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--n", type=int, default=54)
    args = p.parse_args()
    from opencc import OpenCC
    t2s = OpenCC("t2s").convert
    a0.normalize = official_normalizer()
    a0.DUMPS = a0.REPO / "ly/exp/C-03_analysis/dumps"
    cands = []
    for dom in a0.DOMAINS:
        d = a0.load("gtc", "test", dom)
        gs = a0.strings(d, a0.decode(d, GSTAR))
        for row, ref, b0, g in zip(d["rows"], d["targets"], d["baseline"], gs):
            subs = [(x, y) for x, y in c3.substitutions(ref, b0) if a0.is_han(x) and a0.is_han(y) and c3.count(x) < a0.RARE
                    and t2s(unicodedata.normalize("NFKC", x)) != t2s(unicodedata.normalize("NFKC", y))]
            if subs:
                cands.append((row["sample_id"], ref, b0, g, subs))
    rng = np.random.default_rng(0)
    pick = [cands[i] for i in sorted(rng.choice(len(cands), args.n, replace=False))]
    args.out.mkdir(parents=True, exist_ok=True)
    envs, entries, images = {}, [], []
    for k, (sid, ref, b0, g, subs) in enumerate(pick):
        x, y = subs[rng.integers(len(subs))]
        images.append(load_image(sid, envs))
        entries.append({"k": k, "sheet": k // 9, "cell": k % 9, "sample_id": sid, "reference": ref, "B0": b0, "Gstar": g,
                        "audited_pair": [x, y, c3.count(x), c3.count(y)], "lookalike": c3.lookalike(x, y)})
    for s in range(0, len(images), 9):
        grid(images[s:s + 9], args.out / f"audit_grid{s // 9}.png")
    (args.out / "index.json").write_text(json.dumps({"candidates": len(cands), "entries": entries}, indent=1, ensure_ascii=False))
    print("candidates", len(cands), "picked", len(pick))


if __name__ == "__main__":
    main()
