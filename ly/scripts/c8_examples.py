"""C-08 / W-03: example crops for the label-noise audit and Figure 1 (GTC, BCTR test, strict text, G* = tau 0.2,
gamma 0.9). Categories: fixed (rare line, B0 wrong, G* right), broken_rare, broken_common, and a random audit sample
of rare lines whose B0 output substitutes a rare reference character (seed 0). Writes one PNG per sample, a JSON
index and numbered contact sheets (no CJK font on the server: texts are in the JSON).

    python ly/scripts/c8_examples.py --out ly/exp/C-08_ablation/examples
"""
import argparse
import json
from pathlib import Path
import sys

import lmdb
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
import a0_prior_correction as a0  # noqa: E402
import c3_analysis as c3  # noqa: E402

GSTAR = dict(method="G", tau=0.2, k=0, gamma=0.9)
ROOT = Path("/root/autodl-tmp/bctr")


def image(sample_id, envs):
    split, idx = sample_id.split(":")
    dom = split.split("_")[0]
    if split not in envs:
        envs[split] = lmdb.open(str(ROOT / dom / split), readonly=True, lock=False, readahead=False)
    with envs[split].begin() as txn:
        buf = txn.get(f"image-{int(idx):09d}".encode())
    import io
    return Image.open(io.BytesIO(buf)).convert("RGB")


def sheet(paths, out, h=48, width=900):
    rows = []
    for i, p in enumerate(paths):
        im = Image.open(p)
        w = max(1, int(im.width * h / im.height))
        im = im.resize((min(w, width - 50), h))
        row = Image.new("RGB", (width, h + 6), "white")
        row.paste(im, (50, 3))
        ImageDraw.Draw(row).text((5, h // 2 - 5), f"{i:03d}", fill="red")
        rows.append(row)
    canvas = Image.new("RGB", (width, sum(r.height for r in rows)), "white")
    y = 0
    for r in rows:
        canvas.paste(r, (0, y))
        y += r.height
    canvas.save(out)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--dumps", type=Path, default=a0.REPO / "ly/exp/C-03_analysis/dumps")
    p.add_argument("--per-category", dest="per_category", type=int, default=40)
    p.add_argument("--audit", type=int, default=100)
    args = p.parse_args()
    a0.DUMPS = args.dumps
    rows, b0, gs, rare = [], [], [], []
    for dom in a0.DOMAINS:
        d = a0.load("gtc", "test", dom)
        rows += d["rows"]
        b0 += d["baseline"]
        gs += a0.strings(d, a0.decode(d, GSTAR))
        rare += list(d["rare"])
    refs = [a0.normalize(r["target"]) for r in rows]
    rng = np.random.default_rng(0)
    cats = {"fixed": [], "broken_rare": [], "broken_common": [], "audit": []}
    for i, (r, b, g, rr) in enumerate(zip(refs, b0, gs, rare)):
        if rr and b != r and g == r:
            cats["fixed"].append(i)
        elif rr and b == r and g != r:
            cats["broken_rare"].append(i)
        elif not rr and b == r and g != r:
            cats["broken_common"].append(i)
        if rr and b != r and any(a0.is_han(x) and c3.count(x) < a0.RARE for x, _ in c3.substitutions(r, b)):
            cats["audit"].append(i)
    envs, index = {}, {}
    for cat, ids in cats.items():
        n = args.audit if cat == "audit" else args.per_category
        pick = sorted(rng.choice(ids, min(n, len(ids)), replace=False).tolist())
        folder = args.out / cat
        folder.mkdir(parents=True, exist_ok=True)
        entries, paths = [], []
        for k, i in enumerate(pick):
            path = folder / f"{k:03d}.png"
            image(rows[i]["sample_id"], envs).save(path)
            paths.append(path)
            entries.append({"k": k, "sample_id": rows[i]["sample_id"], "target": refs[i], "B0": b0[i], "Gstar": gs[i],
                            "B0_substitutions": [[x, y, c3.count(x), c3.count(y)] for x, y in c3.substitutions(refs[i], b0[i])],
                            "Gstar_substitutions": [[x, y, c3.count(x), c3.count(y)] for x, y in c3.substitutions(refs[i], gs[i])]})
        for s in range(0, len(paths), 25):
            sheet(paths[s:s + 25], args.out / f"{cat}_sheet{s // 25}.png")
        index[cat] = {"available": len(ids), "entries": entries}
        print(cat, "available", len(ids), "picked", len(pick), flush=True)
    (args.out / "index.json").write_text(json.dumps(index, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
