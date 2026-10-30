"""C-16 (a): BCTR's released attention baselines (TransOCR, ASTER; one model per domain) with step-wise frequency
correction -- external, non-SVTR models, inference only.

Models come from ly/third_party/bctr_baselines (authors' code, unmodified); preprocessing follows the authors' test
scripts (RGB, resized to 256 x 32, (x - 0.5) / 0.5). Decoding is our own greedy loop (the authors' ASTER evaluation
feeds the ground-truth previous token): at every step, if the argmax is a real character and its probability is
below gamma, the character is re-chosen as argmax_c (z_c - tau * log pi_c) over real characters; specials / END are
never changed and the chosen token is fed back (as C-09 for SMTR). pi = add-one character counts of the model's own
training domain (`counts`). Samples and reference strings are those of a reference CTC dump (C-05), so the files
line up with every other model; outputs use the C-09 format, so c9_analysis.py evaluates them unchanged.

    python ly/scripts/c16_bctr_baselines.py counts --dataset-root /root/autodl-tmp/bctr --out ly/exp/C-16_external
    python ly/scripts/c16_bctr_baselines.py infer --model transocr --checkpoint <dir>/transocr_scene_base.pth \
        --domain scene --split val --counts ly/exp/C-16_external/train_counts_scene.json \
        --ref ly/exp/C-05_crnn/dumps/crnn_s1_val_scene.jsonl --out ly/exp/C-16_external/baselines
    python ly/scripts/c9_analysis.py --preds ly/exp/C-16_external/baselines --out ly/exp/C-16_external/baselines \
        --norm official --tags transocr aster
Protocol: ly/commit.md, entry "C-13 至 C-17 预注册".
"""
import argparse
import io
import json
from pathlib import Path
import sys
import time
import types

REPO = Path(__file__).resolve().parents[2]
THIRD = REPO / "ly/third_party/bctr_baselines"
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

GRID = [(0.0, 1.01)] + [(t, g) for t in (0.1, 0.2, 0.3, 0.5, 1.0) for g in (0.9, 1.01)]  # = C-09
MAX_STEPS = 30
TRAIN = ("document", "scene", "web")


def cfg_name(tau, gamma):
    return "B0" if tau == 0 else f"t{tau}_g{gamma}"


def strq2b(text):
    """Full-width to half-width, as in the authors' datasets."""
    out = []
    for ch in text:
        code = ord(ch)
        if code == 12288:
            code = 32
        elif 65281 <= code <= 65374:
            code -= 65248
        out.append(chr(code))
    return "".join(out)


def alphabet(model):
    path = THIRD / ("TransOCR/data/benchmark.txt" if model == "transocr" else "ASTER/data/benchmark_new.txt")
    return list(path.read_text(encoding="utf-8").strip())


def counts(args):
    """Per training domain: character counts of the labels after strQ2B + lower (the TransOCR training transform)."""
    import collections
    import lmdb
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for dom in TRAIN:
        env = lmdb.open(str(Path(args.dataset_root) / dom / f"{dom}_train"), readonly=True, lock=False, readahead=False)
        c = collections.Counter()
        with env.begin() as txn:
            for i in range(1, int(txn.get(b"num-samples")) + 1):
                c.update(strq2b(txn.get(b"label-%09d" % i).decode("utf-8")).lower())
        env.close()
        (out / f"train_counts_{dom}.json").write_text(json.dumps(c, ensure_ascii=False))
        print(dom, "lines read; distinct characters", len(c), flush=True)


def sha12(path):
    import hashlib
    sha = hashlib.sha256()
    with open(path, "rb") as stream:
        while block := stream.read(1 << 23):
            sha.update(block)
    return sha.hexdigest()[:12]


def load_state(path):
    import torch
    try:
        saved = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:  # old checkpoints store numpy scalars (e.g. best_res); allow exactly those, still no code
        torch.serialization.add_safe_globals([np.core.multiarray.scalar, np.dtype, type(np.dtype("float64"))])
        saved = torch.load(path, map_location="cpu", weights_only=True)
    state = saved["state_dict"] if isinstance(saved, dict) and "state_dict" in saved else saved
    return {(k[len("module."):] if k.startswith("module.") else k): v for k, v in state.items()}


class TransOCRRunner:
    """START = 0, '\\xad' = 1, characters 2..V-2, END = V-1 (authors' get_alphabet)."""

    def __init__(self, ckpt, device):
        import torch
        self.chars = alphabet("transocr")
        raw = ["START", "\xad"] + self.chars + ["END"]
        sys.modules["utils"] = types.SimpleNamespace(get_alphabet=lambda args, kind: raw)  # stand-in, see SOURCE.md
        sys.path.insert(0, str(THIRD / "TransOCR"))
        from model.transocr import Transformer
        self.model = Transformer(types.SimpleNamespace(radical=False)).to(device)
        self.model.load_state_dict(load_state(ckpt), strict=True)
        self.model.eval()
        self.V, self.first, self.end = len(raw), 2, len(raw) - 1
        self.raw, self.char_mask = raw, [self.first <= k < self.end for k in range(len(raw))]
        self.torch = torch

    def features(self, x):
        return self.model(x, None, None)["conv"]

    def decode(self, x, feats, lp, tau, gamma):
        torch = self.torch
        b = x.shape[0]
        pred = torch.zeros(b, 1, dtype=torch.long, device=x.device)
        done = torch.zeros(b, dtype=torch.bool, device=x.device)
        for i in range(MAX_STEPS):
            length = torch.full((b,), i + 1, dtype=torch.long, device=x.device)
            z = self.model(x, length, pred, conv_feature=feats, test=True)["pred"][:, -1, :].float()
            nxt = choose(z, lp, tau, gamma, self.mask_t, self.end)
            nxt = torch.where(done, torch.full_like(nxt, self.end), nxt)
            pred = torch.cat([pred, nxt[:, None]], 1)
            done |= nxt == self.end
            if done.all():
                break
        return pred[:, 1:]


class ASTERRunner:
    """Label map of the authors' utils/radical_dict.py: PADDING = 0, UNKNOW = 1, '\\xad' = 2, characters 3..A-2,
    'END' = A-1 (A = 7,938 classes); BOS = A (the head's extra embedding row). The released base checkpoints end every
    word with '$' (id 4,926) -- the authors' evaluation (evaluation_metrics/metrics.py) cuts the output at '$' and skips
    PADDING -- so '$' is the stop symbol here and not a character; ids 0-2 and 'END' are never emitted as text."""

    def __init__(self, ckpt, device):
        import torch
        self.chars = alphabet("aster")
        raw = ["PADDING", "UNKNOW", "\xad"] + self.chars + ["END"]
        sys.modules["config"] = types.SimpleNamespace(get_args=lambda argv: types.SimpleNamespace())  # stand-in
        sys.path.insert(0, str(THIRD / "ASTER"))
        from model.attention_recognition_head import AttentionRecognitionHead
        from model.resnet_aster import ResNet_ASTER
        self.encoder = ResNet_ASTER(with_lstm=True, n_group=1).to(device)
        self.head = AttentionRecognitionHead(num_classes=len(raw), in_planes=self.encoder.out_planes, sDim=512,
                                             attDim=512, max_len_labels=MAX_STEPS).to(device)
        state = load_state(ckpt)
        for prefix, module in (("encoder.", self.encoder), ("decoder.", self.head)):
            module.load_state_dict({k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}, strict=True)
        self.encoder.eval()
        self.head.eval()
        self.num_classes, self.first = len(raw), 3
        self.end = raw.index("$")
        self.raw = raw
        self.char_mask = [self.first <= k < len(raw) - 1 and k != self.end for k in range(len(raw))]
        self.torch = torch

    def features(self, x):
        return self.encoder(x)[0].contiguous()

    def decode(self, x, feats, lp, tau, gamma):
        torch = self.torch
        b = feats.shape[0]
        state = torch.zeros(1, b, self.head.sDim, device=feats.device)
        y = torch.full((b,), self.num_classes, dtype=torch.long, device=feats.device)  # BOS
        done = torch.zeros(b, dtype=torch.bool, device=feats.device)
        out = []
        for _ in range(MAX_STEPS):
            z, state, _ = self.head.decoder(feats, state, y)
            nxt = choose(z.float(), lp, tau, gamma, self.mask_t, self.end)
            nxt = torch.where(done, torch.full_like(nxt, self.end), nxt)
            out.append(nxt)
            done |= nxt == self.end
            y = nxt
            if done.all():
                break
        return torch.stack(out, 1)


def choose(z, lp, tau, gamma, mask, end):
    """argmax; a real character (mask) below gamma is re-chosen among the characters with the prior correction.
    lp: log prior over all classes (only character entries matter)."""
    import torch
    pred = z.argmax(-1)
    if tau:
        p = torch.softmax(z, -1).max(-1).values
        alt = (z - tau * lp).masked_fill(~mask, float("-inf")).argmax(-1)
        change = mask[pred] & (p < gamma)
        pred = torch.where(change, alt, pred)
    return pred


def to_text(ids, raw, mask, end):
    out = []
    for i in ids:
        if i == end:
            break
        if mask[i]:
            out.append(raw[i])
    return "".join(out)


def infer(args):
    import lmdb
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms
    if args.device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    runner = (TransOCRRunner if args.model == "transocr" else ASTERRunner)(args.checkpoint, args.device)
    cnt = json.loads(Path(args.counts).read_text(encoding="utf-8"))
    mask = np.array(runner.char_mask)
    c = np.array([cnt.get(ch, 0) if m else 0 for ch, m in zip(runner.raw, mask)], np.float64)
    lp_np = np.zeros(len(runner.raw))
    lp_np[mask] = np.log((c[mask] + 1) / (c[mask] + 1).sum())  # add-one prior over the characters only
    lp = torch.as_tensor(lp_np, dtype=torch.float32, device=args.device)
    runner.mask_t = torch.as_tensor(mask, device=args.device)
    ref = [json.loads(line) for line in open(args.ref, encoding="utf-8")]
    lmdb_path = Path(args.dataset_root) / args.domain / f"{args.domain}_{args.split}"
    to_tensor = transforms.ToTensor()

    class Lines(Dataset):
        def __len__(self):
            return len(ref)

        def __getitem__(self, i):
            if not hasattr(self, "txn"):
                self.txn = lmdb.open(str(lmdb_path), readonly=True, lock=False, readahead=False).begin()
            sid = ref[i]["sample_id"]
            assert sid.startswith(f"{args.domain}_{args.split}:"), sid
            img = Image.open(io.BytesIO(self.txn.get(b"image-%09d" % int(sid.split(":")[1])))).convert("RGB")
            return (to_tensor(img.resize((256, 32), Image.BILINEAR)) - 0.5) / 0.5, i

    loader = DataLoader(Lines(), batch_size=args.batch, num_workers=args.workers)
    args.out.mkdir(parents=True, exist_ok=True)
    stem = args.out / f"{args.model}_{args.split}_{args.domain}"
    rows, start = [None] * len(ref), time.time()
    with torch.inference_mode():
        for number, (x, idx) in enumerate(loader, 1):
            x = x.to(args.device)
            feats = runner.features(x)
            preds = {cfg_name(t, g): runner.decode(x, feats, lp, t, g).cpu().tolist() for t, g in GRID}
            for b, i in enumerate(idx.tolist()):
                rows[i] = {"sample_id": ref[i]["sample_id"], "target": ref[i]["target"],
                           "preds": {k: to_text(v[b], runner.raw, runner.char_mask, runner.end) for k, v in preds.items()}}
            if number % 20 == 0:
                print(f"{stem.name}: {number}/{len(loader)} batches, {time.time() - start:.0f}s", flush=True)
            if args.limit and number >= args.limit:
                rows = [r for r in rows if r is not None]
                break
    with open(stem.with_suffix(".jsonl"), "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    meta = {"model": args.model, "checkpoint": str(args.checkpoint), "checkpoint_sha256_12": sha12(args.checkpoint),
            "split": args.split, "domain": args.domain, "samples": len(rows), "grid": [cfg_name(t, g) for t, g in GRID],
            "prior": str(args.counts), "reference": str(args.ref), "limit": args.limit, "torch": torch.__version__,
            "seconds": round(time.time() - start, 1)}
    stem.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("counts")
    p.add_argument("--dataset-root", dest="dataset_root", default="/root/autodl-tmp/bctr")
    p.add_argument("--out", required=True)
    p = sub.add_parser("infer")
    p.add_argument("--model", choices=("transocr", "aster"), required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--domain", choices=TRAIN, required=True)
    p.add_argument("--split", choices=("val", "test"), required=True)
    p.add_argument("--counts", type=Path, required=True)
    p.add_argument("--ref", type=Path, required=True, help="reference CTC dump .jsonl (sample ids and targets)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--dataset-root", dest="dataset_root", default="/root/autodl-tmp/bctr")
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--limit", type=int, default=0, help="debug: only the first N batches")
    args = parser.parse_args()
    {"counts": counts, "infer": infer}[args.cmd](args)
