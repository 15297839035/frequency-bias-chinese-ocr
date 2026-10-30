"""C-04b: CTC top-K dumps of OpenOCR's practical Chinese models (openocr_svtrv2_ch / openocr_repsvtr_ch) on BCTR.

Each model keeps its own preprocessing (BGR, RecDynamicResize to height 48, min width 320) and dictionary
(ppocr_keys_v1 + space). The space class is folded into blank in the stored top-K (index 6624 -> 0), which leaves
the space-stripped greedy transcript unchanged. Samples and references are those of the C-03 GTC dumps (same
sample_ids), read straight from the BCTR LMDBs, so a0_prior_correction / c4_analysis work unchanged.

    python ly/scripts/c4b_dump_industrial.py --model svtrv2_ch --split test --domain scene \
        --ref ly/exp/C-03_analysis/dumps --out ly/exp/C-04b_industrial/dumps
"""
import argparse
import json
import math
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parents[2]
OCR = REPO / "OpenOCR-main"
sys.path.insert(0, str(OCR))

import cv2  # noqa: E402
import lmdb  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from openrec.modeling import build_model  # noqa: E402

EXP = REPO / "ly/exp/C-04b_industrial"
MODELS = {"svtrv2_ch": "svtrv2_ch.yml", "repsvtr_ch": "repsvtr_ch.yml"}
H, MIN_W = 48, 320
SPACE = 6624


def preprocess(buf):
    img = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)  # BGR, like DecodeImage(img_mode=BGR)
    h, w = img.shape[:2]
    ratio = w / float(h)
    width = int(H * max(ratio, MIN_W / H))
    rw = min(width, int(math.ceil(H * ratio)))
    x = cv2.resize(img, (rw, H)).astype("float32").transpose(2, 0, 1) / 255
    x = (x - 0.5) / 0.5
    out = np.zeros((3, H, width), np.float32)
    out[:, :, :rw] = x
    return out


def load_model(name):
    cfg = yaml.safe_load((EXP / "configs" / MODELS[name]).read_text(encoding="utf-8"))
    arch = cfg["Architecture"]
    arch["Decoder"]["out_channels"] = SPACE + 1
    model = build_model(arch)
    saved = torch.load(EXP / "models" / f"openocr_{name}.pth", map_location="cpu", weights_only=True)
    state = saved.get("state_dict", saved) if isinstance(saved, dict) else saved
    # training-only guidance branch (GTC head) is not part of the CTC inference graph
    state = {k: v for k, v in state.items() if not k.startswith(("decoder.gtc_head.", "decoder.before_gtc."))}
    model.load_state_dict(state, strict=True)
    return model.eval()


def greedy(top1):
    out, prev = [], 0
    for c in top1:
        if c != 0 and c != prev:
            out.append(int(c))
        prev = c
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=sorted(MODELS), required=True)
    p.add_argument("--split", choices=("val", "test"), required=True)
    p.add_argument("--domain", choices=("scene", "web", "document"), required=True)
    p.add_argument("--ref", type=Path, default=REPO / "ly/exp/C-03_analysis/dumps")
    p.add_argument("--out", type=Path, default=EXP / "dumps")
    p.add_argument("--dataset-root", dest="dataset_root", type=Path, default=Path("/root/autodl-tmp/bctr"))
    p.add_argument("--topk", type=int, default=16)
    p.add_argument("--bs", type=int, default=128)
    args = p.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = load_model(args.model).cuda()
    chars = ["blank"] + (OCR / "tools/utils/ppocr_keys_v1.txt").read_text(encoding="utf-8").splitlines()
    ref = [json.loads(l) for l in open(args.ref / f"gtc_{args.split}_{args.domain}.jsonl", encoding="utf-8")]
    env = lmdb.open(str(args.dataset_root / args.domain / f"{args.domain}_{args.split}"), readonly=True, lock=False,
                    readahead=False)
    with env.begin() as txn:
        imgs = [preprocess(txn.get(b"image-%09d" % int(r["sample_id"].split(":")[1]))) for r in ref]
    by_width = {}
    for i, x in enumerate(imgs):
        by_width.setdefault(x.shape[2], []).append(i)
    results = [None] * len(ref)
    start = time.time()
    with torch.inference_mode():
        for width, idx in by_width.items():
            for s in range(0, len(idx), args.bs):
                chunk = idx[s:s + args.bs]
                probs = model(torch.from_numpy(np.stack([imgs[i] for i in chunk])).cuda()).float()
                logp = probs.clamp_min(1e-30).log().cpu()
                top_v, top_i = logp.topk(args.topk, dim=2)
                top_i[top_i == SPACE] = 0  # space -> blank
                for b, i in enumerate(chunk):
                    results[i] = (top_i[b].to(torch.int16).numpy(), top_v[b].to(torch.float16).numpy(),
                                  logp[b, :, 0].to(torch.float16).numpy())
    args.out.mkdir(parents=True, exist_ok=True)
    stem = args.out / f"{args.model}_{args.split}_{args.domain}"
    offsets = np.concatenate([[0], np.cumsum([r[0].shape[0] for r in results])]).astype(np.int64)
    np.savez(stem.with_suffix(".npz"), topk_idx=np.concatenate([r[0] for r in results]),
             topk_logp=np.concatenate([r[1] for r in results]), blank_logp=np.concatenate([r[2] for r in results]),
             offsets=offsets)
    with open(stem.with_suffix(".jsonl"), "w", encoding="utf-8") as fh:
        for r, res in zip(ref, results):
            pred = "".join(chars[c] for c in greedy(res[0][:, 0]))
            fh.write(json.dumps({"sample_id": r["sample_id"], "target": r["target"], "prediction": pred},
                                ensure_ascii=False) + "\n")
    meta = {"model": args.model, "split": args.split, "domain": args.domain, "samples": len(ref),
            "seconds": round(time.time() - start, 1), "widths": len(by_width)}
    stem.with_suffix(".meta.json").write_text(json.dumps(meta) + "\n")
    print(json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
