"""A0: dump per-frame CTC posteriors (top-K) for the seed48 Scheme 5.1 checkpoints.

Reuses, read-only, the frozen-test pipeline of OpenOCR-main (tools/eval_scheme51_frozen_test.py):
the same StrictValidationDataset, Scheme51RatioSampler (base batch 64), CTC-only inference graph,
FP32 and text normalisation, so the greedy decode can be checked against the frozen predictions.

Usage (from the repo root):
    ly/.venv/bin/python ly/scripts/a0_dump_ctc.py --model gtc --split test --domain scene
Writes ly/exp/A0-01_ctc_dumps/dumps/<model>_<split>_<domain>.npz and a matching .jsonl.
C-02/C-03 (2026-10-12): --checkpoint/--tag dump any Scheme 5.1 checkpoint (tag replaces <model> in the
file names), --out changes the output directory, --device cuda; defaults reproduce the A0 behaviour.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parents[2]
OCR = REPO / "OpenOCR-main"
sys.path.insert(0, str(OCR))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from openrec.modeling import build_model  # noqa: E402
from openrec.postprocess import build_post_process  # noqa: E402
from openrec.scheme51 import Scheme51RatioSampler  # noqa: E402
from tools.diagnose_scheme51_pair import (  # noqa: E402
    StrictValidationDataset, collate_pairs_with_exclusions, localize_config)
from tools.run_scheme51 import ctc_state, inference_architecture  # noqa: E402
from tools.scheme51_error_report import normalize_text  # noqa: E402

CHECKPOINTS = {
    "gtc": OCR / "output/rec/bctr_scheme51_bs64/svtrv2_tiny_51_gtc_seed48/epoch_100.pth",
    "b": OCR / "output/rec/bctr_scheme51_ab_bs64/svtrv2_tiny_51_ucr_b_mixedfreq_seed48/epoch_100.pth",
}
DICTIONARY = OCR / "tools/utils/ppocr_keys_v1.txt"
OUT = REPO / "ly/exp/A0-01_ctc_dumps/dumps"


def sha12(path):
    sha = hashlib.sha256()
    with open(path, "rb") as stream:
        while block := stream.read(1 << 23):
            sha.update(block)
    return sha.hexdigest()[:12]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(CHECKPOINTS))
    parser.add_argument("--checkpoint", type=Path, help="any Scheme 5.1 checkpoint (needs --tag)")
    parser.add_argument("--tag", help="file-name prefix for --checkpoint")
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument("--domain", choices=("scene", "web", "document", "hwdb"), required=True)  # hwdb: C-16 handwritten lines
    parser.add_argument("--dataset-root", type=Path, default=REPO / "ly/data/bctr")
    parser.add_argument("--topk", type=int, default=16)
    parser.add_argument("--limit", type=int, default=0, help="debug: only the first N batches")
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()
    if (args.model is None) == (args.checkpoint is None) or (args.checkpoint is not None and not args.tag):
        parser.error("give either --model, or --checkpoint with --tag")
    ckpt_path = CHECKPOINTS[args.model] if args.model else args.checkpoint.resolve()
    tag = args.model or args.tag

    torch.set_num_threads(args.threads)
    if args.device == "cuda":  # full FP32 like the CPU frozen test: no TF32, deterministic kernels
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    saved = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if args.model:
        assert int(saved["epoch"]) == 100
    for part in ("Train", "Eval"):  # C-02 runs train with workers; validate_config insists on 0, eval ignores it
        saved["config"][part]["loader"]["num_workers"] = 0
    cfg = localize_config(saved["config"], args.dataset_root.resolve(), DICTIONARY, args.domain, 64)
    cfg["Eval"]["dataset"]["data_dir_list"] = [
        str(args.dataset_root.resolve() / args.domain / f"{args.domain}_{args.split}")]
    post = build_post_process(cfg["PostProcess"], cfg["Global"])
    arch = inference_architecture(cfg)
    arch["Decoder"]["out_channels"] = post.get_character_num()
    model = build_model(arch)
    model.load_state_dict(ctc_state(saved["state_dict"]), strict=True)
    model = model.to(args.device).eval()

    dataset = StrictValidationDataset(cfg, args.domain, invalid_samples="skip", split=args.split)
    sampler_cfg = copy.deepcopy(cfg["Eval"]["sampler"])
    sampler_cfg.pop("name")
    loader = DataLoader(dataset, batch_sampler=Scheme51RatioSampler(dataset, **sampler_cfg),
                        num_workers=args.workers, collate_fn=collate_pairs_with_exclusions)

    args.out.mkdir(parents=True, exist_ok=True)
    stem = args.out / f"{tag}_{args.split}_{args.domain}"
    idx_chunks, logp_chunks, blank_chunks, lengths = [], [], [], []
    rows, excluded = [], []
    start = time.time()
    with torch.inference_mode():
        for number, (images, labels, info, skipped) in enumerate(loader, 1):
            excluded.extend(skipped)
            if images is not None:
                probs = model(images.to(args.device)).float().cpu()  # (B, T, C) CTC softmax
                decoded = post(probs)
                targets = post.decode(labels)
                logp = probs.clamp_min(1e-30).log()
                top_v, top_i = logp.topk(args.topk, dim=2)
                for b, meta in enumerate(info):
                    idx_chunks.append(top_i[b].to(torch.int16).numpy())
                    logp_chunks.append(top_v[b].to(torch.float16).numpy())
                    blank_chunks.append(logp[b, :, 0].to(torch.float16).numpy())
                    lengths.append(probs.shape[1])
                    target = targets[b][0]
                    rows.append({"sample_id": meta["sample_id"], "target": target,
                                 "target_normalized": normalize_text(target, cfg["Metric"]),
                                 "prediction": decoded[b][0],
                                 "prediction_normalized": normalize_text(decoded[b][0], cfg["Metric"])})
            if number % 100 == 0:
                print(f"{stem.name}: {number}/{len(loader)} batches, {len(rows)} samples, "
                      f"{time.time() - start:.0f}s", flush=True)
            if args.limit and number >= args.limit:
                break
    dataset.close()
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    np.savez(stem.with_suffix(".npz"), topk_idx=np.concatenate(idx_chunks),
             topk_logp=np.concatenate(logp_chunks), blank_logp=np.concatenate(blank_chunks),
             offsets=offsets)
    with open(stem.with_suffix(".jsonl"), "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    meta = {"model": tag, "checkpoint": str(ckpt_path.relative_to(REPO)), "epoch": int(saved["epoch"]),
            "checkpoint_sha256_12": sha12(ckpt_path), "split": args.split,
            "domain": args.domain, "samples": len(rows), "excluded": len(excluded),
            "topk": args.topk, "limit_batches": args.limit, "torch": torch.__version__, "device": args.device,
            "seconds": round(time.time() - start, 1)}
    stem.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
