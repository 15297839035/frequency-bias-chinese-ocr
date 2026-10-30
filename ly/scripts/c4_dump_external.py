"""C-04: dump CTC top-K posteriors of an external (official) Chinese SVTRv2 model on BCTR, in A0's format.

Data pipeline, preprocessing, sample exclusion and text normalisation come from the GTC seed48 config exactly as
in a0_dump_ctc.py (so samples and targets line up with the A0 / C-03 dumps); only the model is replaced by the
external one, built from its own config.yml and reduced to the CTC-only inference graph. The checkpoint is a
download, so it is loaded with weights_only=True.

    python ly/scripts/c4_dump_external.py --config <dir>/config.yml --checkpoint <dir>/best.pth --tag off_b \
        --split test --domain scene --out ly/exp/C-04_external/dumps --device cuda
Protocol: ly/commit.md, entry "C-04 预注册".
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
import yaml  # noqa: E402

from openrec.modeling import build_model  # noqa: E402
from openrec.postprocess import build_post_process  # noqa: E402
from openrec.scheme51 import Scheme51RatioSampler  # noqa: E402
from tools.diagnose_scheme51_pair import (  # noqa: E402
    StrictValidationDataset, collate_pairs_with_exclusions, localize_config)
from tools.run_scheme51 import ctc_state, inference_architecture  # noqa: E402
from tools.scheme51_error_report import normalize_text  # noqa: E402

DATA_CKPT = OCR / "output/rec/bctr_scheme51_bs64/svtrv2_tiny_51_gtc_seed48/epoch_100.pth"  # data pipeline only
DICTIONARY = OCR / "tools/utils/ppocr_keys_v1.txt"


def sha12(path):
    sha = hashlib.sha256()
    with open(path, "rb") as stream:
        while block := stream.read(1 << 23):
            sha.update(block)
    return sha.hexdigest()[:12]


def external_model(config_path, ckpt_path, num_classes, trusted=False):
    ext = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    g = ext["Global"]
    assert Path(g["character_dict_path"]).name == "ppocr_keys_v1.txt" and not g.get("use_space_char"), \
        "external model must use the same dictionary without a space class"
    dec = ext["Architecture"]["Decoder"]
    assert dec["name"] in ("GTCDecoder", "CTCDecoder"), dec["name"]
    if dec["name"] == "CTCDecoder":  # C-05 CRNN: plain CTC model, no auxiliary branch to strip
        arch = copy.deepcopy(ext["Architecture"])
        arch["Decoder"]["out_channels"] = num_classes
        model = build_model(arch)
        saved = torch.load(ckpt_path, map_location="cpu", weights_only=not trusted)
        model.load_state_dict(saved["state_dict"] if "state_dict" in saved else saved, strict=True)
        return model, saved.get("epoch") if isinstance(saved, dict) else None
    if "ctc_decoder" not in dec and "decoder" in dec:  # older OpenOCR key name (official T / S releases)
        dec["ctc_decoder"] = dec.pop("decoder")
    arch = inference_architecture(ext)
    arch["Decoder"]["out_channels"] = num_classes
    model = build_model(arch)
    saved = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    state = saved["state_dict"] if "state_dict" in saved else saved
    state = {("decoder.ctc_decoder." + k[len("decoder.decoder."):] if k.startswith("decoder.decoder.") else k): v
             for k, v in state.items()}
    model.load_state_dict(ctc_state(state), strict=True)
    return model, saved.get("epoch") if isinstance(saved, dict) else None


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--split", choices=("val", "test"), required=True)
    p.add_argument("--domain", choices=("scene", "web", "document", "hwdb"), required=True)  # hwdb: C-16 handwritten lines
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--dataset-root", dest="dataset_root", type=Path, default=Path("/root/autodl-tmp/bctr"))
    p.add_argument("--topk", type=int, default=16)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--trusted", action="store_true", help="own checkpoint (C-05): allow full unpickling; never for downloads")
    args = p.parse_args()

    torch.set_num_threads(args.threads)
    if args.device == "cuda":  # full FP32 as in the frozen test (see C-03 TF32 note)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    data_saved = torch.load(DATA_CKPT, map_location="cpu", weights_only=True)
    cfg = localize_config(data_saved["config"], args.dataset_root.resolve(), DICTIONARY, args.domain, 64)
    cfg["Eval"]["dataset"]["data_dir_list"] = [
        str(args.dataset_root.resolve() / args.domain / f"{args.domain}_{args.split}")]
    ext_eval = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")).get("Eval", {}).get("dataset", {})
    if "base_shape" in ext_eval:  # C-05 CRNN: fixed height 32 for every aspect-ratio bucket
        cfg["Eval"]["dataset"]["base_shape"] = ext_eval["base_shape"]
    post = build_post_process(cfg["PostProcess"], cfg["Global"])
    model, epoch = external_model(args.config, args.checkpoint, post.get_character_num(), trusted=args.trusted)
    model = model.to(args.device).eval()

    dataset = StrictValidationDataset(cfg, args.domain, invalid_samples="skip", split=args.split)
    sampler_cfg = copy.deepcopy(cfg["Eval"]["sampler"])
    sampler_cfg.pop("name")
    loader = DataLoader(dataset, batch_sampler=Scheme51RatioSampler(dataset, **sampler_cfg),
                        num_workers=args.workers, collate_fn=collate_pairs_with_exclusions)
    args.out.mkdir(parents=True, exist_ok=True)
    stem = args.out / f"{args.tag}_{args.split}_{args.domain}"
    idx_chunks, logp_chunks, blank_chunks, lengths, rows, excluded = [], [], [], [], [], []
    start = time.time()
    with torch.inference_mode():
        for images, labels, info, skipped in loader:
            excluded.extend(skipped)
            if images is None:
                continue
            probs = model(images.to(args.device)).float().cpu()
            decoded, targets = post(probs), post.decode(labels)
            logp = probs.clamp_min(1e-30).log()
            top_v, top_i = logp.topk(args.topk, dim=2)
            for b, meta in enumerate(info):
                idx_chunks.append(top_i[b].to(torch.int16).numpy())
                logp_chunks.append(top_v[b].to(torch.float16).numpy())
                blank_chunks.append(logp[b, :, 0].to(torch.float16).numpy())
                lengths.append(probs.shape[1])
                rows.append({"sample_id": meta["sample_id"], "target": targets[b][0],
                             "target_normalized": normalize_text(targets[b][0], cfg["Metric"]),
                             "prediction": decoded[b][0],
                             "prediction_normalized": normalize_text(decoded[b][0], cfg["Metric"])})
    dataset.close()
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    np.savez(stem.with_suffix(".npz"), topk_idx=np.concatenate(idx_chunks), topk_logp=np.concatenate(logp_chunks),
             blank_logp=np.concatenate(blank_chunks), offsets=offsets)
    with open(stem.with_suffix(".jsonl"), "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    meta = {"model": args.tag, "config": str(args.config), "checkpoint": str(args.checkpoint),
            "checkpoint_sha256_12": sha12(args.checkpoint), "epoch": epoch, "split": args.split, "domain": args.domain,
            "samples": len(rows), "excluded": len(excluded), "topk": args.topk, "torch": torch.__version__,
            "seconds": round(time.time() - start, 1)}
    stem.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
