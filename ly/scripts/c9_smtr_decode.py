"""C-09: greedy SMTR (autoregressive) decoding with step-wise conditional frequency correction.

The GTC models (ours and the official SVTRv2 releases) carry an SMTR branch trained jointly with CTC. Here it is used
for inference. At each step, if the model's own argmax is a real character (ids 1..N) and its probability is below the
gate gamma, the character is re-chosen as argmax_c (z_c - tau * log pi_c) over real characters; EOS / special decisions
are never changed (the analogue of G-LA's "which character, not whether"). The chosen token is fed back as context, so
each (tau, gamma) is a full decoding pass. The decoding loop is a copy of OpenOCR-main's SMTRDecoder.forward_test
(next_mode, ds=True; openrec/modeling/decoders/smtr_decoder.py) with that one change; tau = 0 is checked against the
original method on every first batch. Data pipeline and sample exclusions are those of c4_dump_external.py (GTC seed48
config), so sample ids line up with the CTC dumps. FP32, TF32 off.

    python ly/scripts/c9_smtr_decode.py --config <cfg.yml> --checkpoint <ckpt> --tag off_B --split val --domain scene \
        --out ly/exp/C-09_smtr/preds --device cuda
Protocol: ly/commit.md, entry "C-07 至 C-11 预注册" (C-09).
"""
import argparse
import copy
import json
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parents[2]
OCR = REPO / "OpenOCR-main"
sys.path.insert(0, str(OCR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402
from torch.nn import functional as F  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
import yaml  # noqa: E402

from openrec.modeling import build_model  # noqa: E402
from openrec.postprocess import build_post_process  # noqa: E402
from openrec.scheme51 import Scheme51RatioSampler  # noqa: E402
from tools.diagnose_scheme51_pair import (  # noqa: E402
    StrictValidationDataset, collate_pairs_with_exclusions, localize_config)

from c2_finetune import log_prior  # noqa: E402
from c4_dump_external import DATA_CKPT, DICTIONARY, sha12  # noqa: E402

GRID = [(0.0, 1.01)] + [(t, g) for t in (0.1, 0.2, 0.3, 0.5, 1.0) for g in (0.9, 1.01)]


def cfg_name(tau, gamma):
    return "B0" if tau == 0 else f"t{tau}_g{gamma}"


def smtr_model(config_path, ckpt_path):
    """GTC model with the SMTR branch kept (infer_gtc=True); UCR / sampling auxiliaries dropped."""
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    arch = copy.deepcopy(cfg["Architecture"])
    dec = arch["Decoder"]
    ctc = dec.get("ctc_decoder") or dec.get("decoder")
    saved = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    state = saved["state_dict"] if "state_dict" in saved else saved
    state = {("decoder.ctc_decoder." + k[len("decoder.decoder."):] if k.startswith("decoder.decoder.") else k): v
             for k, v in state.items()}
    drop = ("decoder.ucr.", "decoder.ids_table", "decoder.ids_lengths", "decoder.sampling_")
    state = {k: v for k, v in state.items() if not k.startswith(drop)}
    n_smtr = state["decoder.gtc_decoder.ques1_head.weight"].shape[0] + 3
    n_ctc = state["decoder.ctc_decoder.fc.weight"].shape[0]
    arch["Decoder"] = dict(name="GTCDecoder", infer_gtc=True, detach=False, gtc_decoder=copy.deepcopy(dec["gtc_decoder"]),
                           ctc_decoder=copy.deepcopy(ctc), out_channels=[n_smtr, n_ctc])
    model = build_model(arch)
    model.load_state_dict(state, strict=True)
    return model, saved.get("epoch") if isinstance(saved, dict) else None


@torch.no_grad()
def decode(dec, x, lp_chars, tau, gamma):
    """SMTRDecoder.forward_test (next_mode, ds=True) with the step-wise correction; returns ids (B, max_len)."""
    assert dec.next_mode and dec.ds and not dec.pos2d
    n_chars = lp_chars.numel()
    visual_f = x
    bs = x.shape[0]
    ques_next = dec.next_token.tile([bs, 1, 1, 1]).squeeze(1)
    prompt_next_embed = dec.prompt_next_embed.tile([bs, 1, 1, 1]).squeeze(1)
    prompt_id = torch.full([bs, dec.sub_str_len], dec.bos_next, dtype=torch.long, device=x.device)
    pred_id_list = torch.full([bs, dec.max_len], dec.ignore_index, dtype=torch.long, device=x.device)
    mask_pad = torch.zeros([bs, 1], dtype=torch.float32, device=x.device)
    for j in range(0, dec.max_len - 1):
        prompt_next = torch.concat([prompt_next_embed[:, :1, :], prompt_next_embed[:, 1:, :] + dec.char_embed(prompt_id)], 1)
        mask_next = torch.where(prompt_id == dec.bos_next, float('-inf'), 0)
        mask = torch.concat([mask_pad, mask_next], 1)
        ques_next_i = ques_next
        for layer in dec.cmff_decoder:
            ques_next_i = layer(ques_next_i, prompt_next, visual_f, mask.unsqueeze(1))
        logits = dec.ques1_head(dec.norm_pred(ques_next_i))  # (B, 1, V)
        p = F.softmax(logits, -1)
        pred = p.argmax(-1)
        if tau:
            alt = (logits[..., 1:n_chars + 1] - tau * lp_chars).argmax(-1) + 1
            change = (pred >= 1) & (pred <= n_chars) & (p.max(-1).values < gamma)
            pred = torch.where(change, alt, pred)
        pred_id_list[:, j:j + 1] = pred
        if (pred_id_list == dec.eos).any(dim=-1).all():
            break
        prompt_id = torch.concat([prompt_id[:, 1:], pred], 1)
    return pred_id_list


def to_text(ids, chars):
    out = []
    for i in ids:
        if i == 0:
            break
        if 1 <= i < len(chars):
            out.append(chars[i])
    return "".join(out)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--split", choices=("val", "test"), required=True)
    p.add_argument("--domain", choices=("scene", "web", "document"), required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--dataset-root", dest="dataset_root", type=Path, default=Path("/root/autodl-tmp/bctr"))
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--limit", type=int, default=0, help="debug: only the first N batches")
    p.add_argument("--grid", nargs="+", help="C-17: tau:gamma pairs to decode instead of the C-09 grid (B0 always kept)")
    args = p.parse_args()
    global GRID
    if args.grid:
        GRID = [(0.0, 1.01)] + [tuple(map(float, g.split(":"))) for g in args.grid]

    if args.device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    data_saved = torch.load(DATA_CKPT, map_location="cpu", weights_only=True)
    cfg = localize_config(data_saved["config"], args.dataset_root.resolve(), DICTIONARY, args.domain, args.batch)
    cfg["Eval"]["dataset"]["data_dir_list"] = [
        str(args.dataset_root.resolve() / args.domain / f"{args.domain}_{args.split}")]
    post = build_post_process(cfg["PostProcess"], cfg["Global"])  # CTC decode: targets and the character list
    chars = post.character
    model, epoch = smtr_model(args.config, args.checkpoint)
    model = model.to(args.device).eval()
    dec = model.decoder.gtc_decoder
    lp = log_prior()[0]
    assert lp.numel() == len(chars) == dec.ques1_head.out_features - 1, (lp.numel(), len(chars))
    lp_chars = lp[1:].to(args.device)

    dataset = StrictValidationDataset(cfg, args.domain, invalid_samples="skip", split=args.split)
    sampler_cfg = copy.deepcopy(cfg["Eval"]["sampler"])
    sampler_cfg.pop("name")
    loader = DataLoader(dataset, batch_sampler=Scheme51RatioSampler(dataset, **sampler_cfg),
                        num_workers=args.workers, collate_fn=collate_pairs_with_exclusions)
    args.out.mkdir(parents=True, exist_ok=True)
    stem = args.out / f"{args.tag}_{args.split}_{args.domain}"
    rows, excluded, checked = [], [], 0
    start = time.time()
    with torch.inference_mode():
        for number, (images, labels, info, skipped) in enumerate(loader, 1):
            excluded.extend(skipped)
            if images is None:
                continue
            x = images.to(args.device)
            if model.use_transform:
                x = model.transform(x)
            feats = model.encoder(x)
            vis = feats.flatten(2).transpose(1, 2)
            preds = {cfg_name(t, g): decode(dec, vis, lp_chars, t, g).cpu().tolist() for t, g in GRID}
            if checked < 3:  # tau = 0 must equal OpenOCR's own SMTR forward_test
                ref = dec.forward_test(vis).argmax(-1).cpu().tolist()
                assert [to_text(r, chars) for r in ref] == [to_text(r, chars) for r in preds["B0"]], "B0 != forward_test"
                checked += 1
            targets = post.decode(labels)
            for b, meta in enumerate(info):
                rows.append({"sample_id": meta["sample_id"], "target": targets[b][0],
                             "preds": {k: to_text(v[b], chars) for k, v in preds.items()}})
            if number % 50 == 0:
                print(f"{stem.name}: {number}/{len(loader)} batches, {len(rows)} samples, "
                      f"{time.time() - start:.0f}s", flush=True)
            if args.limit and number >= args.limit:
                break
    dataset.close()
    with open(stem.with_suffix(".jsonl"), "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    meta = {"model": args.tag, "config": str(args.config), "checkpoint": str(args.checkpoint),
            "checkpoint_sha256_12": sha12(args.checkpoint), "epoch": epoch, "split": args.split, "domain": args.domain,
            "samples": len(rows), "excluded": len(excluded), "grid": [cfg_name(t, g) for t, g in GRID],
            "b0_checked_batches": checked, "batch": args.batch, "limit": args.limit, "torch": torch.__version__,
            "seconds": round(time.time() - start, 1)}
    stem.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
