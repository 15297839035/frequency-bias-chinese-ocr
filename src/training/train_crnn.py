"""C-05: train a CRNN-style CTC recogniser (ResNet_ASTER CNN + 2-layer BiLSTM + linear CTC head) on BCTR from scratch.

A different architecture family from SVTRv2, for external validation of the C-03 / C-04 findings. Data pipeline,
augmentation, sampler, dictionary (ppocr_keys_v1) and accumulation are those of the GTC seed48 run (its saved
config.yml, relocated); only Architecture / Loss / label encoding / optimiser / schedule differ. Uses OpenOCR's
Trainer read-only, like src/training/finetune_bctr_la.py.

    python src/training/train_crnn.py train --seed 1 --out results/C-05_crnn/crnn_s1
    python src/training/train_crnn.py train --seed 1 --out /tmp/x --smoke-steps 300
Protocol: docs/experiments.md, entry "C-05 预注册".
"""
import argparse
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
OCR = REPO / "OpenOCR-main"
sys.path.insert(0, str(OCR))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from training.finetune_bctr_la import CONFIG, FirstBatches  # noqa: E402

DICT = "ppocr_keys_v1.txt"


def build_cfg(args):
    from tools.engine.config import Config
    from tools.run_scheme51 import relocate_eval_config
    cfg = Config(str(CONFIG))
    relocate_eval_config(cfg, args.dataset_root)
    c = cfg.cfg
    out = Path(args.out).resolve()
    g = c["Global"]
    assert g["character_dict_path"].endswith(DICT)
    g.update(output_dir=out.as_posix(), save_res_path=(out / "predicts.txt").as_posix(), epoch_num=args.epochs,
             seed=args.seed, pretrained_model=None, checkpoints=None, save_epoch_step=[args.epochs - 1, 1],
             eval_batch_step=[0, 10 ** 9], use_tensorboard=False, grad_clip_val=1.0)
    c["Optimizer"] = {"name": "Adam", "lr": args.lr, "weight_decay": 0.0, "filter_bias_and_bn": False}
    c["LRScheduler"] = {"name": "OneCycleLR", "warmup_epoch": args.warmup_epoch, "cycle_momentum": False}
    c["Architecture"] = {"model_type": "rec", "algorithm": "CRNN", "in_channels": 3, "Transform": None,
                         "Encoder": {"name": "ResNet_ASTER"}, "Decoder": {"name": "CTCDecoder"}}
    c["Loss"] = {"name": "CTCLoss", "zero_infinity": True}
    tr = c["Train"]["dataset"]["transforms"]
    gtc = next(i for i, op in enumerate(tr) if "GTCLabelEncode" in op)
    old = tr[gtc]["GTCLabelEncode"]
    tr[gtc] = {"CTCLabelEncode": {"character_dict_path": old["character_dict_path"], "use_space_char": False,
                                  "max_text_length": old["max_text_length"]}}
    keep = next(i for i, op in enumerate(tr) if "KeepKeys" in op)
    tr[keep] = {"KeepKeys": {"keep_keys": ["image", "label", "length"]}}
    # ResNet_ASTER pools height 32 -> 1, so short images keep height 32 too (SVTRv2's default base_shape grows them
    # to 64/48/40); width stays 32 x aspect-ratio bucket. Same setting for training and evaluation.
    for part in ("Train", "Eval"):
        c[part]["dataset"]["base_shape"] = [[32, 32], [64, 32], [96, 32], [128, 32]]
    c["Train"]["loader"]["num_workers"] = args.workers
    c["Eval"]["loader"]["num_workers"] = min(args.workers, 4)
    c["C05"] = {"model": "ResNet_ASTER + BiLSTM(2x256) + CTC", "epochs": args.epochs, "lr": args.lr,
                "warmup_epoch": args.warmup_epoch, "seed": args.seed, "data_pipeline_from": str(CONFIG.relative_to(REPO))}
    return cfg


def train(args):
    from openrec.scheme51 import register_scheme51
    from tools.engine.trainer import Trainer
    from tools.run_scheme51 import data_preflight
    register_scheme51()
    out = Path(args.out)
    if out.exists() and any(out.iterdir()) and not args.smoke_steps and not args.resume:
        raise FileExistsError(f"{out} is not empty")
    cfg = build_cfg(args)
    if args.resume:  # M-10 (2026-10-13): continue an interrupted run from its last whole epoch
        ckpt = out.resolve() / "latest.pth"
        cfg.cfg["Global"]["checkpoints"] = ckpt.as_posix()
        epoch = torch.load(ckpt, map_location="cpu", weights_only=False)["epoch"]
        with open(out / "resume_log.json", "a") as f:
            f.write(json.dumps({"resumed_from": ckpt.name, "checkpoint_epoch": epoch, "time": __import__("time").ctime()}) + "\n")
    counts = data_preflight(cfg.cfg)
    trainer = Trainer(cfg, mode="train_eval", task="rec")
    if args.smoke_steps:
        trainer.train_dataloader = FirstBatches(trainer.train_dataloader, args.smoke_steps)
        trainer.valid_dataloader = None
    if not args.resume:
        (Path(cfg.cfg["Global"]["output_dir"]) / "c05_run.json").write_text(json.dumps(
        {"args": vars(args), "split_counts": counts, "torch": torch.__version__,
         "gpu": torch.cuda.get_device_name(0),
         "params_M": sum(p.numel() for p in trainer.model.parameters()) / 1e6}, indent=2))
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("train")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--warmup-epoch", dest="warmup_epoch", type=float, default=1.5)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--dataset-root", dest="dataset_root", default="/root/autodl-tmp/bctr")
    p.add_argument("--smoke-steps", dest="smoke_steps", type=int, default=0)
    p.add_argument("--resume", action="store_true", help="continue from <out>/latest.pth (M-10)")
    args = parser.parse_args()
    {"train": train}[args.cmd](args)
