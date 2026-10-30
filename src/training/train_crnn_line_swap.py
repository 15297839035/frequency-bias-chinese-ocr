"""C-13: train the C-05 CRNN from scratch with a frequency-flip sampler on BCTR text lines.

Everything (architecture, data pipeline, augmentation, optimiser, schedule, 20 epochs, last epoch kept) is C-05's
`training/train_crnn.py` (`build_cfg`); the only change is the training sampler: C13WeightedRatioSampler (line_swap_sampler.py) draws the
members of every aspect-ratio bucket with the per-line weights written by `line_swap.py weights`, which swap
(arm "rev") or equalise (arm "bal") the training exposure of the selected confusable pairs. The C-05 runs crnn_s1 /
crnn_s2 are the natural arm.

    python src/training/train_crnn_line_swap.py train --arm rev --seed 1 --weights results/C-13_crnn_flip/weights_rev.npz \
        --out results/C-13_crnn_flip/rev_s1
    python src/training/train_crnn_line_swap.py train ... --out /tmp/x --smoke-steps 200
Protocol: docs/experiments.md, entry "C-13 至 C-17 预注册".
"""
import argparse
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
OCR = REPO / "OpenOCR-main"
sys.path.insert(0, str(OCR))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from training.finetune_bctr_la import FirstBatches  # noqa: E402
from inference.dump_external_models import sha12  # noqa: E402
from training.train_crnn import build_cfg  # noqa: E402


def train(args):
    from openrec.scheme51 import register_scheme51
    import tools.data as data
    from tools.engine.trainer import Trainer
    from tools.run_scheme51 import data_preflight
    register_scheme51()
    data.SAMPLER_MODULES["C13WeightedRatioSampler"] = "training.line_swap_sampler"
    out = Path(args.out)
    if out.exists() and any(out.iterdir()) and not args.smoke_steps and not args.resume:
        raise FileExistsError(f"{out} is not empty")
    weights = Path(args.weights).resolve()
    cfg = build_cfg(args)
    sampler = cfg.cfg["Train"]["sampler"]
    if sampler["name"] != "Scheme51RatioSampler":
        raise ValueError(f"unexpected C-05 training sampler {sampler['name']}")
    sampler.update(name="C13WeightedRatioSampler", weights_path=weights.as_posix())
    if args.resume:  # same mechanism as C-05 / C-11 (M-10): continue from the last whole epoch
        ckpt = out.resolve() / "latest.pth"
        cfg.cfg["Global"]["checkpoints"] = ckpt.as_posix()
        epoch = torch.load(ckpt, map_location="cpu", weights_only=False)["epoch"]
        with open(out / "resume_log.json", "a") as f:
            f.write(json.dumps({"resumed_from": ckpt.name, "checkpoint_epoch": epoch, "time": __import__("time").ctime()}) + "\n")
    cfg.cfg["C13"] = {"arm": args.arm, "seed": args.seed, "weights": weights.as_posix(),
                      "weights_sha256_12": sha12(weights), "base": "C-05 crnn config",
                      "note": "frequency-flip sampler, see src/training/line_swap_sampler.py"}
    counts = data_preflight(cfg.cfg)
    trainer = Trainer(cfg, mode="train_eval", task="rec")
    bs = trainer.train_dataloader.batch_sampler
    if type(bs).__name__ != "C13WeightedRatioSampler":
        raise RuntimeError(f"training sampler is {type(bs).__name__}")
    w = bs.line_w
    info = {"lines": int(len(w)), "lines_weight_not_1": int((w != 1).sum()), "weight_sum": float(w.sum()),
            "weight_min": float(w.min()), "weight_max": float(w.max()), "batches_per_epoch": len(bs)}
    print("C13 sampler", json.dumps(info), flush=True)
    if args.smoke_steps:
        trainer.train_dataloader = FirstBatches(trainer.train_dataloader, args.smoke_steps)
        trainer.valid_dataloader = None
    if not args.resume:
        (Path(cfg.cfg["Global"]["output_dir"]) / "c13_run.json").write_text(json.dumps(
            {"args": vars(args), "split_counts": counts, "sampler": info, "torch": torch.__version__,
             "numpy": np.__version__, "gpu": torch.cuda.get_device_name(0),
             "params_M": sum(p.numel() for p in trainer.model.parameters()) / 1e6}, indent=2))
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("train")
    p.add_argument("--arm", choices=("rev", "bal"), required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--weights", required=True)
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
