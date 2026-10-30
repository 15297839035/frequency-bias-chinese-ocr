"""C-11: train the C-05 CRNN from scratch with training-side conditional logit adjustment on the CTC logits.

Everything (architecture, data pipeline, optimiser, schedule, 20 epochs, last epoch kept) is C-05's
`c5_train_crnn.build_cfg`; the only change is C-02's conditional LA wrapped around the CTC loss: per frame the blank
logit is untouched and the non-blank logits move by tau*log(pi_c) with their total softmax mass kept, so an LA-trained
model with post-hoc G-LA tau_post has effective strength tau_eff = tau_train + tau_post. tau_train = 0 would be C-05.

    python ly/scripts/c11_train_crnn_la.py train --tau 0.25 --seed 1 --out ly/exp/C-11_crnn_la/la025_s1
    python ly/scripts/c11_train_crnn_la.py train --tau 1 --seed 1 --out /tmp/x --smoke-steps 200
Protocol: ly/commit.md, entry "C-07 至 C-11 预注册".
"""
import argparse
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
OCR = REPO / "OpenOCR-main"
sys.path.insert(0, str(OCR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

from c2_finetune import FirstBatches, conditional_la, log_prior  # noqa: E402
from c5_train_crnn import build_cfg  # noqa: E402


class PriorAdjustedCTC(torch.nn.Module):
    """CTCLoss on conditionally logit-adjusted CTC logits (ids 1..N shifted, blank kept)."""

    def __init__(self, base, lp, tau):
        super().__init__()
        self.base, self.tau = base, tau
        self.register_buffer("lp", lp[1:].clone())

    def set_progress(self, progress):
        if hasattr(self.base, "set_progress"):
            self.base.set_progress(progress)

    def forward(self, predicts, batch):
        if self.tau:
            predicts = conditional_la(predicts, self.lp, self.tau, 1, predicts.shape[-1])
        return self.base(predicts, batch)


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
    cfg.cfg["C11"] = {"tau_train": args.tau, "seed": args.seed, "base": "C-05 crnn config",
                      "note": "conditional LA on CTC logits, see ly/scripts/c11_train_crnn_la.py"}
    counts = data_preflight(cfg.cfg)
    trainer = Trainer(cfg, mode="train_eval", task="rec")
    lp, _ = log_prior()
    assert trainer.model.decoder.fc.out_features == lp.numel(), (trainer.model.decoder.fc.out_features, lp.numel())
    trainer.loss_class = PriorAdjustedCTC(trainer.loss_class, lp, args.tau).to(trainer.device)
    if args.smoke_steps:
        trainer.train_dataloader = FirstBatches(trainer.train_dataloader, args.smoke_steps)
        trainer.valid_dataloader = None
    if not args.resume:
        (Path(cfg.cfg["Global"]["output_dir"]) / "c11_run.json").write_text(json.dumps(
        {"args": vars(args), "split_counts": counts, "torch": torch.__version__,
         "gpu": torch.cuda.get_device_name(0),
         "params_M": sum(p.numel() for p in trainer.model.parameters()) / 1e6}, indent=2))
    trainer.train()


def selftest(args):
    """CPU: the wrapper equals CTCLoss at tau = 0 and keeps the blank mass at tau > 0."""
    from openrec.losses.ctc_loss import CTCLoss
    lp, _ = log_prior()
    torch.manual_seed(0)
    z = torch.randn(3, 20, lp.numel())
    batch = [None, torch.randint(1, lp.numel(), (3, 5)), torch.tensor([5, 4, 3])]
    base = CTCLoss(zero_infinity=True)
    assert torch.allclose(PriorAdjustedCTC(base, lp, 0.0)(z, batch)["loss"], base(z, batch)["loss"])
    for tau in (0.25, 1.0):
        q = conditional_la(z, lp[1:], tau, 1, z.shape[-1]).softmax(-1)
        torch.testing.assert_close(q[..., 0], z.softmax(-1)[..., 0], rtol=1e-4, atol=1e-6)
        assert torch.isfinite(PriorAdjustedCTC(base, lp, tau)(z, batch)["loss"])
    print("SELFTEST OK")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("train")
    p.add_argument("--tau", type=float, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--warmup-epoch", dest="warmup_epoch", type=float, default=1.5)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--dataset-root", dest="dataset_root", default="/root/autodl-tmp/bctr")
    p.add_argument("--smoke-steps", dest="smoke_steps", type=int, default=0)
    p.add_argument("--resume", action="store_true", help="continue from <out>/latest.pth (M-10)")
    sub.add_parser("selftest")
    args = parser.parse_args()
    {"train": train, "selftest": selftest}[args.cmd](args)
