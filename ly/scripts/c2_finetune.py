"""C-02: continue GTC seed48 (epoch 100) on BCTR with or without training-side prior correction.

Uses OpenOCR-main's Trainer, model, data pipeline and Scheme51Loss read-only; the only change is a
wrapper that applies *conditional* logit adjustment during training:

    CTC  : per frame, blank logit untouched; non-blank logits z_c -> z_c + tau*log(pi_c) + b,
           b chosen so that the total non-blank mass is unchanged (only "which character" moves).
    SMTR : the same on real-character logits (ids 1..N) of ques1_head; EOS / special tokens untouched.

pi is the add-one BCTR training character prior (the one used by A0's G-LA), so an LA-trained model
with post-hoc G-LA tau_post has effective strength tau_eff = tau_train + tau_post. tau_train = 0 is
exactly the original loss (the control arm).

    python ly/scripts/c2_finetune.py selftest
    python ly/scripts/c2_finetune.py train --tau 1.0 --seed 1 --epochs 10 --out ly/exp/C-02_bctr_la/la1_s1
    python ly/scripts/c2_finetune.py train ... --smoke-steps 200     # throughput check, no eval/checkpoint use
Protocol: ly/commit.md, entry "C-02 预注册".
"""
import argparse
import json
import math
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
OCR = REPO / "OpenOCR-main"
sys.path.insert(0, str(OCR))

import numpy as np  # noqa: E402
import torch  # noqa: E402

CKPT = OCR / "output/rec/bctr_scheme51_bs64/svtrv2_tiny_51_gtc_seed48/epoch_100.pth"
CONFIG = CKPT.parent / "config.yml"
FREQ = OCR / "output/rec/bctr_scheme51_ab_bs64/train_character_frequency.json"


def log_prior():
    """Add-one prior over CTC ids 1..N (index 0 = blank kept at 0, never used)."""
    freq = json.loads(FREQ.read_text(encoding="utf-8"))
    counts = np.asarray(freq["counts"], dtype=np.float64)
    lp = np.log((counts + 1.0) / (counts + 1.0)[1:].sum())
    lp[0] = 0.0
    return torch.as_tensor(lp, dtype=torch.float32), freq["characters"]


def conditional_la(z, lp, tau, start, stop):
    """Shift logits z[..., start:stop] by tau*lp, keeping their total softmax mass unchanged."""
    z = z.float()
    sub = z[..., start:stop]
    adj = sub + tau * lp
    adj = adj - adj.logsumexp(-1, keepdim=True) + sub.logsumexp(-1, keepdim=True)
    return torch.cat([z[..., :start], adj, z[..., stop:]], -1)


class PriorAdjustedLoss(torch.nn.Module):
    """Scheme51Loss with conditional LA on the CTC logits (SMTR is handled by a head hook)."""

    def __init__(self, base, lp, tau):
        super().__init__()
        self.base, self.tau = base, tau
        self.register_buffer("lp", lp[1:].clone())

    def set_progress(self, progress):
        if hasattr(self.base, "set_progress"):
            self.base.set_progress(progress)

    def forward(self, predicts, batch):
        if self.tau:
            predicts = dict(predicts)
            z = predicts["ctc_pred"]
            predicts["ctc_pred"] = conditional_la(z, self.lp, self.tau, 1, z.shape[-1])
        return self.base(predicts, batch)


def smtr_hook(lp, tau):
    n = lp.numel() - 1  # real characters, ids 1..n

    def hook(module, inputs, output):
        if not module.training or not tau:
            return output
        return conditional_la(output, lp[1:].to(output.device), tau, 1, 1 + n)
    return hook


def build_cfg(args):
    from tools.engine.config import Config
    from tools.run_scheme51 import relocate_eval_config
    cfg = Config(str(CONFIG))
    relocate_eval_config(cfg, args.dataset_root)
    c = cfg.cfg
    out = Path(args.out).resolve()
    g = c["Global"]
    g.update(output_dir=out.as_posix(), save_res_path=(out / "predicts.txt").as_posix(), epoch_num=args.epochs,
             seed=args.seed, pretrained_model=CKPT.as_posix(), checkpoints=None,
             save_epoch_step=[args.epochs - 1, 1], eval_batch_step=[0, 10 ** 9], use_tensorboard=False)
    c["Optimizer"]["lr"] = args.lr
    c["LRScheduler"] = {"name": "CosineAnnealingLR", "warmup_epoch": args.warmup_epoch}
    c["Train"]["loader"]["num_workers"] = args.workers
    c["Eval"]["loader"]["num_workers"] = min(args.workers, 4)
    c["C02"] = {"tau_train": args.tau, "init_checkpoint": str(CKPT.relative_to(REPO)), "lr_peak": args.lr,
                "warmup_epoch": args.warmup_epoch, "epochs": args.epochs, "seed": args.seed,
                "prior": str(FREQ.relative_to(REPO)), "note": "conditional LA, see ly/scripts/c2_finetune.py"}
    return cfg


class FirstBatches:
    """Throughput check: the first `steps` batches of a dataloader, no dataset reset."""

    def __init__(self, loader, steps):
        self.loader, self.steps, self.dataset = loader, steps, None

    def __len__(self):
        return self.steps

    def __iter__(self):
        for i, batch in enumerate(self.loader):
            if i >= self.steps:
                break
            yield batch


def train(args):
    from openrec.scheme51 import register_scheme51
    from tools.engine.trainer import Trainer
    from tools.run_scheme51 import data_preflight
    register_scheme51()
    out = Path(args.out)
    if out.exists() and any(out.iterdir()) and not args.smoke_steps:
        raise FileExistsError(f"{out} is not empty")
    cfg = build_cfg(args)
    counts = data_preflight(cfg.cfg)
    trainer = Trainer(cfg, mode="train_eval", task="rec")
    # every parameter and buffer must come from the epoch-100 checkpoint
    saved = torch.load(CKPT, map_location="cpu", weights_only=False)["state_dict"]
    current = trainer.model.state_dict()
    assert set(saved) == set(current), set(saved) ^ set(current)
    assert all(torch.equal(saved[k].cpu(), current[k].cpu()) for k in saved), "weights not loaded"
    lp, _ = log_prior()
    trainer.loss_class = PriorAdjustedLoss(trainer.loss_class, lp, args.tau).to(trainer.device)
    head = trainer.model.decoder.gtc_decoder.ques1_head
    assert head.out_features == lp.numel() + 1, (head.out_features, lp.numel())  # EOS + N chars + 1 special
    head.register_forward_hook(smtr_hook(lp, args.tau))
    if args.smoke_steps:
        trainer.train_dataloader = FirstBatches(trainer.train_dataloader, args.smoke_steps)
        trainer.valid_dataloader = None
    (Path(cfg.cfg["Global"]["output_dir"]) / "c02_run.json").write_text(json.dumps(
        {"args": vars(args), "split_counts": counts, "torch": torch.__version__,
         "gpu": torch.cuda.get_device_name(0)}, indent=2))
    trainer.train()


def selftest(args):
    """Checks on synthetic tensors and on the real dictionaries; no data, no training."""
    torch.manual_seed(0)
    lp, chars = log_prior()
    C = lp.numel()
    z = torch.randn(4, 7, C) * 3
    for tau in (0.0, 0.5, 1.0):
        adj = conditional_la(z, lp[1:], tau, 1, C)
        p, q = z.softmax(-1), adj.softmax(-1)
        torch.testing.assert_close(q[..., 0], p[..., 0], rtol=1e-4, atol=1e-6)  # blank mass kept
        ratio = (q[..., 1:] / p[..., 1:]).log() - tau * lp[1:]
        assert float(ratio.std(-1).max()) < 1e-3, "conditional ratio must be prop. to pi^tau"
        if tau == 0:
            torch.testing.assert_close(adj, z)
    # dictionary alignment: frequency file == CTC ids == SMTR ids
    from openrec.preprocess.ctc_label_encode import CTCLabelEncode
    from openrec.preprocess.smtr_label_encode import SMTRLabelEncode
    dict_path = OCR / "tools/utils/ppocr_keys_v1.txt"
    ctc = CTCLabelEncode(25, str(dict_path), False)
    smtr = SMTRLabelEncode(25, str(dict_path), False)
    assert list(ctc.character) == list(chars), "frequency file order differs from CTC ids"
    probe = "中国的字频偏置龘"
    assert ctc.encode(probe) == smtr.encode(probe), (ctc.encode(probe), smtr.encode(probe))
    assert smtr.dict[smtr.EOS] == 0 and len(smtr.character) == C + 4, (smtr.dict[smtr.EOS], len(smtr.character))
    print("SELFTEST OK", {"classes_ctc": C, "classes_smtr": len(smtr.character),
                         "smtr_specials": {k: smtr.dict[k] for k in (smtr.EOS, smtr.BOS, smtr.IN_F, smtr.IN_B, smtr.PAD)
                                           if k in smtr.dict}})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("train")
    p.add_argument("--tau", type=float, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--epochs", type=int, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--lr", type=float, default=6.5e-5, help="peak LR (original OneCycle peak 6.5e-4)")
    p.add_argument("--warmup-epoch", dest="warmup_epoch", type=float, default=0.3)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--dataset-root", dest="dataset_root", default="/root/autodl-tmp/bctr")
    p.add_argument("--smoke-steps", dest="smoke_steps", type=int, default=0)
    sub.add_parser("selftest")
    args = parser.parse_args()
    {"train": train, "selftest": selftest}[args.cmd](args)
