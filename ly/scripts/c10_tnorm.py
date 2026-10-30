"""C-10: post-hoc tau-normalisation (Kang et al., ICLR 2020) of the HWDB ce classifiers, under the C-01 metrics.

Logits f . (W_c / ||W_c||^t), bias dropped as in Kang et al.; t on a grid in [0, 2]. For each metric of C-01
(bal / tr / nat ...), t is selected on val (max; ties -> smaller t) and the selected classifier is read out once on
test. Uses a1_train_hccr (model / data) and c1_reeval_hwdb (metrics) unchanged.
Protocol: ly/commit.md, entry "C-07 至 C-11 预注册" (C-10).

    python ly/scripts/c10_tnorm.py --data ly/data/hwdb --split lt100 --runs ly/exp/A1-02_hwdb_lt/ce_s0 ... \
        --out ly/exp/C-10_lt_baselines/tnorm_lt100.json
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from a1_train_hccr import Split, build_model, to_input  # noqa: E402
from c1_reeval_hwdb import EXP, METRICS, per_class_acc, scores, target_weights  # noqa: E402

T_GRID = np.round(np.arange(0.0, 2.0001, 0.1), 2)


@torch.no_grad()
def features(model, split, device):
    feats, labels = [], []
    store = {}
    hook = model.avgpool.register_forward_hook(lambda m, i, o: store.__setitem__("f", torch.flatten(o, 1)))
    for x, y in split.batches(1024, False):
        logits = model(to_input(x, device))
        feats.append(store["f"].float())
        labels.append(torch.as_tensor(y.astype(np.int64)))
        del logits
    hook.remove()
    return torch.cat(feats), torch.cat(labels).numpy()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True)
    p.add_argument("--split", required=True, help="training split of the runs (lt100 | nat)")
    p.add_argument("--runs", nargs="+", required=True, type=Path)
    p.add_argument("--weights", default=str(EXP / "weights.npz"))
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    data = Path(args.data)
    classes = (data / "classes.txt").read_text(encoding="utf-8")
    tr = Split(data, "train", np.load(data / f"train_{args.split}_indices.npy"))
    lt_counts = np.bincount(tr.labels[tr.idx], minlength=len(classes)).astype(np.float64)
    tw = target_weights(np.load(args.weights), lt_counts)
    splits = {"val": Split(data, "val"), "test": Split(data, "test")}
    out = {"t_grid": T_GRID.tolist(), "split": args.split, "runs": {}}
    for run in args.runs:
        model = build_model(len(classes))
        model.load_state_dict(torch.load(run / "model.pt", map_location="cpu"))
        model = model.to(device).eval()
        W = model.fc.weight.detach().float()
        b = model.fc.bias.detach().float()
        norms = W.norm(dim=1, keepdim=True)
        res = {}
        for name, split in splits.items():
            f, labels = features(model, split, device)
            preds = []
            for t in list(T_GRID) + ["default"]:
                Wt = W if t == "default" else W / norms ** float(t)
                pr = []
                for s in range(0, len(f), 65536):
                    z = f[s:s + 65536] @ Wt.T + (b if t == "default" else 0)
                    pr.append(z.argmax(1).cpu())
                preds.append(torch.cat(pr).numpy())
            acc = per_class_acc(np.stack(preds, 1), labels, len(classes))
            res[name] = scores(acc, tw, lt_counts)
        r = {"default": {m: float(res["test"][m][-1]) for m in res["test"]}, "select": {}, "test_frozen": {}}
        for m in METRICS:
            v = np.asarray(res["val"][m][:-1])
            k = int(np.flatnonzero(v == v.max())[0])
            r["select"][m] = {"t": float(T_GRID[k]), "val": float(v[k])}
            r["test_frozen"][f"t*({m})"] = {mm: float(res["test"][mm][k]) for mm in res["test"]}
        r["curve_test"] = {m: [float(x) for x in res["test"][m][:-1]] for m in res["test"]}
        out["runs"][run.name] = r
        print(run.name, {m: r["select"][m]["t"] for m in ("bal", "tr", "nat")},
              {k: round(v["bal"], 2) for k, v in r["test_frozen"].items() if k == "t*(bal)"},
              {k: round(v["nat"], 2) for k, v in r["test_frozen"].items() if k == "t*(nat)"}, flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
