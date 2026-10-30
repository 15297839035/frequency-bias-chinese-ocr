"""C-01: balanced vs natural-distribution evaluation of the A1-02 / A1-03 HWDB-LT models (no training).

Post-hoc prior correction s = logits - tau_post * log(pi), pi = add-one LT train prior (the one `la` trains with).
Every model is described by the effective strength tau_eff = tau_train + tau_post (tau_train = 1 for la*, else 0),
on one grid for all models. PR arms plug a target prior in (label-shift Bayes rule):
s = logits - (1 - tau_train) * log(pi) + log(q~).

    python ly/scripts/c1_reeval_hwdb.py weights      # local: BCTR frequencies -> ly/exp/C-01_bal_vs_nat/weights.npz
    python ly/scripts/c1_reeval_hwdb.py dump --data ly/data/hwdb --runs ly/exp/A1-02_hwdb_lt/*_s? ly/exp/A1-03_la_plus/*_s? \
        --out ly/exp/C-01_bal_vs_nat/dumps                                   # GPU
    python ly/scripts/c1_reeval_hwdb.py analyze --dumps ly/exp/C-01_bal_vs_nat/dumps --out ly/exp/C-01_bal_vs_nat
    python ly/scripts/c1_reeval_hwdb.py smoke        # CPU, synthetic data, whole pipeline
Protocol: ly/commit.md, entry "C-01 预注册".
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

REPO = Path(__file__).resolve().parents[2]
EXP = REPO / "ly/exp/C-01_bal_vs_nat"
CLASSES = REPO / "ly/exp/A1-00_prep/gb2312_l1_classes.txt"
FREQ = REPO / "OpenOCR-main/output/rec/bctr_scheme51_ab_bs64/train_character_frequency.json"
BCTR = REPO / "ly/data/bctr"
TAUS = np.round(np.arange(-2.0, 2.0001, 0.05), 2)          # 81 values of tau_eff
PR_TARGETS = ("nat", "nat_doc", "nat_scene", "nat_web")
METRICS = ("bal", "tr") + PR_TARGETS
GROUPS = (("many", lambda n: n > 100), ("medium", lambda n: (n >= 20) & (n <= 100)), ("few", lambda n: n < 20))


# ----------------------------------------------------------------------------- weights (local)
def weights(args):
    import lmdb
    classes = CLASSES.read_text(encoding="utf-8").strip()
    index = {c: i for i, c in enumerate(classes)}
    freq = json.loads(FREQ.read_text(encoding="utf-8"))
    out = {"nat": np.array([dict(zip(freq["characters"], freq["counts"])).get(c, 0) for c in classes], np.float64)}
    for dom in ("document", "scene", "web"):
        counts = np.zeros(len(classes))
        env = lmdb.open(str(BCTR / dom / f"{dom}_val"), readonly=True, lock=False)
        with env.begin() as txn:
            for i in range(1, int(txn.get(b"num-samples")) + 1):
                for ch in txn.get(b"label-%09d" % i).decode("utf-8"):
                    if ch in index:
                        counts[index[ch]] += 1
        out[f"nat_{dom[:3] if dom == 'document' else dom}"] = counts
    EXP.mkdir(parents=True, exist_ok=True)
    np.savez(EXP / "weights.npz", classes=np.array(list(classes)), **out)
    summary = {}
    for k, v in out.items():
        q = np.sort(v)[::-1] / v.sum()
        summary[k] = {"total": int(v.sum()), "zero_classes": int((v == 0).sum()),
                      "top100_share": float(q[:100].sum()), "top1000_share": float(q[:1000].sum())}
    (EXP / "weights_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def target_weights(w, lt_counts):
    """Normalised class weights per metric; 'tr' is the LT train prior itself."""
    out = {"bal": np.full(len(lt_counts), 1 / len(lt_counts)), "tr": (lt_counts + 1) / (lt_counts + 1).sum()}
    for k in PR_TARGETS:
        out[k] = w[k] / w[k].sum()
    return out


# ----------------------------------------------------------------------------- dump (GPU)
def dump(args):
    import torch
    from a1_train_hccr import Split, build_model, to_input
    device = torch.device(args.device)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    data = Path(args.data)
    classes = (data / "classes.txt").read_text(encoding="utf-8")
    w = np.load(args.weights)
    assert "".join(w["classes"]) == classes, "class list differs from weights.npz"
    lt = Split(data, "train", np.load(data / f"train_{args.split}_indices.npy"))  # C-01n: --split nat
    lt_counts = np.bincount(lt.labels[lt.idx], minlength=len(classes)).astype(np.float64)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    np.save(out_root / "lt_counts.npy", lt_counts)
    log_pi = torch.log(torch.as_tensor((lt_counts + 1) / (lt_counts + 1).sum(), dtype=torch.float32, device=device))
    log_q = torch.stack([torch.log(torch.as_tensor((w[k] + 1) / (w[k] + 1).sum(), dtype=torch.float32))
                         for k in PR_TARGETS]).to(device)
    splits = {"val": Split(data, "val"), "test": Split(data, "test")}
    for run in map(Path, args.runs):
        run_args = json.loads((run / "args.json").read_text())
        tau_train = 1.0 if run_args["loss"].startswith("la") else 0.0
        out = out_root / run.name
        out.mkdir(exist_ok=True)
        model = build_model(len(classes))
        model.load_state_dict(torch.load(run / "model.pt", map_location="cpu"))
        model = model.to(device).to(memory_format=torch.channels_last).eval()
        taus_post = torch.as_tensor(TAUS - tau_train, dtype=torch.float32, device=device)
        for name, split in splits.items():
            grid, pr, conf, labels = [], [], [], []
            mean_p = torch.zeros(len(classes), dtype=torch.float64, device=device)
            with torch.no_grad():
                for x, y in split.batches(1024, False):
                    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                        logits = model(to_input(x, device))
                    logits = logits.float()
                    p = logits.softmax(1)
                    mean_p += p.sum(0, dtype=torch.float64)
                    conf.append(p.max(1).values.half().cpu().numpy())
                    grid.append(torch.stack([(logits - t * log_pi).argmax(1) for t in taus_post], 1)
                                .to(torch.int16).cpu().numpy())
                    base = logits - (1 - tau_train) * log_pi
                    pr.append(torch.stack([(base + lq).argmax(1) for lq in log_q], 1).to(torch.int16).cpu().numpy())
                    labels.append(y.astype(np.int16))
            np.savez(out / f"{name}.npz", grid=np.concatenate(grid), pr=np.concatenate(pr),
                     conf=np.concatenate(conf), labels=np.concatenate(labels),
                     mean_p=(mean_p / sum(len(c) for c in conf)).cpu().numpy())
        (out / "meta.json").write_text(json.dumps({"loss": run_args["loss"], "seed": run_args["seed"],
                                                   "tau_train": tau_train, "run": str(run)}))
        print("dumped", run.name, flush=True)


# ----------------------------------------------------------------------------- analyze (CPU)
def per_class_acc(pred, labels, n_cls):
    """pred: (N,) or (N, K) -> (n_cls,) or (n_cls, K) per-class accuracy."""
    counts = np.bincount(labels, minlength=n_cls)
    correct = pred == (labels if pred.ndim == 1 else labels[:, None])
    if correct.ndim == 1:
        return np.bincount(labels, weights=correct, minlength=n_cls) / counts
    return np.stack([np.bincount(labels, weights=correct[:, k], minlength=n_cls) for k in range(correct.shape[1])], 1) \
        / counts[:, None]


def scores(acc_c, tw, lt_counts):
    """acc_c: (C,) or (C, K) -> dict metric -> scalar or (K,), in percent."""
    res = {m: 100 * (tw[m] @ acc_c) for m in METRICS}
    for g, sel in GROUPS:
        res[g] = 100 * acc_c[sel(lt_counts)].mean(0)
    return res


def bootstrap_diff(pa, pb, labels, wt, reps=1000, seed=0):
    """Class-stratified paired bootstrap of sum_c wt_c (acc_A,c - acc_B,c); returns (point, lo, hi) in pp."""
    n_cls = len(wt)
    ca, cb = (pa == labels), (pb == labels)
    cells = np.stack([np.bincount(labels, weights=ca & cb, minlength=n_cls),
                      np.bincount(labels, weights=ca & ~cb, minlength=n_cls),
                      np.bincount(labels, weights=~ca & cb, minlength=n_cls),
                      np.bincount(labels, weights=~ca & ~cb, minlength=n_cls)], 1)
    n = cells.sum(1)
    point = 100 * wt @ ((cells[:, 1] - cells[:, 2]) / n)
    rng = np.random.default_rng(seed)
    draws = rng.multinomial(n.astype(np.int64), cells / n[:, None], size=(reps, n_cls))
    diffs = 100 * ((draws[..., 1] - draws[..., 2]) / n) @ wt
    return float(point), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def analyze(args):
    dumps = Path(args.dumps)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    w = np.load(args.weights)
    lt_counts = np.load(dumps / "lt_counts.npy")
    n_cls = len(lt_counts)
    tw = target_weights(w, lt_counts)
    runs = sorted(p for p in dumps.iterdir() if (p / "meta.json").exists())
    results, curves = {}, []
    data = {}
    for run in runs:
        meta = json.loads((run / "meta.json").read_text())
        r = {"loss": meta["loss"], "seed": meta["seed"], "tau_train": meta["tau_train"]}
        for split in ("val", "test"):
            d = np.load(run / f"{split}.npz")
            labels = d["labels"].astype(np.int64)
            grid, pr = d["grid"], d["pr"]
            if split == "test":
                data[(run.name, split)] = (grid, pr, labels)
            acc_grid = per_class_acc(grid, labels, n_cls)
            s_grid = scores(acc_grid, tw, lt_counts)
            top1 = 100 * (grid == labels[:, None]).mean(0)
            for k, t in enumerate(TAUS):
                curves.append({"run": run.name, "split": split, "tau_eff": float(t), "top1": float(top1[k]),
                               **{m: float(s_grid[m][k]) for m in s_grid}})
            r[split] = {"grid": {m: s_grid[m].tolist() for m in s_grid}, "top1": top1.tolist()}
            s_pr = scores(per_class_acc(pr, labels, n_cls), tw, lt_counts)
            r[split]["pr"] = {f"PR_{t}": {m: float(s_pr[m][k]) for m in s_pr} for k, t in enumerate(PR_TARGETS)}
        # selection on val only: max metric, ties -> closest to tau_train
        sel = {}
        k0 = int(np.argmin(np.abs(TAUS - r["tau_train"])))
        for m in METRICS:
            v = np.asarray(r["val"]["grid"][m])
            best = np.flatnonzero(v == v.max())
            k = int(best[np.argmin(np.abs(TAUS[best] - r["tau_train"]))])
            sel[m] = {"tau_eff": float(TAUS[k]), "k": k, "val": float(v[k])}
        r["k_default"] = k0
        r["select"] = sel
        r["test_frozen"] = {"default": {m: float(np.asarray(r["test"]["grid"][m])[k0]) for m in r["test"]["grid"]},
                            **{f"tau*({m})": {mm: float(np.asarray(r["test"]["grid"][mm])[sel[m]["k"]])
                                              for mm in r["test"]["grid"]} for m in METRICS},
                            **r["test"]["pr"]}
        r["test_top1_default"] = float(r["test"]["top1"][k0])
        results[run.name] = r
    # sanity: default top-1 vs the A1 result.json files
    sanity = {}
    for name, r in results.items():
        run_dir = Path(json.loads((dumps / name / "meta.json").read_text())["run"])
        res_file = (Path(args.result_root) / run_dir.parent.name / name if args.result_root else run_dir) / "result.json"
        if res_file.exists():
            ref = json.loads(res_file.read_text())["test"]["acc"]
            sanity[name] = {"result_json": ref, "reeval": r["test_top1_default"], "diff": r["test_top1_default"] - ref}
    sanity_ok = bool(sanity) and all(abs(v["diff"]) <= 0.01 for v in sanity.values())
    judge = judge_hypotheses(results, data, tw, n_cls) if not args.skip_judge else {}
    (out / "c1_results.json").write_text(json.dumps({"taus": TAUS.tolist(), "sanity_ok": sanity_ok, "sanity": sanity,
                                                     "judge": judge, "runs": results}, indent=1))
    import csv
    with open(out / "curves.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(curves[0]))
        wr.writeheader()
        wr.writerows(curves)
    write_report(out, results, judge, sanity, sanity_ok)
    print("sanity_ok", sanity_ok)
    print((out / "report.md").read_text())


def judge_hypotheses(results, data, tw, n_cls):
    def run(loss, seed):
        return f"{loss}_s{seed}"

    def pred(name, key):
        grid, pr, labels = data[(name, "test")]
        r = results[name]
        k = r["k_default"] if key == "default" else r["select"][key]["k"]
        return grid[:, k], labels

    def boot(a, b, metric):
        pa, labels = pred(*a)
        pb, _ = pred(*b)
        return bootstrap_diff(pa, pb, labels, tw[metric])

    seeds = sorted({r["seed"] for r in results.values()})
    J = {"seeds": seeds}
    # H1
    h1 = {"a_tau_bal_minus_nat": {}, "b": {}, "c": {}}
    for s in seeds:
        ce = results[run("ce", s)]["select"]
        h1["a_tau_bal_minus_nat"][s] = ce["bal"]["tau_eff"] - ce["nat"]["tau_eff"]
        h1["b"][s] = boot((run("ce", s), "nat"), (run("ce", s), "bal"), "nat")
        h1["c"][s] = boot((run("ce", s), "nat"), (run("la", s), "default"), "nat")
    a_ok = all(v >= 0.5 for v in h1["a_tau_bal_minus_nat"].values())
    b_ok = bool(np.mean([v[0] for v in h1["b"].values()]) >= 0.5 and all(v[0] > 0 for v in h1["b"].values()))
    c_ok = bool(np.mean([v[0] for v in h1["c"].values()]) >= 0.5 and all(v[0] > 0 for v in h1["c"].values()))
    h1.update(a_ok=a_ok, b_ok=b_ok, c_ok=c_ok, passed=bool(a_ok and b_ok and c_ok))
    J["H1"] = h1
    # H2
    h2 = {"tau_tr": {}, "tau_bal": {}, "gain_tr": {}}
    for s in seeds:
        ce = results[run("ce", s)]["select"]
        h2["tau_tr"][s] = ce["tr"]["tau_eff"]
        h2["tau_bal"][s] = ce["bal"]["tau_eff"]
        h2["gain_tr"][s] = boot((run("ce", s), "tr"), (run("ce", s), "default"), "tr")
    h2["passed"] = bool(all(v > 0 for v in h2["tau_tr"].values()) and all(v[0] > 0 for v in h2["gain_tr"].values()))
    J["H2"] = h2
    # H3 (descriptive)
    J["H3"] = {m: {s: boot((run("la", s), m), (run("ce", s), m), m) for s in seeds} for m in ("bal", "tr", "nat")}
    # H4
    h4 = {}
    for loss in ("ls", "csls", "sals", "gls", "ugls", "ugls_f", "la", "la_ls", "la_sals", "la_ugls", "la_csls"):
        if run(loss, seeds[0]) not in results:
            continue
        h4[loss] = {"gap_default": {s: boot((run(loss, s), "default"), (run("ce", s), "default"), "bal") for s in seeds},
                    "gap_tuned": {s: boot((run(loss, s), "bal"), (run("ce", s), "bal"), "bal") for s in seeds}}
    for loss in ("ugls", "csls"):
        if loss in h4:
            h4[loss]["passed"] = bool(all(h4[loss]["gap_tuned"][s][0] <= h4[loss]["gap_default"][s][0] / 3
                                          for s in seeds))
    h4["passed"] = bool(all(h4[l].get("passed", False) for l in ("ugls", "csls") if l in h4))
    J["H4"] = h4
    return J


def write_report(out, results, judge, sanity, sanity_ok):
    L = ["# C-01 HWDB-LT: balanced vs natural-distribution evaluation", "",
         f"Sanity (default top-1 vs A1 result.json, |diff| <= 0.01pp): **{'OK' if sanity_ok else 'FAILED'}**; "
         f"max |diff| = {max((abs(v['diff']) for v in sanity.values()), default=float('nan')):.4f}", ""]
    L += ["## Test, frozen configs (mean of seeds)", "",
          "| model | config | tau_eff | Acc_bal | Acc_tr | Acc_nat | nat_doc | nat_scene | nat_web | many | medium | few |",
          "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    losses = sorted({r["loss"] for r in results.values()}, key=lambda l: (l.startswith("la"), l))
    for loss in losses:
        names = [n for n, r in results.items() if r["loss"] == loss]
        for cfg in ["default"] + [f"tau*({m})" for m in ("bal", "tr", "nat")] + ["PR_nat"]:
            vals = [results[n]["test_frozen"][cfg] for n in names]
            if cfg == "default":
                tau = f"{results[names[0]]['tau_train']:.2f}"
            elif cfg.startswith("tau*"):
                tau = " / ".join(f"{results[n]['select'][cfg[5:-1]]['tau_eff']:.2f}" for n in names)
            else:
                tau = "PR"
            row = [np.mean([v[m] for v in vals]) for m in ("bal", "tr", "nat", "nat_doc", "nat_scene", "nat_web",
                                                           "many", "medium", "few")]
            L.append(f"| {loss} | {cfg} | {tau} | " + " | ".join(f"{x:.2f}" for x in row) + " |")
    if judge:
        L += ["", "## Judgements (test; point [95% CI] per seed, pp)", ""]
        h1 = judge["H1"]
        L.append(f"- H1 (a) tau*(bal) - tau*(nat), ce: {h1['a_tau_bal_minus_nat']} -> {h1['a_ok']}")
        L.append(f"- H1 (b) ce Acc_nat tau*(nat) - tau*(bal): {fmt(h1['b'])} -> {h1['b_ok']}")
        L.append(f"- H1 (c) ce tau*(nat) - la default, Acc_nat: {fmt(h1['c'])} -> {h1['c_ok']}")
        L.append(f"- **H1 passed: {h1['passed']}**")
        h2 = judge["H2"]
        L.append(f"- H2 ce tau*(tr): {h2['tau_tr']}; tau*(bal): {h2['tau_bal']}; Acc_tr gain: {fmt(h2['gain_tr'])}")
        L.append(f"- **H2 passed: {h2['passed']}**")
        for m, v in judge["H3"].items():
            L.append(f"- H3 la - ce at tau*({m}), Acc_{m}: {fmt(v)}")
        for loss, v in judge["H4"].items():
            if loss == "passed":
                continue
            L.append(f"- H4 {loss}: gap_default {fmt(v['gap_default'])}; gap_tuned {fmt(v['gap_tuned'])}"
                     + (f" -> {v['passed']}" if "passed" in v else ""))
        L.append(f"- **H4 passed: {judge['H4']['passed']}**")
    (out / "report.md").write_text("\n".join(L) + "\n")


def fmt(d):
    return "; ".join(f"s{s}: {v[0]:+.3f} [{v[1]:+.3f}, {v[2]:+.3f}]" for s, v in d.items())


# ----------------------------------------------------------------------------- smoke (CPU)
def smoke(args):
    import tempfile
    import torch
    from a1_train_hccr import Split, build_model, evaluate
    tmp = Path(tempfile.mkdtemp())
    classes = CLASSES.read_text(encoding="utf-8").strip()[:200]  # small for CPU; nothing depends on 3,755
    data = tmp / "hwdb"
    data.mkdir()
    (data / "classes.txt").write_text(classes, encoding="utf-8")
    rng = np.random.default_rng(0)
    C = len(classes)
    for name, n in (("train", 3 * C), ("val", 2 * C), ("test", 2 * C)):
        np.save(data / f"{name}_images.npy", rng.integers(0, 256, (n, 64, 64), dtype=np.uint8))
        lab = np.concatenate([np.arange(C)] * (n // C)).astype(np.int32)
        lab[:5] = -1
        np.save(data / f"{name}_labels.npy", lab)
    np.save(data / "train_lt100_indices.npy", rng.choice(3 * C, 2 * C, replace=False))
    wts = {k: rng.integers(0, 1000, C).astype(np.float64) for k in PR_TARGETS}
    np.savez(tmp / "weights.npz", classes=np.array(list(classes)), **wts)
    runs = []
    for exp in ("A1-02_hwdb_lt", "A1-03_la_plus"):
        for loss in (("ce", "la", "ugls", "csls") if exp == "A1-02_hwdb_lt" else ("la_ls",)):
            for seed in (0, 1):
                run = tmp / exp / f"{loss}_s{seed}"
                run.mkdir(parents=True)
                torch.manual_seed(seed + len(loss))
                model = build_model(C)
                torch.save(model.state_dict(), run / "model.pt")
                (run / "args.json").write_text(json.dumps({"loss": loss, "seed": seed}))
                lt = Split(data, "train", np.load(data / "train_lt100_indices.npy"))
                counts = np.bincount(lt.labels[lt.idx], minlength=C).astype(np.float64)
                res = evaluate(model, Split(data, "test"), torch.device("cpu"), counts, classes)
                (run / "result.json").write_text(json.dumps({"test": res}))
                runs.append(str(run))
    dump(argparse.Namespace(device="cpu", data=str(data), weights=str(tmp / "weights.npz"), runs=runs,
                            out=str(tmp / "dumps"), split="lt100"))
    # self-checks on the dump
    for run in runs:
        name = Path(run).name
        d = np.load(tmp / "dumps" / name / "test.npz")
        meta = json.loads((tmp / "dumps" / name / "meta.json").read_text())
        k0 = int(np.argmin(np.abs(TAUS - meta["tau_train"])))
        assert d["grid"].shape[1] == len(TAUS) and d["pr"].shape[1] == len(PR_TARGETS)
        model = build_model(C)
        model.load_state_dict(torch.load(Path(run) / "model.pt"))
        model.eval()
        from a1_train_hccr import to_input
        sp = Split(data, "test")
        with torch.no_grad():
            ref = np.concatenate([model(to_input(x, torch.device("cpu"))).argmax(1).numpy() for x, _ in sp.batches(1024, False)])
        assert (ref == d["grid"][:, k0]).all(), "default column must equal plain argmax"
    analyze(argparse.Namespace(dumps=str(tmp / "dumps"), out=str(tmp / "out"), weights=str(tmp / "weights.npz"),
                               result_root=str(tmp), skip_judge=False))
    res = json.loads((tmp / "out" / "c1_results.json").read_text())
    assert res["sanity_ok"], res["sanity"]
    # bootstrap check: A vs A is exactly zero
    lab = rng.integers(0, 50, 5000)
    pa = np.where(rng.random(5000) < 0.7, lab, (lab + 1) % 50)
    assert bootstrap_diff(pa, pa, lab, np.full(50, 1 / 50)) == (0.0, 0.0, 0.0)
    print("SMOKE OK", tmp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("weights")
    p = sub.add_parser("dump")
    p.add_argument("--data", required=True)
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--weights", default=str(EXP / "weights.npz"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--split", default="lt100", help="training split whose counts define the prior (lt100 | nat)")
    p = sub.add_parser("analyze")
    p.add_argument("--dumps", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--weights", default=str(EXP / "weights.npz"))
    p.add_argument("--result-root", dest="result_root", help="dir holding <exp>/<run>/result.json (A1 outputs)")
    p.add_argument("--skip-judge", dest="skip_judge", action="store_true")
    sub.add_parser("smoke")
    args = parser.parse_args()
    {"weights": weights, "dump": dump, "analyze": analyze, "smoke": smoke}[args.cmd](args)
