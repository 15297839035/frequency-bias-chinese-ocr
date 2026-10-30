"""Quick check of rare-character and look-alike errors in the seed48 Scheme 5.1 test predictions.

Reads only files already in OpenOCR-main (training-frequency table, UCR author IDS table,
frozen per-sample test predictions); runs no model. Usage:
    python3 ly/scripts/longtail_quickcheck.py OpenOCR-main
"""
import collections
import json
import random
import sys

ROOT = sys.argv[1] if len(sys.argv) > 1 else "OpenOCR-main"
MODELS = ["gtc", "b"]
BUCKETS = ["0", "1-9", "10-99", "100-999", ">=1000"]
IDC = set("⿰⿱⿲⿳⿴⿵⿶⿷⿸⿹⿺⿻")

freq = json.load(open(f"{ROOT}/output/rec/bctr_scheme51_ab_bs64/train_character_frequency.json"))
cnt = dict(zip(freq["characters"], freq["counts"]))
ids = {}
ids_path = f"{ROOT}/third_party/ucr_author_320ea5e8f141/Pandarenlql-UCR-320ea5e/files/IDS_V7.txt"
for line in open(ids_path, encoding="utf-8-sig"):
    ch, sep, seq = line.rstrip("\n").partition(":")
    if sep:
        ids[ch] = seq.split()


def bucket(n):
    if n == 0:
        return "0"
    if n < 10:
        return "1-9"
    if n < 100:
        return "10-99"
    if n < 1000:
        return "100-999"
    return ">=1000"


def is_han(c):
    return "一" <= c <= "鿿" or "㐀" <= c <= "䶿"


def align(ref, hyp):
    n, m = len(ref), len(hyp)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]))
    i, j, ops = n, m, []
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            ops.append(("M" if ref[i - 1] == hyp[j - 1] else "S", ref[i - 1], hyp[j - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and d[i][j] == d[i - 1][j] + 1:
            ops.append(("D", ref[i - 1], None))
            i -= 1
        else:
            ops.append(("I", None, hyp[j - 1]))
            j -= 1
    return ops[::-1]


def shares_component(a, b):
    return bool((set(ids[a]) & set(ids[b])) - IDC)



def main():
    random.seed(0)
    pool = [c for c in cnt if c in ids and cnt[c] > 0]
    weights = [cnt[c] for c in pool]

    for dom in ["scene", "web", "document"]:
        tot = {m: collections.Counter() for m in MODELS}
        hit = {m: collections.Counter() for m in MODELS}
        line_tot = collections.Counter()
        line_ok = {m: collections.Counter() for m in MODELS}
        subs = []
        for raw in open(f"{ROOT}/output/rec/scheme51_frozen_test_seed48/{dom}_predictions.jsonl", encoding="utf-8"):
            r = json.loads(raw)
            ref = r["target_normalized"]
            group = "has<100" if any(is_han(c) and cnt.get(c, 0) < 100 for c in ref) else "all>=100"
            line_tot[group] += 1
            for m in MODELS:
                line_ok[m][group] += r["models"][m]["correct"]
                for op, a, b in align(ref, r["models"][m]["prediction_normalized"]):
                    if a is None or not is_han(a):
                        continue
                    k = bucket(cnt.get(a, 0))
                    tot[m][k] += 1
                    hit[m][k] += op == "M"
                    if m == "gtc" and op == "S" and is_han(b) and a in ids and b in ids:
                        subs.append((a, b))
        print(f"\n=== {dom} ===")
        print("bucket(train count) | share of test Han chars % | " + " | ".join(f"{m} recall %" for m in MODELS) + " | share of gtc char errors %")
        all_chars = sum(tot["gtc"].values())
        errs = {k: tot["gtc"][k] - hit["gtc"][k] for k in BUCKETS}
        n_err = sum(errs.values())
        for k in BUCKETS:
            if tot["gtc"][k]:
                print(f"{k} | {100 * tot['gtc'][k] / all_chars:.2f} | "
                      + " | ".join(f"{100 * hit[m][k] / tot[m][k]:.2f}" for m in MODELS)
                      + f" | {100 * errs[k] / n_err:.2f}")
        for g in ["has<100", "all>=100"]:
            print(f"lines {g}: n={line_tot[g]} | " + " | ".join(f"{m} line acc {100 * line_ok[m][g] / line_tot[g]:.2f}%" for m in MODELS))
        rnd = [(random.choices(pool, weights)[0], random.choices(pool, weights)[0]) for _ in range(len(subs))]
        s_obs = 100 * sum(shares_component(a, b) for a, b in subs) / max(1, len(subs))
        s_rnd = 100 * sum(shares_component(a, b) for a, b in rnd) / max(1, len(rnd))
        print(f"gtc Han substitutions (both chars in IDS table): {len(subs)}; share >=1 component: {s_obs:.1f}% (frequency-matched random pairs: {s_rnd:.1f}%)")


if __name__ == "__main__":
    main()
