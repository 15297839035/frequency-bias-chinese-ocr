"""A0 self-test of the correction code on one val dump (no test data is touched).

Checks that B0 reproduces the model's greedy output, that tau=0 changes nothing, and prints
example lines changed by one L-LA setting so the behaviour can be eyeballed.
Usage: ly/.venv/bin/python ly/scripts/a0_selftest.py web
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from a0_prior_correction import decode, load, strings  # noqa: E402

domain = sys.argv[1] if len(sys.argv) > 1 else "web"
d = load("gtc", "val", domain)
base = strings(d, decode(d, None))
assert base == d["baseline"], "B0 does not reproduce the dumped greedy predictions"
for cfg in (dict(method="G", tau=0.0, k=0, gamma=1.01), dict(method="L", tau=0.0, k=50, gamma=1.01)):
    assert strings(d, decode(d, cfg)) == base, f"tau=0 must be a no-op: {cfg}"
print(f"{domain} val: {len(base)} lines, {len(d['seg_star'])} segments, {len(d['cand_seg'])} candidates; checks passed")
cfg = dict(method="L", tau=1.0, k=10, gamma=1.01)
new = strings(d, decode(d, cfg))
changed = [i for i, (a, b) in enumerate(zip(base, new)) if a != b]
fixed = sum(new[i] == d["targets"][i] for i in changed)
broken = sum(base[i] == d["targets"][i] for i in changed)
print(f"{cfg}: changed {len(changed)} lines, fixed {fixed}, broken {broken}")
for i in changed[:15]:
    print(f"  ref={d['targets'][i]}  B0={base[i]}  L-LA={new[i]}")
