"""Re-read (no new selection) C-06's self-prior test results for the two industrial models, keeping the repaired /
broken rare-line counts that c6_results.json dropped (for the paired CIs of Table 2). Same code path as c6.run()."""
import json, sys
from pathlib import Path
sys.path.insert(0, "ly/scripts")
import a0_prior_correction as a0
import c4_analysis as c4
import c6_selfprior as c6
out = {}
for tag in ("svtrv2_ch", "repsvtr_ch"):
    a0.DUMPS = c6.EXP / "C-04b_industrial/dumps"
    a0.normalize = c4.official_normalizer()
    val, test = c4.load_split(tag, "val"), c4.load_split(tag, "test")
    lp_hat, _ = c6.self_prior(val)
    a0.LOG_PRIOR = lp_hat
    _, _, chosen = c4.select(val)
    t = c4.test_eval(test, {"B0": None, "G_selected": chosen})
    n_rare = int(sum(test[d]["rare"].sum() for d in a0.DOMAINS))
    out[tag] = {"selected": chosen, "n_rare": n_rare, **{k: v for k, v in t["G_selected"].items() if k != "domains"}}
    a0.LOG_PRIOR = c6.BCTR_LOG_PRIOR
    print(tag, out[tag], flush=True)
Path("ly/exp/V4A_cpu/selfprior_counts.json").write_text(json.dumps(out, indent=1, default=float))
