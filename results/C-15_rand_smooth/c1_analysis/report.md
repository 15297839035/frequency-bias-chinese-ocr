# C-01 HWDB-LT: balanced vs natural-distribution evaluation

Sanity (default top-1 vs A1 result.json, |diff| <= 0.01pp): **OK**; max |diff| = 0.0000

## Test, frozen configs (mean of seeds)

| model | config | tau_eff | Acc_bal | Acc_tr | Acc_nat | nat_doc | nat_scene | nat_web | many | medium | few |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rgls | default | 0.00 | 93.94 | 96.57 | 97.09 | 97.12 | 97.02 | 97.10 | 97.02 | 94.69 | 89.40 |
| rgls | tau*(bal) | 0.85 / 0.80 / 0.80 / 0.80 | 95.62 | 95.87 | 95.93 | 95.94 | 95.87 | 96.02 | 96.01 | 95.52 | 95.28 |
| rgls | tau*(tr) | 0.20 / 0.20 / 0.15 / 0.25 | 94.68 | 96.58 | 96.98 | 97.01 | 96.91 | 97.01 | 96.93 | 95.09 | 91.55 |
| rgls | tau*(nat) | -0.45 / -0.60 / -0.55 / -0.55 | 90.18 | 96.18 | 97.21 | 97.24 | 97.15 | 97.14 | 97.09 | 92.89 | 78.76 |
| rgls | PR_nat | PR | 88.00 | 95.92 | 97.33 | 97.43 | 97.19 | 97.10 | 96.87 | 92.69 | 71.88 |
