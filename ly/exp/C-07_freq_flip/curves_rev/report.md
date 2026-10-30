# C-01 HWDB-LT: balanced vs natural-distribution evaluation

Sanity (default top-1 vs A1 result.json, |diff| <= 0.01pp): **OK**; max |diff| = 0.0000

## Test, frozen configs (mean of seeds)

| model | config | tau_eff | Acc_bal | Acc_tr | Acc_nat | nat_doc | nat_scene | nat_web | many | medium | few |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ce | default | 0.00 | 92.02 | 97.40 | 76.27 | 75.25 | 77.39 | 79.36 | 98.18 | 94.37 | 81.94 |
| ce | tau*(bal) | 1.15 / 1.15 / 1.15 / 1.15 | 95.63 | 96.92 | 92.60 | 92.54 | 92.57 | 93.28 | 97.11 | 95.70 | 93.79 |
| ce | tau*(tr) | 0.40 / 0.45 / 0.30 / 0.40 | 94.16 | 97.55 | 84.76 | 84.47 | 84.91 | 86.62 | 98.05 | 95.31 | 88.18 |
| ce | tau*(nat) | 2.00 / 2.00 / 2.00 / 2.00 | 93.27 | 91.09 | 94.93 | 94.89 | 94.94 | 95.15 | 90.97 | 93.85 | 95.30 |
| ce | PR_nat | PR | 72.43 | 40.30 | 94.45 | 94.61 | 94.23 | 93.90 | 40.95 | 85.48 | 93.80 |
