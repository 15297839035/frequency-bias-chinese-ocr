# C-01 HWDB-LT: balanced vs natural-distribution evaluation

Sanity (default top-1 vs A1 result.json, |diff| <= 0.01pp): **OK**; max |diff| = 0.0000

## Test, frozen configs (mean of seeds)

| model | config | tau_eff | Acc_bal | Acc_tr | Acc_nat | nat_doc | nat_scene | nat_web | many | medium | few |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ce | default | 0.00 | 92.47 | 97.19 | 90.23 | 89.92 | 90.53 | 90.64 | 97.82 | 94.32 | 83.94 |
| ce | tau*(bal) | 1.10 / 1.05 | 95.90 | 96.60 | 94.47 | 94.41 | 94.49 | 94.77 | 96.62 | 95.77 | 95.21 |
| ce | tau*(tr) | 0.30 / 0.30 | 94.25 | 97.29 | 92.23 | 91.97 | 92.49 | 92.65 | 97.69 | 95.07 | 89.21 |
| ce | tau*(nat) | 1.05 / 1.05 | 95.90 | 96.64 | 94.46 | 94.41 | 94.48 | 94.74 | 96.68 | 95.77 | 95.13 |
| ce | PR_nat | PR | 79.86 | 82.20 | 96.21 | 96.33 | 96.06 | 95.81 | 82.37 | 80.32 | 76.34 |
