| run | tau_train | selected (tau_eff, gamma) | val acc / rare | test default acc / rare | test selected acc / rare |
|---|---:|---|---|---|---|
| c0_s1 | 0.0 | 0.2, 0.9 | 85.979 / 78.316 | 85.842 / 77.810 | 85.847 / 79.061 |
| c0_s2 | 0.0 | 0.2, 0.9 | 85.894 / 78.251 | 85.813 / 77.679 | 85.825 / 78.840 |
| la025_s1 | 0.25 | 0.2, 0.9 | 85.954 / 78.238 | 85.801 / 79.218 | 85.828 / 79.035 |
| la025_s2 | 0.25 | 0.2, 0.9 | 85.906 / 78.303 | 85.739 / 79.035 | 85.777 / 78.931 |
| la1_s1 | 1.0 | 0.1, 1.01 (infeasible) | 85.710 / 77.502 | 78.815 / 79.400 | 85.609 / 78.644 |
| la1_s2 | 1.0 | 0.1, 1.01 (infeasible) | 85.698 / 77.644 | 78.973 / 79.218 | 85.552 / 78.357 |
| epoch100 (reference) | 0 | default | 85.987 / 77.464 | 85.877 / 77.940 | — |

- la025 vs c0 (selected vs selected): verdict **equivalent**
  - seed 1: rare -0.026 [-0.261, +0.222], acc -0.019 [-0.075, +0.033], rare fixed/broken 45/47 (McNemar p=0.92)
  - seed 2: rare +0.091 [-0.156, +0.339], acc -0.048 [-0.100, +0.007], rare fixed/broken 49/42 (McNemar p=0.53)
- la1 vs c0 (selected vs selected): verdict **equivalent**
  - seed 1: rare -0.417 [-0.691, -0.091], acc -0.239 [-0.309, -0.170], rare fixed/broken 57/89 (McNemar p=0.01)
  - seed 2: rare -0.482 [-0.783, -0.169], acc -0.273 [-0.345, -0.202], rare fixed/broken 52/89 (McNemar p=0.0023)
