# C-13 frequency flip on BCTR text lines (official normalisation)

Pairs analysed: 54 (dropped for exposure misfit: 0)

## test

| arm | errors | toward high | expected | D [95% CI] |
|---|---:|---:|---:|---|
| nat | 228 | 149 | 46.3 | 3.22 [2.50, 3.96] |
| rev | 497 | 44 | 91.6 | 0.48 [0.30, 0.67] |
| bal | 308 | 97 | 58.9 | 1.65 [1.25, 2.03] |

- beta (own exposure, joint over arms): 1.146 [0.978, 1.377]
- secondary (pairs with >= 3 errors under nat and rev): {'pairs': 22, 'nat': 0.6666666666666666, 'rev': 0.09090909090909091, 'bal': 0.3193916349809886}

| model | arm | acc | rare-line acc | target-line acc | recall high / low |
|---|---|---:|---:|---:|---|
| crnn_s1 | nat | 71.65 | 62.63 | 63.42 | 78.2 / 74.3 |
| crnn_s2 | nat | 71.53 | 61.51 | 63.05 | 78.6 / 74.8 |
| rev_s1 | rev | 71.18 | 62.03 | 61.18 | 72.9 / 77.7 |
| rev_s2 | rev | 71.34 | 61.84 | 61.56 | 73.3 / 78.9 |
| bal_s1 | bal | 72.12 | 63.26 | 63.19 | 75.9 / 77.2 |
| bal_s2 | bal | 71.79 | 62.99 | 62.67 | 75.5 / 77.0 |

## val (pairs were selected on val: descriptive only)

| arm | errors | toward high | expected | D [95% CI] |
|---|---:|---:|---:|---|
| nat | 367 | 228 | 82.8 | 2.75 [2.23, 3.24] |
| rev | 523 | 78 | 105.1 | 0.74 [0.55, 0.97] |
| bal | 381 | 121 | 79.9 | 1.51 [1.23, 1.81] |

- beta (own exposure, joint over arms): 1.045 [0.869, 1.224]
- secondary (pairs with >= 3 errors under nat and rev): {'pairs': 44, 'nat': 0.6295180722891566, 'rev': 0.14814814814814814, 'bal': 0.3114754098360656}

| model | arm | acc | rare-line acc | target-line acc | recall high / low |
|---|---|---:|---:|---:|---|
| crnn_s1 | nat | 71.68 | 61.99 | 62.84 | 77.9 / 74.9 |
| crnn_s2 | nat | 71.58 | 61.59 | 62.94 | 78.4 / 75.5 |
| rev_s1 | rev | 71.43 | 61.24 | 61.24 | 73.1 / 79.2 |
| rev_s2 | rev | 71.42 | 61.57 | 61.51 | 73.8 / 79.2 |
| bal_s1 | bal | 72.18 | 62.77 | 63.03 | 76.0 / 77.7 |
| bal_s2 | bal | 71.88 | 62.25 | 62.71 | 75.6 / 77.6 |

**Criteria (test): {'F1_flip': True, 'F2_dose': True, 'F3_beta_own_exposure': True}**
