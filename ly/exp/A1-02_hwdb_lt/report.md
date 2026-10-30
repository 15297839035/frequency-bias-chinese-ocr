# A1-02 HWDB-LT results (ICDAR2013 test, seeds [0, 1])

| loss | top-1 | many | medium | few | ECE | err IDS-ED | err look-alike % | top-1 per seed |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| ce | 93.64 | 97.16 | 94.66 | 88.25 | 9.26 | 2.82 | 68.84 | s0: 93.62, s1: 93.65 |
| ls | 93.77 | 97.14 | 94.80 | 88.55 | 20.46 | 2.78 | 69.90 | s0: 93.74, s1: 93.80 |
| la | 96.09 | 96.38 | 96.05 | 95.78 | 9.50 | 2.71 | 68.07 | s0: 96.09, s1: 96.08 |
| csls | 94.67 | 96.93 | 95.09 | 91.51 | 16.86 | 2.75 | 69.33 | s0: 94.71, s1: 94.63 |
| sals | 94.21 | 97.07 | 94.78 | 90.15 | 18.38 | 2.70 | 71.12 | s0: 94.21, s1: 94.21 |
| gls | 93.84 | 96.97 | 94.60 | 89.24 | 18.53 | 2.86 | 69.36 | s0: 93.85, s1: 93.84 |
| ugls | 94.21 | 96.99 | 94.77 | 90.26 | 17.99 | 2.76 | 70.10 | s0: 94.21, s1: 94.21 |
| ugls_f | 94.42 | 96.92 | 94.77 | 91.04 | 20.39 | 2.76 | 69.78 | s0: 94.42, s1: 94.42 |

## Pre-registered verdict (A1-00 定稿)

- ugls - ls (top-1, mean): +0.44pp (need >= +0.50)
- ugls - csls (top-1, mean): -0.46pp (need >= +0.50)
- ugls beats ls and csls in every seed: False
- **Main line: FAIL**
- ugls - ls (few group, mean): +1.71pp (need >= +2.00, with top-1 not below ls)
- **Auxiliary line (tail-effective): FAIL**
