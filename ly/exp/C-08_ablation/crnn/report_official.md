# C-08 ablations (official normalisation; test, each variant selected on val)

| model | B0 acc / rare | seg | seg_nogate | seg_top2 | seg_top5 | frame | tau=1 (g0.9) |
|---|---|---|---|---|---|---|---|
| crnn_s1 | 71.65 / 62.63 | +2.54 / +0.02 (τ0.3, γ0.9) | +2.54 / +0.02 (τ0.3, γ1.01) | +2.14 / +0.03 (τ0.3, γ0.9) | +2.50 / +0.02 (τ0.3, γ0.9) | +1.86 / +0.08 (τ0.2, γ1.01) | +2.62 / -3.30 |
| crnn_s2 | 71.53 / 61.51 | +2.76 / +0.04 (τ0.3, γ0.9) | +2.76 / +0.04 (τ0.3, γ1.01) | +2.28 / +0.05 (τ0.3, γ0.9) | +2.72 / +0.05 (τ0.3, γ0.9) | +0.90 / -0.01 (τ0.1, γ1.01) | +2.76 / -3.28 |

Cells: rare-line gain / overall change (pp) of the val-selected config.

## crnn_s1
- G*: CER 13.08 -> 13.07, NED 82.26 -> 82.27; rare-output precision 94.0 -> 88.5
- recall by bucket B0 -> G*: 0 0.0->0.0, 1-9 52.4->57.4, 10-99 77.0->79.1, 100-999 81.4->82.1, >=1000 88.4->88.3
- precision by bucket B0 -> G*: 0 0.0->0.0, 1-9 93.9->85.1, 10-99 94.0->88.8, 100-999 92.3->89.5, >=1000 92.8->93.2
- rare threshold (lines, B0 -> G*): <10: 765, 40.26->43.79, <50: 4021, 57.85->60.08, <100: 7670, 62.63->64.43, <500: 28735, 67.89->68.74
- look-alike share of 1558 rare substitutions (observed / random substitute): ids_shared_component 47.1/7.1, glyph_top10 27.2/0.1, glyph_top50 40.9/0.9, union_used_in_paper 56.7/8.0
- beta by min errors: N>=1: 29294 pairs, T/E 2.54, beta 1.35, N>=3: 1454 pairs, T/E 2.01, beta 1.23, N>=5: 365 pairs, T/E 1.91, beta 1.22, N>=10: 64 pairs, T/E 1.65, beta 1.17
- beta with intercept 1.16 [1.0796184024265787, 1.24920611741778] (alpha 0.13 [0.02993873267129772, 0.22839328181766974]); with IDS-complexity 1.27 [1.2207963208346455, 1.3237729813096604] (eta -0.154 [-0.21735469705737184, -0.09155492097751729])
- ceiling: 2028 rare substitutions, 56.1% recoverable, G* repairs 20.0% of those

## crnn_s2
- G*: CER 12.92 -> 12.92, NED 82.60 -> 82.63; rare-output precision 94.2 -> 89.2
- recall by bucket B0 -> G*: 0 2.3->2.3, 1-9 52.2->57.4, 10-99 76.8->78.9, 100-999 81.7->82.4, >=1000 88.6->88.5
- precision by bucket B0 -> G*: 0 100.0->100.0, 1-9 94.9->88.7, 10-99 94.2->89.2, 100-999 92.5->89.7, >=1000 92.8->93.1
- rare threshold (lines, B0 -> G*): <10: 765, 40.92->44.31, <50: 4021, 56.68->59.24, <100: 7670, 61.51->63.53, <500: 28735, 67.37->68.29
- look-alike share of 1590 rare substitutions (observed / random substitute): ids_shared_component 49.1/8.1, glyph_top10 26.5/0.2, glyph_top50 41.2/0.9, union_used_in_paper 57.5/8.8
- beta by min errors: N>=1: 29553 pairs, T/E 2.56, beta 1.37, N>=3: 1501 pairs, T/E 2.01, beta 1.28, N>=5: 374 pairs, T/E 1.89, beta 1.30, N>=10: 63 pairs, T/E 1.84, beta 1.34
- beta with intercept 1.25 [1.1903192881211129, 1.3307148751140883] (alpha 0.05 [-0.04847421018213748, 0.14279953121773917]); with IDS-complexity 1.31 [1.2700360046067551, 1.3667385091351902] (eta -0.141 [-0.20074490477331675, -0.08090328266630278])
- ceiling: 2053 rare substitutions, 55.4% recoverable, G* repairs 19.5% of those

