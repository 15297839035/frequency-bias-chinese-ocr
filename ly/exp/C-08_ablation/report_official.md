# C-08 ablations (official normalisation; test, each variant selected on val)

| model | B0 acc / rare | seg | seg_nogate | seg_top2 | seg_top5 | frame | tau=1 (g0.9) |
|---|---|---|---|---|---|---|---|
| gtc | 86.39 / 79.79 | +1.07 / +0.04 (τ0.2, γ0.9) | +1.07 / +0.04 (τ0.2, γ1.01) | +1.00 / +0.04 (τ0.2, γ0.9) | +1.06 / +0.04 (τ0.2, γ0.9) | +0.72 / +0.01 (τ0.1, γ1.01) | +1.72 / -1.91 |
| b | 86.65 / 80.55 | +0.95 / +0.02 (τ0.2, γ0.9) | +0.95 / +0.02 (τ0.2, γ1.01) | +0.87 / +0.02 (τ0.2, γ0.9) | +0.95 / +0.02 (τ0.2, γ0.9) | +0.65 / +0.01 (τ0.1, γ1.01) | +1.29 / -1.81 |
| off_T | 84.45 / 78.31 | +0.83 / -0.02 (τ0.2, γ0.9) | +0.83 / -0.02 (τ0.2, γ1.01) | +0.72 / -0.02 (τ0.2, γ0.9) | +0.83 / -0.02 (τ0.2, γ0.9) | +1.02 / -0.07 (τ0.2, γ1.01) | +1.58 / -1.84 |
| off_S | 86.40 / 80.48 | +1.00 / -0.06 (τ0.3, γ0.9) | +1.00 / -0.06 (τ0.3, γ1.01) | +0.89 / -0.05 (τ0.3, γ0.9) | +1.00 / -0.06 (τ0.3, γ0.9) | +0.90 / -0.04 (τ0.2, γ1.01) | +1.51 / -1.05 |
| off_B | 87.98 / 82.65 | +0.93 / -0.03 (τ0.3, γ0.9) | +0.93 / -0.03 (τ0.3, γ1.01) | +0.77 / -0.03 (τ0.3, γ0.9) | +0.93 / -0.03 (τ0.3, γ0.9) | +0.42 / +0.01 (τ0.1, γ1.01) | +1.34 / -0.68 |
| svtrv2_ch | 77.78 / 65.58 | +0.42 / -0.01 (τ0.1, γ0.9) | +0.42 / -0.01 (τ0.1, γ1.01) | +0.35 / -0.01 (τ0.1, γ0.9) | +0.42 / -0.01 (τ0.1, γ0.9) | +0.57 / +0.07 (τ0.1, γ1.01) | +0.00 / -3.29 |
| repsvtr_ch | 74.89 / 64.04 | +0.70 / +0.02 (τ0.1, γ0.9) | +0.70 / +0.02 (τ0.1, γ1.01) | +0.66 / +0.02 (τ0.1, γ0.9) | +0.70 / +0.02 (τ0.1, γ0.9) | +1.08 / +0.11 (τ0.1, γ1.01) | +0.03 / -4.11 |

Cells: rare-line gain / overall change (pp) of the val-selected config.

## gtc
- G*: CER 4.20 -> 4.20, NED 94.05 -> 94.04; rare-output precision 95.6 -> 93.6
- recall by bucket B0 -> G*: 0 2.3->2.3, 1-9 69.8->73.0, 10-99 90.5->91.5, 100-999 94.1->94.4, >=1000 97.2->97.1
- precision by bucket B0 -> G*: 0 100.0->100.0, 1-9 93.4->90.4, 10-99 95.8->93.8, 100-999 95.6->94.7, >=1000 97.3->97.4
- rare threshold (lines, B0 -> G*): <10: 765, 59.08->61.31, <50: 4021, 75.65->77.05, <100: 7670, 79.79->80.86, <500: 28735, 84.16->84.68
- look-alike share of 857 rare substitutions (observed / random substitute): ids_shared_component 61.0/6.9, glyph_top10 33.4/0.2, glyph_top50 48.2/1.2, union_used_in_paper 69.1/7.8
- beta by min errors: N>=1: 12858 pairs, T/E 2.41, beta 1.23, N>=3: 373 pairs, T/E 1.97, beta 1.17, N>=5: 102 pairs, T/E 1.91, beta 1.23, N>=10: 14 pairs, T/E 2.06, beta 1.52
- beta with intercept 1.19 [1.0654216597092663, 1.312174697091662] (alpha -0.03 [-0.22323193104014807, 0.12868189607561453]); with IDS-complexity 1.17 [1.0877127887096603, 1.2504684264987074] (eta -0.010 [-0.14052104494542417, 0.09170385191830512])
- ceiling: 1249 rare substitutions, 63.5% recoverable, G* repairs 16.1% of those

## b
- G*: CER 4.13 -> 4.14, NED 94.18 -> 94.17; rare-output precision 96.1 -> 94.3
- recall by bucket B0 -> G*: 0 2.3->2.3, 1-9 71.0->74.0, 10-99 90.9->91.7, 100-999 94.2->94.5, >=1000 97.2->97.2
- precision by bucket B0 -> G*: 0 100.0->100.0, 1-9 94.4->91.6, 10-99 96.3->94.5, 100-999 95.7->94.8, >=1000 97.4->97.6
- rare threshold (lines, B0 -> G*): <10: 765, 59.87->62.35, <50: 4021, 76.55->77.84, <100: 7670, 80.55->81.50, <500: 28735, 84.62->85.09
- look-alike share of 789 rare substitutions (observed / random substitute): ids_shared_component 61.9/6.6, glyph_top10 32.1/0.4, glyph_top50 47.4/2.2, union_used_in_paper 68.9/8.1
- beta by min errors: N>=1: 12317 pairs, T/E 2.40, beta 1.22, N>=3: 362 pairs, T/E 2.04, beta 1.15, N>=5: 96 pairs, T/E 1.98, beta 1.17, N>=10: 14 pairs, T/E 2.22, beta 1.49
- beta with intercept 1.13 [1.0147758570024366, 1.236736878178662] (alpha 0.05 [-0.10704630499222137, 0.20883744315979344]); with IDS-complexity 1.15 [1.073964955533458, 1.2264313233625252] (eta 0.011 [-0.08634113767428077, 0.13271535579923296])
- ceiling: 1178 rare substitutions, 63.6% recoverable, G* repairs 15.8% of those

## off_T
- G*: CER 6.30 -> 6.31, NED 92.02 -> 92.00; rare-output precision 96.0 -> 93.9
- recall by bucket B0 -> G*: 0 0.0->0.0, 1-9 67.8->71.9, 10-99 89.1->89.9, 100-999 91.5->91.8, >=1000 94.6->94.6
- precision by bucket B0 -> G*: 0 0.0->0.0, 1-9 95.4->91.7, 10-99 96.1->94.0, 100-999 95.6->94.5, >=1000 97.1->97.2
- rare threshold (lines, B0 -> G*): <10: 765, 57.39->60.78, <50: 4021, 74.43->75.70, <100: 7670, 78.31->79.14, <500: 28735, 82.06->82.41
- look-alike share of 812 rare substitutions (observed / random substitute): ids_shared_component 61.0/6.7, glyph_top10 33.9/0.4, glyph_top50 47.7/1.6, union_used_in_paper 69.0/8.0
- beta by min errors: N>=1: 13441 pairs, T/E 2.36, beta 1.21, N>=3: 406 pairs, T/E 2.04, beta 1.22, N>=5: 107 pairs, T/E 1.93, beta 1.25, N>=10: 16 pairs, T/E 1.81, beta 1.38
- beta with intercept 1.27 [1.1617784386318035, 1.387728104604084] (alpha -0.09 [-0.2699864321778142, 0.09186824724874088]); with IDS-complexity 1.23 [1.1606274048926872, 1.298242787111745] (eta -0.032 [-0.13808037821211594, 0.07890984519367])
- ceiling: 1160 rare substitutions, 63.4% recoverable, G* repairs 16.5% of those

## off_S
- G*: CER 5.49 -> 5.50, NED 93.22 -> 93.19; rare-output precision 95.8 -> 94.1
- recall by bucket B0 -> G*: 0 0.0->0.0, 1-9 72.7->76.0, 10-99 90.0->90.5, 100-999 92.5->92.7, >=1000 95.4->95.3
- precision by bucket B0 -> G*: 0 0.0->0.0, 1-9 95.1->91.8, 10-99 95.9->94.3, 100-999 95.3->94.4, >=1000 97.2->97.3
- rare threshold (lines, B0 -> G*): <10: 765, 63.01->65.36, <50: 4021, 76.75->77.92, <100: 7670, 80.48->81.25, <500: 28735, 84.05->84.35
- look-alike share of 772 rare substitutions (observed / random substitute): ids_shared_component 58.5/7.9, glyph_top10 31.3/0.1, glyph_top50 44.8/0.6, union_used_in_paper 65.8/8.3
- beta by min errors: N>=1: 13483 pairs, T/E 2.35, beta 1.16, N>=3: 357 pairs, T/E 2.15, beta 1.18, N>=5: 88 pairs, T/E 2.00, beta 1.20, N>=10: 11 pairs, T/E 1.82, beta 1.59
- beta with intercept 1.10 [0.9948155042731502, 1.2343741252788314] (alpha 0.16 [-0.05369592276082288, 0.34818140062133984]); with IDS-complexity 1.17 [1.087630444868343, 1.2628616296188993] (eta 0.031 [-0.10236444049835114, 0.1815335319008465])
- ceiling: 1137 rare substitutions, 62.5% recoverable, G* repairs 13.4% of those

## off_B
- G*: CER 4.76 -> 4.77, NED 94.21 -> 94.19; rare-output precision 96.0 -> 94.1
- recall by bucket B0 -> G*: 0 9.3->9.3, 1-9 77.1->79.7, 10-99 91.3->91.7, 100-999 93.5->93.7, >=1000 96.0->96.0
- precision by bucket B0 -> G*: 0 100.0->66.7, 1-9 94.1->91.1, 10-99 96.1->94.4, 100-999 95.8->95.1, >=1000 97.5->97.6
- rare threshold (lines, B0 -> G*): <10: 765, 66.67->69.15, <50: 4021, 79.43->80.28, <100: 7670, 82.65->83.26, <500: 28735, 85.75->86.02
- look-alike share of 678 rare substitutions (observed / random substitute): ids_shared_component 58.0/8.0, glyph_top10 31.3/0.6, glyph_top50 44.4/2.1, union_used_in_paper 65.2/8.7
- beta by min errors: N>=1: 11887 pairs, T/E 2.31, beta 1.13, N>=3: 282 pairs, T/E 2.03, beta 1.15, N>=5: 64 pairs, T/E 2.18, beta 1.22, N>=10: 15 pairs, T/E 2.11, beta 1.28
- beta with intercept 1.11 [0.9907688834875152, 1.2398140523977133] (alpha 0.08 [-0.11421276780965228, 0.2891816190309724]); with IDS-complexity 1.14 [1.058736376550584, 1.2354756550441075] (eta 0.052 [-0.11016795997309244, 0.18983153796033553])
- ceiling: 1043 rare substitutions, 64.1% recoverable, G* repairs 12.9% of those

## svtrv2_ch
- G*: CER 9.55 -> 9.61, NED 87.52 -> 87.39; rare-output precision 94.8 -> 93.4
- recall by bucket B0 -> G*: 0 9.3->11.6, 1-9 34.7->38.4, 10-99 79.4->80.4, 100-999 86.5->86.8, >=1000 91.5->91.4
- precision by bucket B0 -> G*: 0 100.0->83.3, 1-9 90.3->87.6, 10-99 95.0->93.6, 100-999 94.7->93.5, >=1000 96.2->96.5
- rare threshold (lines, B0 -> G*): <10: 765, 32.16->35.03, <50: 4021, 58.79->60.08, <100: 7670, 65.58->66.51, <500: 28735, 73.28->73.61
- look-alike share of 1345 rare substitutions (observed / random substitute): ids_shared_component 63.2/8.3, glyph_top10 38.6/0.1, glyph_top50 54.9/1.1, union_used_in_paper 72.3/9.1
- beta by min errors: N>=1: 15991 pairs, T/E 2.49, beta 1.31, N>=3: 684 pairs, T/E 2.01, beta 1.17, N>=5: 208 pairs, T/E 1.72, beta 0.99, N>=10: 43 pairs, T/E 1.39, beta 0.70
- beta with intercept 1.13 [0.9074654125562801, 1.3410804376229821] (alpha 0.08 [-0.19681229404504424, 0.3782376205366793]); with IDS-complexity 1.18 [1.0461498765337716, 1.3139352948780492] (eta -0.050 [-0.16023358270435317, 0.05928465596862468])
- ceiling: 1848 rare substitutions, 50.2% recoverable, G* repairs 14.5% of those

## repsvtr_ch
- G*: CER 10.86 -> 10.92, NED 85.68 -> 85.54; rare-output precision 94.8 -> 92.3
- recall by bucket B0 -> G*: 0 7.0->7.0, 1-9 41.9->45.9, 10-99 78.9->80.4, 100-999 84.7->85.0, >=1000 90.3->90.2
- precision by bucket B0 -> G*: 0 100.0->100.0, 1-9 90.6->86.3, 10-99 94.9->92.6, 100-999 94.2->92.9, >=1000 96.1->96.5
- rare threshold (lines, B0 -> G*): <10: 765, 36.99->39.87, <50: 4021, 58.34->59.96, <100: 7670, 64.04->65.28, <500: 28735, 70.58->71.08
- look-alike share of 1263 rare substitutions (observed / random substitute): ids_shared_component 63.3/8.6, glyph_top10 38.2/0.2, glyph_top50 53.3/1.1, union_used_in_paper 72.1/9.3
- beta by min errors: N>=1: 16293 pairs, T/E 2.49, beta 1.29, N>=3: 770 pairs, T/E 2.02, beta 1.18, N>=5: 208 pairs, T/E 1.77, beta 1.02, N>=10: 45 pairs, T/E 1.37, beta 0.62
- beta with intercept 1.15 [0.9986030784749835, 1.3220398449656954] (alpha 0.05 [-0.2088887707587827, 0.26364278085214443]); with IDS-complexity 1.18 [1.0890837096058144, 1.273587202598356] (eta 0.004 [-0.10546019863726194, 0.11375849904963353])
- ceiling: 1733 rare substitutions, 58.6% recoverable, G* repairs 17.5% of those

