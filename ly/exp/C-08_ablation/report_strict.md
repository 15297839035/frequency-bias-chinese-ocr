# C-08 ablations (strict normalisation; test, each variant selected on val)

| model | B0 acc / rare | seg | seg_nogate | seg_top2 | seg_top5 | frame | tau=1 (g0.9) |
|---|---|---|---|---|---|---|---|
| gtc | 85.88 / 77.94 | +1.28 / +0.03 (τ0.2, γ0.9) | +1.28 / +0.03 (τ0.2, γ1.01) | +1.21 / +0.04 (τ0.2, γ0.9) | +1.26 / +0.03 (τ0.2, γ0.9) | +0.72 / +0.01 (τ0.1, γ1.01) | +2.29 / -2.20 |
| b | 86.12 / 78.63 | +1.11 / +0.01 (τ0.2, γ0.9) | +1.11 / +0.01 (τ0.2, γ1.01) | +1.02 / +0.01 (τ0.2, γ0.9) | +1.11 / +0.01 (τ0.2, γ0.9) | +0.74 / +0.01 (τ0.1, γ1.01) | +1.84 / -2.10 |
| off_T | 72.70 / 59.07 | +0.93 / -0.04 (τ0.2, γ0.9) | +0.93 / -0.04 (τ0.2, γ1.01) | +0.81 / -0.04 (τ0.2, γ0.9) | +0.93 / -0.04 (τ0.2, γ0.9) | +0.77 / +0.00 (τ0.1, γ1.01) | +1.89 / -1.87 |
| off_S | 74.61 / 61.12 | +1.15 / -0.09 (τ0.3, γ0.9) | +1.15 / -0.09 (τ0.3, γ1.01) | +1.03 / -0.08 (τ0.3, γ0.9) | +1.15 / -0.09 (τ0.3, γ0.9) | +0.66 / +0.03 (τ0.1, γ1.01) | +1.92 / -1.10 |
| off_B | 76.11 / 62.93 | +0.70 / -0.01 (τ0.2, γ0.9) | +0.70 / -0.01 (τ0.2, γ1.01) | +0.93 / -0.04 (τ0.3, γ0.9) | +0.70 / -0.01 (τ0.2, γ0.9) | +0.44 / +0.00 (τ0.1, γ1.01) | +1.80 / -0.70 |
| svtrv2_ch | 73.10 / 59.26 | infeasible | infeasible | infeasible | infeasible | +0.33 / +0.02 (τ0.05, γ1.01) | +0.27 / -3.65 |
| repsvtr_ch | 70.43 / 58.03 | infeasible | infeasible | infeasible | infeasible | +0.72 / +0.06 (τ0.05, γ1.01) | +0.00 / -4.52 |
| crnn_s1 | 71.25 / 61.72 | +1.93 / +0.10 (τ0.2, γ0.9) | +1.93 / +0.10 (τ0.2, γ1.01) | +2.27 / +0.03 (τ0.3, γ0.9) | +1.93 / +0.10 (τ0.2, γ0.9) | +1.99 / +0.07 (τ0.2, γ1.01) | +2.88 / -3.87 |
| crnn_s2 | 71.01 / 60.46 | +2.86 / +0.03 (τ0.3, γ0.9) | +2.86 / +0.03 (τ0.3, γ1.01) | +2.33 / +0.04 (τ0.3, γ0.9) | +2.82 / +0.03 (τ0.3, γ0.9) | +0.90 / -0.02 (τ0.1, γ1.01) | +3.00 / -3.81 |

Cells: rare-line gain / overall change (pp) of the val-selected config.

## gtc
- G*: CER 4.34 -> 4.35, NED 93.84 -> 93.82; rare-output precision 93.4 -> 90.6
- recall by bucket B0 -> G*: 0 0.0->0.0, 1-9 62.9->66.4, 10-99 87.6->88.9, 100-999 93.7->94.1, >=1000 97.2->97.1
- precision by bucket B0 -> G*: 0 0.0->0.0, 1-9 90.1->86.6, 10-99 93.6->90.9, 100-999 95.3->94.2, >=1000 97.3->97.4
- rare threshold (lines, B0 -> G*): <10: 765, 54.38->57.39, <50: 4021, 73.74->75.50, <100: 7670, 77.94->79.22, <500: 28735, 83.30->83.93
- look-alike share of 1235 rare substitutions (observed / random substitute): ids_shared_component 63.2/8.5, glyph_top10 39.0/0.1, glyph_top50 53.6/1.2, union_used_in_paper 71.6/9.6
- beta by min errors: N>=1: 13094 pairs, T/E 2.55, beta 1.22, N>=3: 420 pairs, T/E 2.36, beta 1.12, N>=5: 131 pairs, T/E 2.56, beta 1.14, N>=10: 28 pairs, T/E 3.65, beta 1.20
- beta with intercept 1.09 [1.0108719173569802, 1.16593908549855] (alpha 0.09 [-0.06980200284187033, 0.24786036920236695]); with IDS-complexity 1.14 [1.0833667085613288, 1.1962097134725085] (eta -0.069 [-0.17732384754943367, 0.020148038482197532])
- ceiling: 1249 rare substitutions, 63.5% recoverable, G* repairs 16.1% of those

## b
- G*: CER 4.28 -> 4.29, NED 93.96 -> 93.95; rare-output precision 93.6 -> 91.1
- recall by bucket B0 -> G*: 0 0.0->0.0, 1-9 64.2->67.2, 10-99 87.9->89.0, 100-999 93.8->94.2, >=1000 97.3->97.2
- precision by bucket B0 -> G*: 0 0.0->0.0, 1-9 90.4->87.3, 10-99 93.8->91.4, 100-999 95.4->94.4, >=1000 97.4->97.5
- rare threshold (lines, B0 -> G*): <10: 765, 55.82->58.69, <50: 4021, 74.53->75.98, <100: 7670, 78.63->79.74, <500: 28735, 83.76->84.31
- look-alike share of 1162 rare substitutions (observed / random substitute): ids_shared_component 64.4/7.5, glyph_top10 38.6/0.0, glyph_top50 53.7/0.9, union_used_in_paper 71.7/7.8
- beta by min errors: N>=1: 12558 pairs, T/E 2.54, beta 1.20, N>=3: 412 pairs, T/E 2.44, beta 1.09, N>=5: 128 pairs, T/E 2.63, beta 1.07, N>=10: 25 pairs, T/E 3.51, beta 1.16
- beta with intercept 1.04 [0.9563127168554756, 1.1116709365112143] (alpha 0.14 [-0.015581948978717168, 0.31771615045801976]); with IDS-complexity 1.10 [1.0425348206255547, 1.1497658211273842] (eta -0.037 [-0.11547441179665206, 0.06610062266206382])
- ceiling: 1178 rare substitutions, 63.6% recoverable, G* repairs 15.8% of those

## off_T
- G*: CER 8.57 -> 8.58, NED 90.18 -> 90.16; rare-output precision 93.9 -> 90.8
- recall by bucket B0 -> G*: 0 0.0->0.0, 1-9 61.0->66.0, 10-99 85.7->86.7, 100-999 91.1->91.4, >=1000 94.7->94.6
- precision by bucket B0 -> G*: 0 0.0->0.0, 1-9 92.0->87.1, 10-99 94.0->91.1, 100-999 95.2->94.1, >=1000 97.0->97.2
- rare threshold (lines, B0 -> G*): <10: 765, 36.34->40.13, <50: 4021, 54.56->55.96, <100: 7670, 59.07->60.00, <500: 28735, 65.98->66.34
- look-alike share of 1143 rare substitutions (observed / random substitute): ids_shared_component 62.2/7.5, glyph_top10 37.8/0.2, glyph_top50 51.4/1.0, union_used_in_paper 70.3/8.4
- beta by min errors: N>=1: 13678 pairs, T/E 2.47, beta 1.21, N>=3: 450 pairs, T/E 2.33, beta 1.14, N>=5: 132 pairs, T/E 2.37, beta 1.10, N>=10: 24 pairs, T/E 2.78, beta 1.20
- beta with intercept 1.12 [1.023163980409811, 1.2027559047806535] (alpha 0.05 [-0.10167774762882364, 0.2286910554122005]); with IDS-complexity 1.16 [1.0980762645859097, 1.2257001832609618] (eta -0.086 [-0.21416681271415206, 0.020564963590649705])
- ceiling: 1160 rare substitutions, 63.4% recoverable, G* repairs 16.5% of those

## off_S
- G*: CER 7.76 -> 7.77, NED 91.38 -> 91.34; rare-output precision 93.7 -> 91.3
- recall by bucket B0 -> G*: 0 0.0->0.0, 1-9 65.5->68.9, 10-99 86.6->87.4, 100-999 92.2->92.4, >=1000 95.4->95.3
- precision by bucket B0 -> G*: 0 0.0->0.0, 1-9 90.9->87.0, 10-99 93.9->91.7, 100-999 95.0->94.0, >=1000 97.2->97.3
- rare threshold (lines, B0 -> G*): <10: 765, 41.83->44.05, <50: 4021, 56.53->57.80, <100: 7670, 61.12->62.01, <500: 28735, 67.86->68.21
- look-alike share of 1120 rare substitutions (observed / random substitute): ids_shared_component 61.8/9.6, glyph_top10 37.6/0.2, glyph_top50 50.4/1.0, union_used_in_paper 68.8/10.4
- beta by min errors: N>=1: 13733 pairs, T/E 2.47, beta 1.17, N>=3: 394 pairs, T/E 2.46, beta 1.15, N>=5: 108 pairs, T/E 2.51, beta 1.17, N>=10: 17 pairs, T/E 2.85, beta 1.21
- beta with intercept 1.07 [0.9881967024319895, 1.1592653486562112] (alpha 0.22 [0.042955670967800945, 0.3846944894637587]); with IDS-complexity 1.14 [1.0711657295318018, 1.2122722948789535] (eta 0.045 [-0.09428762971842308, 0.18207301545485632])
- ceiling: 1137 rare substitutions, 62.5% recoverable, G* repairs 13.4% of those

## off_B
- G*: CER 7.04 -> 7.05, NED 92.36 -> 92.34; rare-output precision 93.6 -> 91.1
- recall by bucket B0 -> G*: 0 5.4->5.4, 1-9 69.5->72.5, 10-99 88.0->88.8, 100-999 93.2->93.3, >=1000 96.1->96.0
- precision by bucket B0 -> G*: 0 60.0->42.9, 1-9 90.0->86.7, 10-99 93.9->91.5, 100-999 95.5->94.7, >=1000 97.5->97.6
- rare threshold (lines, B0 -> G*): <10: 765, 45.36->47.58, <50: 4021, 58.97->59.89, <100: 7670, 62.93->63.64, <500: 28735, 69.44->69.77
- look-alike share of 1025 rare substitutions (observed / random substitute): ids_shared_component 60.4/7.7, glyph_top10 37.9/0.2, glyph_top50 50.5/0.9, union_used_in_paper 67.8/8.3
- beta by min errors: N>=1: 12126 pairs, T/E 2.45, beta 1.13, N>=3: 328 pairs, T/E 2.48, beta 1.09, N>=5: 93 pairs, T/E 2.98, beta 1.08, N>=10: 19 pairs, T/E 2.84, beta 1.18
- beta with intercept 1.02 [0.9271364500438902, 1.1078874877124247] (alpha 0.20 [0.03376820769907324, 0.4186174820673668]); with IDS-complexity 1.10 [1.0349146914901737, 1.1644825251939672] (eta -0.022 [-0.14563520895143076, 0.11425858006345375])
- ceiling: 1043 rare substitutions, 64.1% recoverable, G* repairs 12.9% of those

## svtrv2_ch
- G*: CER 10.60 -> 10.70, NED 86.62 -> 86.46; rare-output precision 93.9 -> 91.7
- recall by bucket B0 -> G*: 0 7.1->8.9, 1-9 33.8->37.6, 10-99 75.0->76.2, 100-999 86.0->86.3, >=1000 91.6->91.4
- precision by bucket B0 -> G*: 0 80.0->62.5, 1-9 88.6->84.6, 10-99 94.1->92.1, 100-999 94.4->93.3, >=1000 96.2->96.5
- rare threshold (lines, B0 -> G*): <10: 765, 26.54->29.80, <50: 4021, 52.70->54.14, <100: 7670, 59.26->60.21, <500: 28735, 67.42->67.63
- look-alike share of 1788 rare substitutions (observed / random substitute): ids_shared_component 62.2/7.8, glyph_top10 40.0/0.2, glyph_top50 56.0/1.0, union_used_in_paper 71.9/8.6
- beta by min errors: N>=1: 16229 pairs, T/E 2.62, beta 1.33, N>=3: 729 pairs, T/E 2.29, beta 1.26, N>=5: 230 pairs, T/E 2.08, beta 1.19, N>=10: 51 pairs, T/E 1.82, beta 1.07
- beta with intercept 1.26 [1.0787522312301918, 1.4531269810007348] (alpha 0.01 [-0.24085662359217272, 0.29547880311462565]); with IDS-complexity 1.26 [1.1319518879694652, 1.3777913462246625] (eta 0.007 [-0.10662054941575733, 0.12391035867853817])
- ceiling: 1848 rare substitutions, 50.2% recoverable, G* repairs 14.5% of those

## repsvtr_ch
- G*: CER 11.86 -> 11.98, NED 84.79 -> 84.62; rare-output precision 93.6 -> 90.4
- recall by bucket B0 -> G*: 0 8.9->8.9, 1-9 41.2->45.0, 10-99 74.2->75.9, 100-999 84.2->84.5, >=1000 90.4->90.2
- precision by bucket B0 -> G*: 0 83.3->71.4, 1-9 87.4->82.2, 10-99 93.9->91.0, 100-999 94.0->92.6, >=1000 96.0->96.4
- rare threshold (lines, B0 -> G*): <10: 765, 31.90->34.25, <50: 4021, 52.77->54.19, <100: 7670, 58.03->59.13, <500: 28735, 65.06->65.35
- look-alike share of 1670 rare substitutions (observed / random substitute): ids_shared_component 63.4/7.2, glyph_top10 39.5/0.2, glyph_top50 55.0/1.0, union_used_in_paper 73.1/7.7
- beta by min errors: N>=1: 16525 pairs, T/E 2.61, beta 1.31, N>=3: 807 pairs, T/E 2.30, beta 1.25, N>=5: 232 pairs, T/E 2.13, beta 1.19, N>=10: 50 pairs, T/E 1.78, beta 1.04
- beta with intercept 1.25 [1.1239041488948673, 1.4020567405742794] (alpha -0.00 [-0.23975522900332263, 0.22203913044950027]); with IDS-complexity 1.24 [1.1546146613820933, 1.3348399004080371] (eta 0.054 [-0.042762172579739854, 0.15967917800117035])
- ceiling: 1733 rare substitutions, 58.6% recoverable, G* repairs 17.5% of those

## crnn_s1
- G*: CER 13.20 -> 13.19, NED 82.10 -> 82.11; rare-output precision 92.0 -> 85.6
- recall by bucket B0 -> G*: 0 0.0->0.0, 1-9 45.4->50.3, 10-99 73.1->75.4, 100-999 81.0->81.7, >=1000 88.5->88.3
- precision by bucket B0 -> G*: 0 0.0->0.0, 1-9 91.5->82.0, 10-99 92.0->85.8, 100-999 92.0->89.0, >=1000 92.8->93.3
- rare threshold (lines, B0 -> G*): <10: 765, 37.91->41.70, <50: 4021, 56.73->59.14, <100: 7670, 61.72->63.65, <500: 28735, 67.36->68.25
- look-alike share of 1977 rare substitutions (observed / random substitute): ids_shared_component 46.6/7.2, glyph_top10 28.1/0.2, glyph_top50 41.4/1.2, union_used_in_paper 56.0/7.9
- beta by min errors: N>=1: 29567 pairs, T/E 2.61, beta 1.35, N>=3: 1454 pairs, T/E 2.06, beta 1.20, N>=5: 368 pairs, T/E 1.99, beta 1.18, N>=10: 67 pairs, T/E 1.80, beta 1.19
- beta with intercept 1.12 [1.0574494070793765, 1.1901040827145124] (alpha 0.16 [0.061672164354059936, 0.2641306195472979]); with IDS-complexity 1.24 [1.1994452202572323, 1.284070843138418] (eta -0.155 [-0.21238026109143399, -0.0943887726950779])
- ceiling: 2028 rare substitutions, 56.1% recoverable, G* repairs 20.0% of those

## crnn_s2
- G*: CER 13.07 -> 13.06, NED 82.41 -> 82.43; rare-output precision 92.3 -> 86.1
- recall by bucket B0 -> G*: 0 0.0->0.0, 1-9 46.3->50.8, 10-99 73.0->75.3, 100-999 81.3->82.0, >=1000 88.7->88.6
- precision by bucket B0 -> G*: 0 0.0->0.0, 1-9 93.0->86.4, 10-99 92.3->86.1, 100-999 92.2->89.3, >=1000 92.7->93.2
- rare threshold (lines, B0 -> G*): <10: 765, 37.78->41.57, <50: 4021, 55.48->58.12, <100: 7670, 60.46->62.53, <500: 28735, 66.70->67.62
- look-alike share of 2006 rare substitutions (observed / random substitute): ids_shared_component 48.8/8.2, glyph_top10 28.3/0.0, glyph_top50 42.0/0.6, union_used_in_paper 57.3/8.7
- beta by min errors: N>=1: 29843 pairs, T/E 2.63, beta 1.36, N>=3: 1509 pairs, T/E 2.07, beta 1.24, N>=5: 386 pairs, T/E 1.98, beta 1.22, N>=10: 64 pairs, T/E 1.97, beta 1.31
- beta with intercept 1.19 [1.109016076077808, 1.2580059624000297] (alpha 0.10 [-0.010014661085540014, 0.19179765706170884]); with IDS-complexity 1.27 [1.2212288945589245, 1.3182564011509272] (eta -0.140 [-0.2124116071339638, -0.08183021160579552])
- ceiling: 2053 rare substitutions, 55.4% recoverable, G* repairs 19.5% of those

