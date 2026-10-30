# V4A lines: BCTR test, B0, official normalisation

| model | pairs | T/E [CI] | T/E Bayes | beta [CI] | beta Bayes (sim) | beta - Bayes [CI] | free alpha, beta |
|---|---:|---|---:|---|---|---|---|
| ctc:gtc | 373 | 1.97 [1.82, 2.13] | 1.97 | 1.17 [1.10, 1.25] | 1.18 (1.18) | -0.01 [-0.08, +0.07] | 0.37, 0.54 |
| ctc:b | 362 | 2.04 [1.88, 2.21] | 2.02 | 1.15 [1.08, 1.22] | 1.18 (1.18) | -0.03 [-0.10, +0.05] | 0.51, 0.66 |
| ctc:off_T | 406 | 2.04 [1.88, 2.24] | 2.00 | 1.22 [1.15, 1.30] | 1.18 (1.18) | +0.05 [-0.03, +0.12] | 0.53, 0.76 |
| ctc:off_S | 357 | 2.15 [1.96, 2.36] | 2.08 | 1.18 [1.10, 1.27] | 1.20 (1.21) | -0.02 [-0.11, +0.07] | 0.13, 0.32 |
| ctc:off_B | 282 | 2.03 [1.84, 2.23] | 2.01 | 1.15 [1.07, 1.24] | 1.20 (1.20) | -0.04 [-0.13, +0.05] | 1.03, 1.18 |
| ctc:crnn_s1 | 1454 | 2.01 [1.92, 2.09] | 1.87 | 1.23 [1.18, 1.29] | 1.15 (1.15) | +0.09 [+0.04, +0.14] | 0.70, 0.93 |
| ctc:crnn_s2 | 1501 | 2.01 [1.92, 2.11] | 1.86 | 1.28 [1.23, 1.34] | 1.14 (1.14) | +0.15 [+0.09, +0.20] | 0.94, 1.22 |
| ctc:svtrv2_ch | 684 | 2.01 [1.82, 2.22] | 2.00 | 1.17 [1.03, 1.31] | 1.22 (1.22) | -0.05 [-0.20, +0.10] | 0.88, 1.05 |
| ctc:repsvtr_ch | 770 | 2.02 [1.83, 2.20] | 2.00 | 1.18 [1.08, 1.28] | 1.20 (1.20) | -0.02 [-0.13, +0.09] | 0.89, 1.07 |
| smtr:gtc | 388 | 2.20 [2.03, 2.40] | 2.06 | 1.27 [1.21, 1.34] | 1.19 (1.19) | +0.08 [+0.00, +0.15] | 0.62, 0.89 |
| smtr:b | 380 | 2.02 [1.87, 2.18] | 1.94 | 1.19 [1.12, 1.26] | 1.19 (1.19) | -0.00 [-0.08, +0.09] | 0.53, 0.72 |
| smtr:off_T | 585 | 1.95 [1.82, 2.09] | 1.93 | 1.16 [1.08, 1.23] | 1.17 (1.17) | -0.02 [-0.10, +0.06] | 1.05, 1.20 |
| smtr:off_S | 422 | 2.02 [1.86, 2.21] | 2.00 | 1.15 [1.08, 1.23] | 1.20 (1.20) | -0.05 [-0.13, +0.04] | 1.19, 1.34 |
| smtr:off_B | 335 | 2.02 [1.85, 2.22] | 1.99 | 1.17 [1.09, 1.26] | 1.17 (1.17) | -0.00 [-0.08, +0.09] | 1.56, 1.73 |

## A1 decomposition (CTC; rare-character substitutions, all rarity bins)

| model | n | not cand. % | cand.+freq % | cand.+other % | repaired at tau 0.2 / 0.5 / 1 / 2 (% of all) |
|---|---:|---:|---:|---:|---|
| ctc:gtc | 1054 | 38.7 | 58.7 | 2.6 | 10.1 / 21.9 / 33.9 / 41.9 |
| ctc:b | 972 | 38.8 | 58.0 | 3.2 | 9.5 / 22.0 / 33.1 / 40.6 |
| ctc:off_T | 989 | 39.3 | 57.9 | 2.7 | 9.6 / 20.9 / 34.0 / 41.9 |
| ctc:off_S | 947 | 41.0 | 55.4 | 3.6 | 7.4 / 18.2 / 30.0 / 38.5 |
| ctc:off_B | 850 | 40.6 | 56.2 | 3.2 | 6.9 / 16.9 / 27.9 / 34.2 |
| ctc:crnn_s1 | 1881 | 43.8 | 54.4 | 1.8 | 11.3 / 24.9 / 36.9 / 41.4 |
| ctc:crnn_s2 | 1901 | 44.6 | 53.7 | 1.7 | 11.0 / 24.5 / 36.0 / 40.3 |
| ctc:svtrv2_ch | 1579 | 53.5 | 44.1 | 2.4 | 6.6 / 13.9 / 20.5 / 25.8 |
| ctc:repsvtr_ch | 1469 | 43.8 | 53.2 | 3.0 | 10.1 / 20.9 / 31.2 / 33.9 |

## A5 look-alike share of rare-character substitutions (observed / random substitute, %)

- ctc:gtc: ids_any 61.0/6.9; ids_jaccard50 18.2/0.1; glyph10 33.4/0.2; glyph50 48.2/1.2; union_paper 69.1/7.8; strict 41.1/0.4 | T/E by group: union_paper/lookalike 2.00 (272); union_paper/other 1.89 (101); strict/lookalike 2.06 (204); strict/other 1.85 (169)
- ctc:b: ids_any 61.9/6.6; ids_jaccard50 18.5/0.3; glyph10 32.1/0.4; glyph50 47.4/2.2; union_paper 68.9/8.1; strict 40.8/0.6 | T/E by group: union_paper/lookalike 2.05 (281); union_paper/other 1.99 (81); strict/lookalike 2.10 (196); strict/other 1.95 (166)
- ctc:off_T: ids_any 61.0/6.7; ids_jaccard50 19.0/0.2; glyph10 33.9/0.4; glyph50 47.7/1.6; union_paper 69.0/8.0; strict 42.2/0.5 | T/E by group: union_paper/lookalike 2.06 (272); union_paper/other 1.98 (134); strict/lookalike 2.02 (175); strict/other 2.05 (231)
- ctc:off_S: ids_any 58.5/7.9; ids_jaccard50 18.1/0.1; glyph10 31.3/0.1; glyph50 44.8/0.6; union_paper 65.8/8.3; strict 40.3/0.3 | T/E by group: union_paper/lookalike 2.16 (213); union_paper/other 2.14 (144); strict/lookalike 2.15 (156); strict/other 2.16 (201)
- ctc:off_B: ids_any 58.0/8.0; ids_jaccard50 17.6/0.3; glyph10 31.3/0.6; glyph50 44.4/2.1; union_paper 65.2/8.7; strict 40.1/0.7 | T/E by group: union_paper/lookalike 2.08 (176); union_paper/other 1.91 (106); strict/lookalike 2.09 (132); strict/other 1.96 (150)
- ctc:crnn_s1: ids_any 47.1/7.1; ids_jaccard50 10.1/0.1; glyph10 27.2/0.1; glyph50 40.9/0.9; union_paper 56.7/8.0; strict 30.9/0.3 | T/E by group: union_paper/lookalike 2.20 (625); union_paper/other 1.84 (829); strict/lookalike 2.19 (361); strict/other 1.93 (1093)
- ctc:crnn_s2: ids_any 49.1/8.1; ids_jaccard50 11.6/0.2; glyph10 26.5/0.2; glyph50 41.2/0.9; union_paper 57.5/8.8; strict 31.6/0.4 | T/E by group: union_paper/lookalike 2.19 (687); union_paper/other 1.84 (814); strict/lookalike 2.21 (369); strict/other 1.93 (1132)
- ctc:svtrv2_ch: ids_any 63.2/8.3; ids_jaccard50 16.6/0.4; glyph10 38.6/0.1; glyph50 54.9/1.1; union_paper 72.3/9.1; strict 45.2/0.5 | T/E by group: union_paper/lookalike 2.03 (453); union_paper/other 1.99 (231); strict/lookalike 2.01 (295); strict/other 2.02 (389)
- ctc:repsvtr_ch: ids_any 63.3/8.6; ids_jaccard50 17.5/0.1; glyph10 38.2/0.2; glyph50 53.3/1.1; union_paper 72.1/9.3; strict 45.0/0.2 | T/E by group: union_paper/lookalike 2.00 (529); union_paper/other 2.08 (241); strict/lookalike 1.99 (324); strict/other 2.05 (446)
- smtr:gtc: ids_any 61.6/7.2; ids_jaccard50 16.8/0.2; glyph10 35.6/0.3; glyph50 51.8/0.7; union_paper 71.0/7.3; strict 41.9/0.5 | T/E by group: union_paper/lookalike 2.23 (284); union_paper/other 2.12 (104); strict/lookalike 2.22 (205); strict/other 2.18 (183)
- smtr:b: ids_any 63.6/7.5; ids_jaccard50 18.2/0.2; glyph10 35.6/0.3; glyph50 51.7/1.2; union_paper 71.8/8.3; strict 42.9/0.5 | T/E by group: union_paper/lookalike 2.06 (264); union_paper/other 1.92 (116); strict/lookalike 2.12 (186); strict/other 1.90 (194)
- smtr:off_T: ids_any 58.3/5.4; ids_jaccard50 16.4/0.3; glyph10 32.4/0.1; glyph50 46.8/0.5; union_paper 65.7/5.7; strict 39.8/0.4 | T/E by group: union_paper/lookalike 2.07 (315); union_paper/other 1.78 (270); strict/lookalike 2.01 (204); strict/other 1.91 (381)
- smtr:off_S: ids_any 60.4/7.1; ids_jaccard50 17.3/0.2; glyph10 33.6/0.3; glyph50 47.9/1.0; union_paper 68.6/7.5; strict 41.2/0.5 | T/E by group: union_paper/lookalike 2.11 (246); union_paper/other 1.89 (176); strict/lookalike 2.17 (172); strict/other 1.90 (250)
- smtr:off_B: ids_any 57.5/7.4; ids_jaccard50 18.6/0.1; glyph10 30.9/0.3; glyph50 43.0/1.4; union_paper 64.7/8.6; strict 39.4/0.4 | T/E by group: union_paper/lookalike 2.14 (194); union_paper/other 1.84 (141); strict/lookalike 2.09 (131); strict/other 1.97 (204)
