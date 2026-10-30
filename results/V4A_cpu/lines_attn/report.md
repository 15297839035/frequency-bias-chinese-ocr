# V4A lines: BCTR test, B0, official normalisation

| model | pairs | T/E [CI] | T/E Bayes | beta [CI] | beta Bayes (sim) | beta - Bayes [CI] | free alpha, beta |
|---|---:|---|---:|---|---|---|---|
| attn:transocr | 981 | 2.03 [1.93, 2.15] | 1.94 | 1.28 [1.23, 1.33] | 1.17 (1.17) | +0.11 [+0.06, +0.16] | 0.93, 1.21 |
| attn:aster | 1900 | 2.08 [2.00, 2.16] | 1.94 | 1.32 [1.28, 1.36] | 1.17 (1.17) | +0.16 [+0.11, +0.19] | 0.86, 1.18 |

## A1 decomposition (CTC; rare-character substitutions, all rarity bins)

| model | n | not cand. % | cand.+freq % | cand.+other % | repaired at tau 0.2 / 0.5 / 1 / 2 (% of all) |
|---|---:|---:|---:|---:|---|

## A5 look-alike share of rare-character substitutions (observed / random substitute, %)

- attn:transocr: ids_any 37.1/8.4; ids_jaccard50 6.9/0.3; glyph10 16.8/0.1; glyph50 28.6/1.2; union_paper 44.8/9.3; strict 20.1/0.4 | T/E by group: union_paper/lookalike 2.17 (422); union_paper/other 1.88 (530); strict/lookalike 2.18 (243); strict/other 1.95 (709)
- attn:transocr: ids_any 28.0/7.5; ids_jaccard50 5.7/0.2; glyph10 12.0/0.3; glyph50 20.7/1.4; union_paper 34.3/8.2; strict 14.6/0.4 | T/E by group: union_paper/lookalike 2.40 (11); union_paper/other nan (3); strict/lookalike 2.52 (10); strict/other nan (4)
- attn:transocr: ids_any 69.7/8.2; ids_jaccard50 13.3/0.2; glyph10 49.5/0.0; glyph50 69.2/0.3; union_paper 83.3/8.2; strict 52.2/0.2 | T/E by group: union_paper/lookalike 4.71 (15); union_paper/other nan (0); strict/lookalike 3.96 (13); strict/other nan (2)
- attn:aster: ids_any 31.9/7.1; ids_jaccard50 5.7/0.2; glyph10 13.3/0.2; glyph50 23.2/1.1; union_paper 39.1/7.8; strict 16.2/0.4 | T/E by group: union_paper/lookalike 2.41 (668); union_paper/other 1.87 (1180); strict/lookalike 2.44 (352); strict/other 1.97 (1496)
- attn:aster: ids_any 25.8/6.4; ids_jaccard50 4.0/0.3; glyph10 8.8/0.2; glyph50 17.6/0.9; union_paper 31.7/7.0; strict 11.2/0.6 | T/E by group: union_paper/lookalike 1.96 (28); union_paper/other nan (9); strict/lookalike 1.84 (20); strict/other 1.98 (17)
- attn:aster: ids_any 66.4/7.4; ids_jaccard50 10.7/0.3; glyph10 49.7/0.2; glyph50 69.1/0.8; union_paper 79.6/7.9; strict 52.3/0.3 | T/E by group: union_paper/lookalike 2.30 (13); union_paper/other nan (2); strict/lookalike 2.30 (13); strict/other nan (2)
