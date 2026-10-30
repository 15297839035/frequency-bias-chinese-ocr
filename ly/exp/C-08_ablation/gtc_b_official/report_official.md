## gtc (official normalisation)
- val B0 acc 86.478 rare 79.285; selected {'method': 'G', 'tau': 0.2, 'k': 0, 'gamma': 0.9}
- test B0: acc 86.394 rare 79.791 | domains scene 77.89, web 79.03, document 99.17
- test G_selected: acc 86.431 (+0.038) rare 80.860 (+1.069), fixed/broken 94/12, p=6.2e-17
- test G*: acc 86.431 (+0.038) rare 80.860 (+1.069), fixed/broken 94/12, p=6.2e-17
- val direction: pairs 381, obs 55.6% vs sym 27.5%, T/E 2.02, z 27.2, beta 1.170 [1.090, 1.257]
- val over-production slope B0 0.0129 [0.0101, 0.0156], G* 0.0053
- test direction: pairs 373, obs 54.5% vs sym 27.7%, T/E 1.97, z 25.8, beta 1.171 [1.097, 1.250]
- test over-production slope B0 0.0135 [0.0110, 0.0161], G* 0.0067
- criteria: (i) True  (ii) True  (iii) True

## b (official normalisation)
- val B0 acc 86.693 rare 79.866; selected {'method': 'G', 'tau': 0.2, 'k': 0, 'gamma': 0.9}
- test B0: acc 86.654 rare 80.548 | domains scene 78.37, web 79.10, document 99.21
- test G_selected: acc 86.673 (+0.019) rare 81.499 (+0.952), fixed/broken 87/14, p=4.8e-14
- test G*: acc 86.673 (+0.019) rare 81.499 (+0.952), fixed/broken 87/14, p=4.8e-14
- val direction: pairs 354, obs 54.7% vs sym 27.1%, T/E 2.02, z 25.9, beta 1.140 [1.062, 1.221]
- val over-production slope B0 0.0122 [0.0096, 0.0149], G* 0.0056
- test direction: pairs 362, obs 55.1% vs sym 27.1%, T/E 2.04, z 26.6, beta 1.152 [1.078, 1.230]
- test over-production slope B0 0.0139 [0.0116, 0.0163], G* 0.0083
- criteria: (i) True  (ii) True  (iii) True

**External validation passed (all models, all criteria): True**
