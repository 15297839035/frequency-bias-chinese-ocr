## svtrv2_ch (official normalisation)
- val B0 acc 77.763 rare 64.600; selected {'method': 'G', 'tau': 0.1, 'k': 0, 'gamma': 0.9}
- test B0: acc 77.779 rare 65.580 | domains scene 66.62, web 76.01, document 92.36
- test G_selected: acc 77.766 (-0.013) rare 65.997 (+0.417), fixed/broken 44/12, p=2.1e-05
- test G*: acc 77.665 (-0.113) rare 66.506 (+0.926), fixed/broken 93/22, p=1.4e-11
- val direction: pairs 697, obs 56.7% vs sym 26.8%, T/E 2.12, z 41.1, beta 1.186 [1.061, 1.318]
- val over-production slope B0 0.0275 [0.0206, 0.0344], G* 0.0203
- test direction: pairs 684, obs 55.9% vs sym 27.8%, T/E 2.01, z 38.0, beta 1.171 [1.039, 1.303]
- test over-production slope B0 0.0249 [0.0195, 0.0306], G* 0.0192
- criteria: (i) False  (ii) True  (iii) True

## repsvtr_ch (official normalisation)
- val B0 acc 75.080 rare 63.606; selected {'method': 'G', 'tau': 0.1, 'k': 0, 'gamma': 0.9}
- test B0: acc 74.894 rare 64.042 | domains scene 63.02, web 73.10, document 90.38
- test G_selected: acc 74.912 (+0.018) rare 64.746 (+0.704), fixed/broken 65/11, p=1.8e-10
- test G*: acc 74.816 (-0.078) rare 65.280 (+1.239), fixed/broken 124/29, p=3.4e-15
- val direction: pairs 787, obs 55.2% vs sym 26.8%, T/E 2.06, z 41.4, beta 1.139 [1.049, 1.237]
- val over-production slope B0 0.0259 [0.0195, 0.0321], G* 0.0185
- test direction: pairs 770, obs 56.0% vs sym 27.7%, T/E 2.02, z 39.9, beta 1.177 [1.084, 1.280]
- test over-production slope B0 0.0230 [0.0179, 0.0283], G* 0.0135
- criteria: (i) True  (ii) True  (iii) True

**External validation passed (all models, all criteria): False**
