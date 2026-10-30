## crnn_s1 (strict normalisation)
- val B0 acc 71.298 rare 61.010; selected {'method': 'G', 'tau': 0.2, 'k': 0, 'gamma': 0.9}
- test B0: acc 71.251 rare 61.721 | domains scene 52.84, web 63.05, document 96.77
- test G_selected: acc 71.349 (+0.098) rare 63.651 (+1.930), fixed/broken 164/16, p=4.2e-32
- test G*: acc 71.349 (+0.098) rare 63.651 (+1.930), fixed/broken 164/16, p=4.2e-32
- val direction: pairs 1470, obs 55.9% vs sym 27.5%, T/E 2.03, z 54.0, beta 1.183 [1.139, 1.228]
- val over-production slope B0 0.0364 [0.0319, 0.0416], G* 0.0163
- test direction: pairs 1454, obs 57.2% vs sym 27.8%, T/E 2.06, z 55.4, beta 1.200 [1.149, 1.247]
- test over-production slope B0 0.0369 [0.0322, 0.0417], G* 0.0158
- criteria: (i) True  (ii) True  (iii) True

## crnn_s2 (strict normalisation)
- val B0 acc 71.098 rare 60.468; selected {'method': 'G', 'tau': 0.3, 'k': 0, 'gamma': 0.9}
- test B0: acc 71.012 rare 60.456 | domains scene 53.56, web 63.41, document 95.16
- test G_selected: acc 71.038 (+0.025) rare 63.312 (+2.855), fixed/broken 251/32, p=2.6e-43
- test G*: acc 71.096 (+0.083) rare 62.529 (+2.073), fixed/broken 179/20, p=4.1e-33
- val direction: pairs 1448, obs 57.4% vs sym 27.7%, T/E 2.07, z 56.4, beta 1.226 [1.179, 1.275]
- val over-production slope B0 0.0363 [0.0317, 0.0409], G* 0.0163
- test direction: pairs 1509, obs 57.7% vs sym 27.8%, T/E 2.07, z 56.9, beta 1.238 [1.193, 1.284]
- test over-production slope B0 0.0399 [0.0349, 0.0448], G* 0.0177
- criteria: (i) True  (ii) True  (iii) True

**External validation passed (all models, all criteria): True**
