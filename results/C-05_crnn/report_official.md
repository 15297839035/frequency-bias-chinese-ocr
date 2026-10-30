## crnn_s1 (official normalisation)
- val B0 acc 71.685 rare 61.991; selected {'method': 'G', 'tau': 0.3, 'k': 0, 'gamma': 0.9}
- test B0: acc 71.649 rare 62.634 | domains scene 53.32, web 63.70, document 97.00
- test G_selected: acc 71.665 (+0.017) rare 65.176 (+2.542), fixed/broken 230/35, p=2.4e-36
- test G*: acc 71.746 (+0.098) rare 64.433 (+1.799), fixed/broken 156/18, p=1.3e-28
- val direction: pairs 1451, obs 55.9% vs sym 28.5%, T/E 1.96, z 50.8, beta 1.227 [1.176, 1.278]
- val over-production slope B0 0.0287 [0.0246, 0.0336], G* 0.0126
- test direction: pairs 1454, obs 57.5% vs sym 28.6%, T/E 2.01, z 53.3, beta 1.235 [1.184, 1.288]
- test over-production slope B0 0.0302 [0.0259, 0.0343], G* 0.0130
- criteria: (i) True  (ii) True  (iii) True

## crnn_s2 (official normalisation)
- val B0 acc 71.584 rare 61.591; selected {'method': 'G', 'tau': 0.3, 'k': 0, 'gamma': 0.9}
- test B0: acc 71.530 rare 61.512 | domains scene 54.07, web 64.13, document 95.63
- test G_selected: acc 71.571 (+0.042) rare 64.276 (+2.764), fixed/broken 245/33, p=3.4e-41
- test G*: acc 71.624 (+0.094) rare 63.533 (+2.021), fixed/broken 176/21, p=1.1e-31
- val direction: pairs 1443, obs 57.5% vs sym 28.6%, T/E 2.01, z 53.6, beta 1.267 [1.220, 1.315]
- val over-production slope B0 0.0292 [0.0253, 0.0333], G* 0.0144
- test direction: pairs 1501, obs 57.9% vs sym 28.8%, T/E 2.01, z 54.1, beta 1.285 [1.232, 1.335]
- test over-production slope B0 0.0320 [0.0275, 0.0364], G* 0.0140
- criteria: (i) True  (ii) True  (iii) True

**External validation passed (all models, all criteria): True**
