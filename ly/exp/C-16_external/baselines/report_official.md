## transocr SMTR (official normalisation)
- val B0 acc 79.677 rare 63.348; selected t0.5_g0.9
- test B0: acc 79.379 rare 63.833 cer 8.952 | scene 68.22, web 63.51, document 97.87
- test selected: acc 79.692 (+0.313) rare 68.696 (+4.863), fixed/broken 411/38, p=3.5e-80
- test G*: acc 79.663 (+0.283) rare 66.428 (+2.595), fixed/broken 213/14, p=7.3e-47
- val direction: pairs 1093, T/E 2.03, z 45.7, beta 1.238 [1.193, 1.293]; over-production B0 0.0506 [0.0454, 0.0550] -> G* 0.0346
- test direction: pairs 1139, T/E 2.00, z 45.2, beta 1.271 [1.226, 1.318]; over-production B0 0.0517 [0.0466, 0.0572] -> G* 0.0366
- criteria: (i) True  (ii) True  (iii) True

## aster SMTR (official normalisation)
- val B0 acc 74.651 rare 58.724; selected t0.3_g0.9
- test B0: acc 74.513 rare 59.765 cer 12.861 | scene 59.72, web 58.58, document 97.61
- test selected: acc 74.666 (+0.153) rare 62.282 (+2.516), fixed/broken 213/20, p=6.3e-42
- test G*: acc 74.705 (+0.192) rare 61.682 (+1.917), fixed/broken 161/14, p=7.8e-33
- val direction: pairs 2147, T/E 1.96, z 61.3, beta 1.239 [1.201, 1.276]; over-production B0 0.0498 [0.0430, 0.0558] -> G* 0.0291
- test direction: pairs 2163, T/E 2.04, z 67.0, beta 1.293 [1.257, 1.332]; over-production B0 0.0511 [0.0444, 0.0576] -> G* 0.0323
- criteria: (i) True  (ii) True  (iii) True

