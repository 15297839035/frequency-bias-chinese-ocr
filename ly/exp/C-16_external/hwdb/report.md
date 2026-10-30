# C-16 (b, c): CTC models on handwritten lines (hwdb val / ICDAR 2013 test), official norm.

## off_T: test B0 acc 5.51, rare 4.40; samples {'val': 2165, 'test': 1217} (excluded {'val': 3058, 'test': 2215})
- val direction: pairs 199, T/E 1.85, z 20.5, beta 1.157 [1.001, 1.329]
- test direction: pairs 170, T/E 1.50, z 12.2, beta 0.869 [0.658, 1.120]
- bctr_prior G*: rare +0.00 (fixed/broken 0/0, p=1), overall -0.08
- bctr_prior G_selected: rare +0.00 (fixed/broken 0/0, p=1), overall -0.08, selected {'method': 'G', 'tau': 0.2, 'k': 0, 'gamma': 0.9}
- self_prior G*: rare +0.37 (fixed/broken 1/0, p=1), overall +0.00
- criterion (ii) direction: True

## off_S: test B0 acc 6.57, rare 2.93; samples {'val': 2165, 'test': 1217} (excluded {'val': 3058, 'test': 2215})
- val direction: pairs 168, T/E 1.83, z 18.2, beta 1.030 [0.862, 1.254]
- test direction: pairs 143, T/E 1.46, z 11.0, beta 0.860 [0.641, 1.122]
- bctr_prior G*: rare +0.73 (fixed/broken 2/0, p=0.5), overall +0.00
- bctr_prior G_selected: rare +0.73 (fixed/broken 2/0, p=0.5), overall +0.00, selected {'method': 'G', 'tau': 0.2, 'k': 0, 'gamma': 0.9}
- self_prior G*: rare +0.00 (fixed/broken 0/0, p=1), overall +0.00
- criterion (ii) direction: False

## off_B: test B0 acc 9.78, rare 5.86; samples {'val': 2165, 'test': 1217} (excluded {'val': 3058, 'test': 2215})
- val direction: pairs 138, T/E 1.85, z 16.4, beta 1.043 [0.906, 1.221]
- test direction: pairs 119, T/E 1.50, z 10.8, beta 0.798 [0.568, 1.092]
- bctr_prior G*: rare +0.37 (fixed/broken 2/1, p=1), overall +0.25
- bctr_prior G_selected: rare +0.73 (fixed/broken 2/0, p=0.5), overall +0.25, selected {'method': 'G', 'tau': 0.1, 'k': 0, 'gamma': 0.9}
- self_prior G*: rare +0.37 (fixed/broken 2/1, p=1), overall +0.00
- self_prior G_selected: rare +0.73 (fixed/broken 2/0, p=0.5), overall +0.16, selected {'method': 'G', 'tau': 0.1, 'k': 0, 'gamma': 0.9}
- criterion (ii) direction: False

## gtc: test B0 acc 0.49, rare 0.37; samples {'val': 2165, 'test': 1217} (excluded {'val': 3058, 'test': 2215})
- val direction: pairs 480, T/E 1.65, z 22.9, beta 0.782 [0.697, 0.865]
- test direction: pairs 276, T/E 1.55, z 16.9, beta 0.793 [0.675, 0.927]
- bctr_prior G*: rare +0.00 (fixed/broken 0/0, p=1), overall +0.00
- bctr_prior G_selected: rare +0.00 (fixed/broken 0/0, p=1), overall +0.00, selected {'method': 'G', 'tau': 0.1, 'k': 0, 'gamma': 0.9}
- self_prior G*: rare +0.00 (fixed/broken 0/0, p=1), overall +0.08
- self_prior G_selected: rare +0.00 (fixed/broken 0/0, p=1), overall +0.00, selected {'method': 'G', 'tau': 0.1, 'k': 0, 'gamma': 0.9}
- criterion (ii) direction: True

## b: test B0 acc 0.66, rare 0.37; samples {'val': 2165, 'test': 1217} (excluded {'val': 3058, 'test': 2215})
- val direction: pairs 499, T/E 1.79, z 27.8, beta 0.934 [0.852, 1.017]
- test direction: pairs 331, T/E 1.61, z 19.5, beta 0.919 [0.822, 1.017]
- bctr_prior G*: rare +0.00 (fixed/broken 0/0, p=1), overall -0.16
- self_prior G*: rare +0.00 (fixed/broken 0/0, p=1), overall -0.16
- self_prior G_selected: rare +0.00 (fixed/broken 0/0, p=1), overall -0.16, selected {'method': 'G', 'tau': 0.1, 'k': 0, 'gamma': 0.9}
- criterion (ii) direction: True

## crnn_s1: test B0 acc 0.08, rare 0.00; samples {'val': 2165, 'test': 1217} (excluded {'val': 3058, 'test': 2215})
- val direction: pairs 646, T/E 1.65, z 26.5, beta 0.820 [0.745, 0.895]
- test direction: pairs 399, T/E 1.56, z 20.2, beta 0.859 [0.751, 0.982]
- bctr_prior G*: rare +0.37 (fixed/broken 1/0, p=1), overall +0.08
- bctr_prior G_selected: rare +0.37 (fixed/broken 1/0, p=1), overall +0.08, selected {'method': 'G', 'tau': 0.1, 'k': 0, 'gamma': 0.9}
- self_prior G*: rare +0.37 (fixed/broken 1/0, p=1), overall +0.08
- criterion (ii) direction: True

## crnn_s2: test B0 acc 0.00, rare 0.00; samples {'val': 2165, 'test': 1217} (excluded {'val': 3058, 'test': 2215})
- val direction: pairs 633, T/E 1.69, z 27.3, beta 0.808 [0.737, 0.887]
- test direction: pairs 400, T/E 1.61, z 22.0, beta 0.936 [0.826, 1.053]
- bctr_prior G*: rare +0.00 (fixed/broken 0/0, p=1), overall +0.00
- bctr_prior G_selected: rare +0.00 (fixed/broken 0/0, p=1), overall +0.00, selected {'method': 'G', 'tau': 0.2, 'k': 0, 'gamma': 0.9}
- self_prior G*: rare +0.00 (fixed/broken 0/0, p=1), overall +0.00
- self_prior G_selected: rare +0.00 (fixed/broken 0/0, p=1), overall +0.00, selected {'method': 'G', 'tau': 0.1, 'k': 0, 'gamma': 0.9}
- criterion (ii) direction: True

