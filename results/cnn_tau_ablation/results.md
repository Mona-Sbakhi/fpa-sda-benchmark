# Results: cnn_tau_ablation

Hybrid transition point on the CNN-head task (20 runs per setting).

Test-set metrics, mean ± SD [95 % bootstrap CI of the mean] over runs. Percentages except MCC/AUC.

| Algorithm | Runs | Accuracy | Balanced acc. | Precision | Recall | Specificity | F1 | MCC | AUC | Time (s) | Evals |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Hybrid-tau25 | 20 | 71.62 ± 2.31 [70.67, 72.63] | 68.10 ± 2.33 [67.11, 69.10] | 74.92 ± 1.82 [74.15, 75.70] | 82.17 ± 4.24 [80.41, 84.03] | 54.04 ± 5.21 [51.82, 56.30] | 78.31 ± 2.06 [77.47, 79.22] | 0.38 ± 0.05 [0.36, 0.40] | 0.77 ± 0.04 [0.76, 0.79] | 40.2 | 6795 |
| Hybrid-tau50 | 20 | 70.77 ± 2.16 [69.83, 71.70] | 66.83 ± 2.84 [65.61, 68.07] | 73.89 ± 2.42 [72.85, 74.93] | 82.58 ± 3.43 [81.08, 84.05] | 51.09 ± 7.12 [48.03, 54.19] | 77.92 ± 1.60 [77.23, 78.60] | 0.36 ± 0.05 [0.33, 0.38] | 0.77 ± 0.04 [0.75, 0.78] | 40.5 | 6795 |
| Hybrid-tau75 | 20 | 70.03 ± 4.43 [68.21, 72.00] | 66.37 ± 4.62 [64.46, 68.40] | 73.77 ± 3.33 [72.38, 75.25] | 81.01 ± 6.68 [78.09, 83.74] | 51.73 ± 8.86 [47.78, 55.43] | 77.08 ± 3.86 [75.47, 78.72] | 0.35 ± 0.09 [0.31, 0.39] | 0.75 ± 0.05 [0.72, 0.77] | 40.8 | 6795 |
| Hybrid-tau100 | 20 | 66.82 ± 5.81 [64.13, 69.05] | 63.96 ± 5.68 [61.42, 66.19] | 72.60 ± 4.24 [70.73, 74.28] | 75.38 ± 8.67 [71.32, 78.54] | 52.54 ± 9.36 [48.46, 56.28] | 73.75 ± 5.94 [70.92, 75.90] | 0.29 ± 0.11 [0.23, 0.33] | 0.71 ± 0.07 [0.68, 0.74] | 41.0 | 6795 |
| Hybrid-tau125 | 20 | 64.29 ± 4.92 [62.13, 66.31] | 61.32 ± 4.46 [59.37, 63.21] | 70.88 ± 3.75 [69.30, 72.51] | 73.21 ± 10.13 [68.79, 77.36] | 49.44 ± 10.57 [45.28, 54.21] | 71.63 ± 5.53 [69.20, 73.87] | 0.24 ± 0.09 [0.19, 0.27] | 0.68 ± 0.07 [0.66, 0.71] | 41.3 | 6795 |

Data: {"n_opt": 4186, "n_val": 1046, "n_test": 624, "opt_class_counts": [1079, 3107], "val_class_counts": [270, 776], "test_class_counts": [234, 390]}

## Statistical tests (gradient-free optimisers, paired by run seed)

### mcc

Friedman χ² = 29.520, p = 6.13e-06, blocks = 20.

Mean rank (1 = best): Hybrid-tau25 1.95, Hybrid-tau50 2.35, Hybrid-tau75 3.00, Hybrid-tau100 3.25, Hybrid-tau125 4.45

Wilcoxon signed-rank, Hybrid-tau75 vs each (Holm-adjusted); A = Vargha–Delaney (>0.5 favours Hybrid-tau75).

| vs | median diff | p | p (Holm) | A |
|---|---|---|---|---|
| Hybrid-tau25 | -0.0626 | 0.0328 | 0.0983 | 0.340 |
| Hybrid-tau50 | -0.0376 | 0.312 | 0.555 | 0.420 |
| Hybrid-tau100 | -0.0052 | 0.277 | 0.555 | 0.583 |
| Hybrid-tau125 | +0.0969 | 0.000395 | 0.00158 | 0.797 |

### accuracy

Friedman χ² = 36.010, p = 2.88e-07, blocks = 20.

Mean rank (1 = best): Hybrid-tau25 1.88, Hybrid-tau50 2.25, Hybrid-tau75 2.90, Hybrid-tau100 3.40, Hybrid-tau125 4.58

Wilcoxon signed-rank, Hybrid-tau75 vs each (Holm-adjusted); A = Vargha–Delaney (>0.5 favours Hybrid-tau75).

| vs | median diff | p | p (Holm) | A |
|---|---|---|---|---|
| Hybrid-tau25 | -0.0232 | 0.0702 | 0.211 | 0.347 |
| Hybrid-tau50 | -0.0160 | 0.232 | 0.301 | 0.404 |
| Hybrid-tau100 | +0.0072 | 0.151 | 0.301 | 0.605 |
| Hybrid-tau125 | +0.0433 | 0.00068 | 0.00272 | 0.825 |

### auc

Friedman χ² = 31.440, p = 2.49e-06, blocks = 20.

Mean rank (1 = best): Hybrid-tau25 2.05, Hybrid-tau50 2.05, Hybrid-tau75 3.00, Hybrid-tau100 3.55, Hybrid-tau125 4.35

Wilcoxon signed-rank, Hybrid-tau75 vs each (Holm-adjusted); A = Vargha–Delaney (>0.5 favours Hybrid-tau75).

| vs | median diff | p | p (Holm) | A |
|---|---|---|---|---|
| Hybrid-tau25 | -0.0268 | 0.0973 | 0.248 | 0.338 |
| Hybrid-tau50 | -0.0269 | 0.0826 | 0.248 | 0.365 |
| Hybrid-tau100 | +0.0148 | 0.0826 | 0.248 | 0.610 |
| Hybrid-tau125 | +0.0369 | 0.00169 | 0.00676 | 0.780 |

### balanced_accuracy

Friedman χ² = 24.760, p = 5.62e-05, blocks = 20.

Mean rank (1 = best): Hybrid-tau25 2.00, Hybrid-tau50 2.50, Hybrid-tau75 3.00, Hybrid-tau100 3.15, Hybrid-tau125 4.35

Wilcoxon signed-rank, Hybrid-tau75 vs each (Holm-adjusted); A = Vargha–Delaney (>0.5 favours Hybrid-tau75).

| vs | median diff | p | p (Holm) | A |
|---|---|---|---|---|
| Hybrid-tau25 | -0.0306 | 0.0362 | 0.109 | 0.351 |
| Hybrid-tau50 | -0.0120 | 0.37 | 0.737 | 0.431 |
| Hybrid-tau100 | -0.0026 | 0.368 | 0.737 | 0.554 |
| Hybrid-tau125 | +0.0393 | 0.00169 | 0.00676 | 0.785 |
