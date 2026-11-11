# Results: cnn_redesign_ablation

APH component ablations and cross-field gradient-free baselines (20 runs).

Test-set metrics, mean ± SD [95 % bootstrap CI of the mean] over runs. Percentages except MCC/AUC.

| Algorithm | Runs | Accuracy | Balanced acc. | Precision | Recall | Specificity | F1 | MCC | AUC | Time (s) | Evals |
|---|---|---|---|---|---|---|---|---|---|---|---|
| APH-uniform-ops | 20 | 79.48 ± 2.50 [78.38, 80.49] | 75.57 ± 2.87 [74.31, 76.74] | 79.19 ± 2.15 [78.26, 80.07] | 91.19 ± 2.49 [90.01, 92.13] | 59.96 ± 5.14 [57.74, 62.05] | 84.75 ± 1.84 [83.92, 85.48] | 0.55 ± 0.06 [0.53, 0.58] | 0.88 ± 0.02 [0.87, 0.89] | 45.0 | 6795 |
| APH-no-crossover | 20 | 79.42 ± 1.69 [78.72, 80.14] | 75.65 ± 2.02 [74.80, 76.50] | 79.35 ± 1.60 [78.68, 80.01] | 90.72 ± 1.71 [89.99, 91.45] | 60.58 ± 3.87 [58.93, 62.16] | 84.64 ± 1.22 [84.14, 85.16] | 0.55 ± 0.04 [0.53, 0.57] | 0.88 ± 0.02 [0.87, 0.88] | 49.2 | 6795 |
| APH-plant-only | 20 | 71.51 ± 3.88 [69.82, 73.14] | 68.56 ± 4.13 [66.72, 70.26] | 75.65 ± 3.01 [74.29, 76.86] | 80.38 ± 4.67 [78.45, 82.45] | 56.73 ± 6.86 [53.68, 59.44] | 77.88 ± 3.12 [76.55, 79.21] | 0.38 ± 0.09 [0.34, 0.42] | 0.77 ± 0.05 [0.75, 0.79] | 39.1 | 6795 |
| OpenAI-ES | 20 | 72.68 ± 3.11 [71.31, 73.90] | 69.18 ± 3.58 [67.61, 70.62] | 75.65 ± 2.83 [74.44, 76.84] | 83.18 ± 3.45 [81.69, 84.64] | 55.17 ± 6.71 [52.33, 58.03] | 79.19 ± 2.34 [78.17, 80.12] | 0.40 ± 0.07 [0.37, 0.43] | 0.78 ± 0.05 [0.76, 0.80] | 34.2 | 6795 |
| ZO-SGD | 20 | 71.92 ± 3.38 [70.44, 73.30] | 68.54 ± 4.10 [66.76, 70.24] | 75.41 ± 3.46 [73.94, 76.91] | 82.06 ± 3.95 [80.33, 83.72] | 55.02 ± 8.48 [51.45, 58.74] | 78.51 ± 2.50 [77.40, 79.50] | 0.39 ± 0.08 [0.35, 0.42] | 0.77 ± 0.04 [0.75, 0.79] | 33.7 | 6795 |

Data: {"n_opt": 4186, "n_val": 1046, "n_test": 624, "opt_class_counts": [1079, 3107], "val_class_counts": [270, 776], "test_class_counts": [234, 390]}

## Statistical tests (gradient-free optimisers, paired by run seed)

### mcc

Friedman χ² = 56.080, p = 1.93e-11, blocks = 20.

Mean rank (1 = best): APH-no-crossover 1.55, APH-uniform-ops 1.60, OpenAI-ES 3.55, APH-plant-only 4.15, ZO-SGD 4.15

Wilcoxon signed-rank, APH-uniform-ops vs each (Holm-adjusted); A = Vargha–Delaney (>0.5 favours APH-uniform-ops).

| vs | median diff | p | p (Holm) | A |
|---|---|---|---|---|
| APH-no-crossover | +0.0275 | 0.729 | 0.729 | 0.560 |
| APH-plant-only | +0.1596 | 1.91e-06 | 7.63e-06 | 0.960 |
| OpenAI-ES | +0.1441 | 1.34e-05 | 2.86e-05 | 0.958 |
| ZO-SGD | +0.1562 | 9.54e-06 | 2.86e-05 | 0.958 |

### accuracy

Friedman χ² = 59.739, p = 3.29e-12, blocks = 20.

Mean rank (1 = best): APH-uniform-ops 1.55, APH-no-crossover 1.55, OpenAI-ES 3.42, ZO-SGD 4.22, APH-plant-only 4.25

Wilcoxon signed-rank, APH-uniform-ops vs each (Holm-adjusted); A = Vargha–Delaney (>0.5 favours APH-uniform-ops).

| vs | median diff | p | p (Holm) | A |
|---|---|---|---|---|
| APH-no-crossover | +0.0120 | 0.723 | 0.723 | 0.556 |
| APH-plant-only | +0.0737 | 8.84e-05 | 0.000177 | 0.963 |
| OpenAI-ES | +0.0649 | 9.54e-06 | 2.86e-05 | 0.960 |
| ZO-SGD | +0.0697 | 3.81e-06 | 1.53e-05 | 0.965 |

### auc

Friedman χ² = 60.040, p = 2.85e-12, blocks = 20.

Mean rank (1 = best): APH-uniform-ops 1.40, APH-no-crossover 1.65, OpenAI-ES 3.70, APH-plant-only 3.90, ZO-SGD 4.35

Wilcoxon signed-rank, APH-uniform-ops vs each (Holm-adjusted); A = Vargha–Delaney (>0.5 favours APH-uniform-ops).

| vs | median diff | p | p (Holm) | A |
|---|---|---|---|---|
| APH-no-crossover | +0.0148 | 0.154 | 0.154 | 0.650 |
| APH-plant-only | +0.1072 | 1.91e-06 | 7.63e-06 | 0.988 |
| OpenAI-ES | +0.1022 | 3.81e-06 | 7.63e-06 | 0.983 |
| ZO-SGD | +0.1131 | 1.91e-06 | 7.63e-06 | 0.993 |

### balanced_accuracy

Friedman χ² = 48.680, p = 6.81e-10, blocks = 20.

Mean rank (1 = best): APH-no-crossover 1.60, APH-uniform-ops 1.75, OpenAI-ES 3.50, ZO-SGD 4.05, APH-plant-only 4.10

Wilcoxon signed-rank, APH-uniform-ops vs each (Holm-adjusted); A = Vargha–Delaney (>0.5 favours APH-uniform-ops).

| vs | median diff | p | p (Holm) | A |
|---|---|---|---|---|
| APH-no-crossover | +0.0100 | 0.808 | 0.808 | 0.542 |
| APH-plant-only | +0.0718 | 1.91e-06 | 7.63e-06 | 0.932 |
| OpenAI-ES | +0.0605 | 6.29e-05 | 0.000143 | 0.925 |
| ZO-SGD | +0.0639 | 4.77e-05 | 0.000143 | 0.915 |
