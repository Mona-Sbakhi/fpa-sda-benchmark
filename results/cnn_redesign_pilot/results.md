# Results: cnn_redesign_pilot

Track 2 pilot: redesigned hybrid (APH), memetic variants and state-of-the-art gradient-free baselines on the CNN-head task; same seeds as cnn_head runs 0-4.

Test-set metrics, mean ± SD [95 % bootstrap CI of the mean] over runs. Percentages except MCC/AUC.

| Algorithm | Runs | Accuracy | Balanced acc. | Precision | Recall | Specificity | F1 | MCC | AUC | Time (s) | Evals |
|---|---|---|---|---|---|---|---|---|---|---|---|
| APH | 5 | 80.90 ± 1.02 [80.10, 81.70] | 76.58 ± 1.58 [75.36, 77.80] | 79.39 ± 1.45 [78.29, 80.56] | 93.85 ± 0.94 [93.03, 94.46] | 59.32 ± 3.91 [56.41, 62.48] | 86.00 ± 0.58 [85.56, 86.45] | 0.59 ± 0.02 [0.57, 0.60] | 0.90 ± 0.01 [0.89, 0.91] | 56.7 | 6795 |
| L-SHADE | 5 | 81.09 ± 1.61 [79.87, 82.44] | 76.97 ± 2.00 [75.45, 78.50] | 79.79 ± 1.55 [78.63, 80.96] | 93.44 ± 1.13 [92.62, 94.26] | 60.51 ± 3.76 [57.78, 63.25] | 86.07 ± 1.10 [85.26, 86.95] | 0.59 ± 0.04 [0.56, 0.62] | 0.91 ± 0.01 [0.90, 0.92] | 63.4 | 6795 |
| OpenAI-ES | 5 | 74.55 ± 1.35 [73.81, 75.77] | 71.26 ± 2.46 [69.53, 73.31] | 77.22 ± 2.75 [75.13, 79.45] | 84.41 ± 3.66 [81.49, 87.18] | 58.12 ± 7.97 [51.88, 64.36] | 80.56 ± 0.93 [79.77, 81.22] | 0.44 ± 0.04 [0.42, 0.48] | 0.81 ± 0.02 [0.79, 0.83] | 36.3 | 6795 |
| ZO-SGD | 5 | 73.78 ± 1.61 [72.72, 75.16] | 70.63 ± 3.08 [68.23, 73.03] | 77.01 ± 3.39 [74.32, 79.70] | 83.23 ± 4.08 [79.95, 86.21] | 58.03 ± 9.75 [50.26, 65.81] | 79.87 ± 0.92 [79.14, 80.65] | 0.43 ± 0.04 [0.40, 0.47] | 0.80 ± 0.01 [0.79, 0.81] | 35.8 | 6795 |
| Memetic-APH | 5 | 80.67 ± 1.75 [79.13, 81.76] | 75.21 ± 2.18 [73.31, 76.64] | 77.63 ± 1.55 [76.32, 78.69] | 97.08 ± 0.80 [96.36, 97.64] | 53.33 ± 4.00 [50.00, 56.15] | 86.27 ± 1.13 [85.27, 86.94] | 0.59 ± 0.04 [0.56, 0.61] | 0.92 ± 0.01 [0.91, 0.93] | 48.0 | 3421 |
| Memetic-Hybrid | 5 | 81.12 ± 0.77 [80.54, 81.76] | 76.11 ± 0.93 [75.39, 76.83] | 78.49 ± 0.67 [77.97, 79.01] | 96.15 ± 0.54 [95.69, 96.62] | 56.07 ± 1.69 [54.70, 57.35] | 86.43 ± 0.52 [86.05, 86.88] | 0.60 ± 0.02 [0.58, 0.61] | 0.92 ± 0.01 [0.91, 0.93] | 41.0 | 3421 |
| Adam-only | 5 | 75.90 ± 1.64 [74.55, 77.08] | 70.07 ± 1.59 [68.68, 71.29] | 74.51 ± 0.88 [73.76, 75.24] | 93.38 ± 1.93 [91.69, 94.77] | 46.75 ± 1.64 [45.56, 48.12] | 82.88 ± 1.26 [81.87, 83.74] | 0.47 ± 0.04 [0.44, 0.50] | 0.84 ± 0.03 [0.82, 0.86] | 21.0 | 46 |

Data: {"n_opt": 4186, "n_val": 1046, "n_test": 624, "opt_class_counts": [1079, 3107], "val_class_counts": [270, 776], "test_class_counts": [234, 390]}

## Statistical tests (gradient-free optimisers, paired by run seed)

### mcc

Friedman χ² = 24.257, p = 0.000468, blocks = 5.

Mean rank (1 = best): Memetic-APH 2.20, APH 2.40, Memetic-Hybrid 2.40, L-SHADE 3.00, Adam-only 5.20, OpenAI-ES 6.00, ZO-SGD 6.80

### accuracy

Friedman χ² = 24.280, p = 0.000464, blocks = 5.

Mean rank (1 = best): APH 2.40, Memetic-APH 2.40, Memetic-Hybrid 2.40, L-SHADE 2.80, Adam-only 5.20, OpenAI-ES 5.90, ZO-SGD 6.90

### auc

Friedman χ² = 27.943, p = 9.63e-05, blocks = 5.

Mean rank (1 = best): Memetic-Hybrid 1.40, Memetic-APH 1.80, L-SHADE 3.20, APH 3.60, Adam-only 5.00, OpenAI-ES 6.20, ZO-SGD 6.80

### balanced_accuracy

Friedman χ² = 21.086, p = 0.00177, blocks = 5.

Mean rank (1 = best): APH 2.20, L-SHADE 2.40, Memetic-Hybrid 2.80, Memetic-APH 3.00, OpenAI-ES 5.20, ZO-SGD 5.80, Adam-only 6.60
