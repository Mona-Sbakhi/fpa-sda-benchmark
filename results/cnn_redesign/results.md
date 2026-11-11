# Results: cnn_redesign

Track 2 confirmatory experiment: redesigned hybrid (APH), L-SHADE and memetic variants on the CNN-head task, 50 runs with the same seeds as cnn_head.

Test-set metrics, mean ± SD [95 % bootstrap CI of the mean] over runs. Percentages except MCC/AUC.

| Algorithm | Runs | Accuracy | Balanced acc. | Precision | Recall | Specificity | F1 | MCC | AUC | Time (s) | Evals |
|---|---|---|---|---|---|---|---|---|---|---|---|
| APH | 50 | 80.71 ± 1.68 [80.25, 81.17] | 76.60 ± 2.11 [76.02, 77.16] | 79.59 ± 1.70 [79.12, 80.04] | 93.07 ± 1.54 [92.64, 93.49] | 60.13 ± 4.28 [58.94, 61.27] | 85.79 ± 1.15 [85.47, 86.09] | 0.58 ± 0.04 [0.57, 0.59] | 0.90 ± 0.01 [0.89, 0.90] | 53.5 | 6795 |
| L-SHADE | 50 | 80.89 ± 1.51 [80.48, 81.31] | 76.72 ± 1.86 [76.22, 77.25] | 79.60 ± 1.45 [79.21, 80.01] | 93.39 ± 1.00 [93.12, 93.66] | 60.06 ± 3.47 [59.12, 61.03] | 85.94 ± 1.03 [85.66, 86.23] | 0.59 ± 0.03 [0.58, 0.59] | 0.90 ± 0.01 [0.90, 0.90] | 59.5 | 6795 |
| Memetic-APH | 50 | 81.08 ± 1.20 [80.75, 81.41] | 75.78 ± 1.56 [75.34, 76.19] | 78.07 ± 1.16 [77.75, 78.39] | 97.00 ± 0.71 [96.81, 97.19] | 54.56 ± 3.11 [53.69, 55.39] | 86.51 ± 0.77 [86.30, 86.72] | 0.60 ± 0.03 [0.59, 0.61] | 0.92 ± 0.01 [0.92, 0.92] | 47.2 | 3421 |
| Memetic-Hybrid | 50 | 80.21 ± 1.13 [79.90, 80.53] | 74.82 ± 1.45 [74.42, 75.22] | 77.48 ± 1.08 [77.18, 77.78] | 96.37 ± 0.72 [96.18, 96.57] | 53.27 ± 2.89 [52.49, 54.08] | 85.89 ± 0.72 [85.70, 86.10] | 0.58 ± 0.03 [0.57, 0.59] | 0.91 ± 0.01 [0.91, 0.92] | 40.6 | 3421 |
| Adam-only | 50 | 77.38 ± 1.82 [76.89, 77.88] | 71.83 ± 2.31 [71.21, 72.47] | 75.71 ± 1.69 [75.26, 76.19] | 94.02 ± 1.42 [93.63, 94.41] | 49.63 ± 4.61 [48.40, 50.93] | 83.86 ± 1.18 [83.55, 84.19] | 0.51 ± 0.04 [0.50, 0.52] | 0.85 ± 0.02 [0.84, 0.86] | 21.2 | 46 |

Data: {"n_opt": 4186, "n_val": 1046, "n_test": 624, "opt_class_counts": [1079, 3107], "val_class_counts": [270, 776], "test_class_counts": [234, 390]}

## Statistical tests (gradient-free optimisers, paired by run seed)

### mcc

Friedman χ² = 71.792, p = 9.5e-15, blocks = 50.

Mean rank (1 = best): Memetic-APH 2.20, L-SHADE 2.48, APH 2.78, Memetic-Hybrid 2.92, Adam-only 4.62

Wilcoxon signed-rank, APH vs each (Holm-adjusted); A = Vargha–Delaney (>0.5 favours APH).

| vs | median diff | p | p (Holm) | A |
|---|---|---|---|---|
| L-SHADE | -0.0122 | 0.415 | 0.704 | 0.472 |
| Memetic-APH | -0.0115 | 0.0137 | 0.0411 | 0.349 |
| Memetic-Hybrid | +0.0106 | 0.352 | 0.704 | 0.544 |
| Adam-only | +0.0757 | 9.08e-11 | 3.63e-10 | 0.898 |

### accuracy

Friedman χ² = 76.784, p = 8.36e-16, blocks = 50.

Mean rank (1 = best): L-SHADE 2.30, Memetic-APH 2.32, APH 2.62, Memetic-Hybrid 3.12, Adam-only 4.64

Wilcoxon signed-rank, APH vs each (Holm-adjusted); A = Vargha–Delaney (>0.5 favours APH).

| vs | median diff | p | p (Holm) | A |
|---|---|---|---|---|
| L-SHADE | -0.0048 | 0.372 | 0.536 | 0.474 |
| Memetic-APH | -0.0008 | 0.268 | 0.536 | 0.430 |
| Memetic-Hybrid | +0.0072 | 0.0411 | 0.123 | 0.608 |
| Adam-only | +0.0345 | 1.09e-08 | 4.35e-08 | 0.909 |

### auc

Friedman χ² = 143.360, p = 5.39e-30, blocks = 50.

Mean rank (1 = best): Memetic-APH 1.46, Memetic-Hybrid 2.02, L-SHADE 3.18, APH 3.44, Adam-only 4.90

Wilcoxon signed-rank, APH vs each (Holm-adjusted); A = Vargha–Delaney (>0.5 favours APH).

| vs | median diff | p | p (Holm) | A |
|---|---|---|---|---|
| L-SHADE | -0.0033 | 0.393 | 0.393 | 0.472 |
| Memetic-APH | -0.0247 | 4.49e-13 | 1.35e-12 | 0.047 |
| Memetic-Hybrid | -0.0152 | 2.02e-10 | 4.05e-10 | 0.149 |
| Adam-only | +0.0494 | 2.43e-13 | 9.73e-13 | 0.962 |

### balanced_accuracy

Friedman χ² = 86.196, p = 8.45e-18, blocks = 50.

Mean rank (1 = best): L-SHADE 2.07, APH 2.21, Memetic-APH 2.71, Memetic-Hybrid 3.41, Adam-only 4.60

Wilcoxon signed-rank, APH vs each (Holm-adjusted); A = Vargha–Delaney (>0.5 favours APH).

| vs | median diff | p | p (Holm) | A |
|---|---|---|---|---|
| L-SHADE | -0.0034 | 0.581 | 0.581 | 0.489 |
| Memetic-APH | +0.0115 | 0.0254 | 0.0509 | 0.633 |
| Memetic-Hybrid | +0.0212 | 3.75e-05 | 0.000113 | 0.760 |
| Adam-only | +0.0496 | 4.01e-09 | 1.6e-08 | 0.930 |
