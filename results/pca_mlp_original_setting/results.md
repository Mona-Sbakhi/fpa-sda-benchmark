# Results: pca_mlp_original_setting

Reproduces the notebook's setting: raw (unstandardised) PCA scores; same split/budget as the revised run, for an ablation of the standardisation fix. 10 runs per algorithm: a one-run pilot showed every gradient-free optimiser at chance level (AUC ~0.5) in this 138,530-dimensional setting, so this experiment is reported as a negative result rather than tuned further.

Test-set metrics, mean ± SD [95 % bootstrap CI of the mean] over runs. Percentages except MCC/AUC.

| Algorithm | Runs | Accuracy | Balanced acc. | Precision | Recall | Specificity | F1 | MCC | AUC | Time (s) | Evals |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Hybrid | 10 | 60.38 ± 6.59 [56.55, 64.33] | 59.56 ± 6.20 [56.12, 63.37] | 70.53 ± 5.28 [67.63, 73.76] | 62.87 ± 10.55 [56.62, 69.13] | 56.24 ± 10.38 [50.09, 62.31] | 66.15 ± 7.23 [61.82, 70.34] | 0.19 ± 0.12 [0.12, 0.26] | 0.62 ± 0.08 [0.58, 0.67] | 246.1 | 6795 |
| FPA | 10 | 56.52 ± 4.95 [53.48, 59.26] | 56.82 ± 3.94 [54.49, 59.14] | 69.31 ± 4.32 [66.85, 71.91] | 55.64 ± 14.38 [46.90, 63.90] | 57.99 ± 15.44 [48.76, 66.92] | 60.58 ± 9.38 [54.51, 65.54] | 0.14 ± 0.08 [0.09, 0.18] | 0.59 ± 0.05 [0.57, 0.62] | 250.9 | 6795 |
| SDA | 10 | 68.51 ± 4.44 [65.83, 71.04] | 67.41 ± 4.04 [65.07, 69.89] | 76.66 ± 3.80 [74.52, 78.95] | 71.82 ± 9.37 [65.85, 76.79] | 62.99 ± 10.00 [57.26, 68.80] | 73.76 ± 5.31 [70.31, 76.52] | 0.35 ± 0.08 [0.30, 0.39] | 0.73 ± 0.04 [0.71, 0.76] | 238.2 | 6795 |
| Random | 10 | 58.32 ± 4.41 [55.77, 60.93] | 56.45 ± 3.90 [54.15, 58.68] | 67.98 ± 3.39 [65.95, 69.92] | 63.92 ± 12.33 [56.31, 70.67] | 48.97 ± 14.06 [41.03, 57.52] | 65.14 ± 7.01 [60.78, 68.94] | 0.13 ± 0.08 [0.08, 0.18] | 0.58 ± 0.05 [0.55, 0.61] | 193.5 | 6795 |

Data: {"n_opt": 4186, "n_val": 1046, "n_test": 624, "opt_class_counts": [1079, 3107], "val_class_counts": [270, 776], "test_class_counts": [234, 390], "pca_explained_variance": 0.9807139039039612}

## Statistical tests (gradient-free optimisers, paired by run seed)

### mcc

Friedman χ² = 11.640, p = 0.00872, blocks = 10.

Mean rank (1 = best): SDA 1.30, Hybrid 2.80, Random 2.90, FPA 3.00

Wilcoxon signed-rank, Hybrid vs each (Holm-adjusted); A = Vargha–Delaney (>0.5 favours Hybrid).

| vs | median diff | p | p (Holm) | A |
|---|---|---|---|---|
| FPA | +0.0165 | 0.695 | 1 | 0.670 |
| SDA | -0.1999 | 0.00977 | 0.0293 | 0.120 |
| Random | -0.0049 | 0.77 | 1 | 0.670 |

### accuracy

Friedman χ² = 15.520, p = 0.00142, blocks = 10.

Mean rank (1 = best): SDA 1.15, Hybrid 2.70, FPA 3.00, Random 3.15

Wilcoxon signed-rank, Hybrid vs each (Holm-adjusted); A = Vargha–Delaney (>0.5 favours Hybrid).

| vs | median diff | p | p (Holm) | A |
|---|---|---|---|---|
| FPA | +0.0192 | 0.193 | 0.387 | 0.670 |
| SDA | -0.0929 | 0.0195 | 0.0586 | 0.145 |
| Random | +0.0096 | 0.496 | 0.496 | 0.610 |

### auc

Friedman χ² = 13.800, p = 0.00319, blocks = 10.

Mean rank (1 = best): SDA 1.20, Hybrid 2.80, FPA 2.90, Random 3.10

Wilcoxon signed-rank, Hybrid vs each (Holm-adjusted); A = Vargha–Delaney (>0.5 favours Hybrid).

| vs | median diff | p | p (Holm) | A |
|---|---|---|---|---|
| FPA | +0.0039 | 0.625 | 0.625 | 0.630 |
| SDA | -0.1232 | 0.00586 | 0.0176 | 0.090 |
| Random | +0.0274 | 0.275 | 0.551 | 0.690 |

### balanced_accuracy

Friedman χ² = 11.880, p = 0.00781, blocks = 10.

Mean rank (1 = best): SDA 1.30, Hybrid 2.70, FPA 3.00, Random 3.00

Wilcoxon signed-rank, Hybrid vs each (Holm-adjusted); A = Vargha–Delaney (>0.5 favours Hybrid).

| vs | median diff | p | p (Holm) | A |
|---|---|---|---|---|
| FPA | +0.0085 | 0.625 | 1 | 0.655 |
| SDA | -0.0996 | 0.0137 | 0.041 | 0.120 |
| Random | +0.0017 | 0.557 | 1 | 0.690 |
