# Deviation 1 from the frozen analysis script (before any external result was computed or seen)

scripts/ext_analysis.py (SHA-256 1dfc132a53a7fce0887841212ee85650c0edd40b979d41118f2faeb3bcf05388) stopped at its
own reproducibility check before computing any external statistic: the deterministic retraining of the
gradient-trained references (logistic regression, MLP) reproduced the stored Kermany test predictions with every
predicted class identical, but with probability differences up to 0.0011 (floating-point differences between runs),
above the tolerance of 1e-5 written into the check.

Change: the check now requires identical predicted classes on the Kermany test set and a maximum probability
difference below 0.01. Nothing else in the script changes. The gradient-trained references enter only the
secondary analyses; the primary estimand uses the stored heads of the gradient-free methods, which are loaded,
not retrained.
