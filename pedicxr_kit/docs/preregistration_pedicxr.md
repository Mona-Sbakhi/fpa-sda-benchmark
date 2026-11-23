# Pre-registration: does the current-to-pbest operator matter more on a fine-tuned representation?
## External test on an independent paediatric dataset (VinDr-PCXR / PediCXR)

Written and fixed before any image or label of the external dataset was downloaded or opened.
SHA-256 of this file is recorded in the repository history at the time of registration.

### 1. Hypothesis

H1. Inside the redesigned hybrid (APH), the benefit of the current-to-pbest operator (O3) on test MCC is
larger when the 2048-16-2 classifier head is trained on features of a ResNet-50 fine-tuned on the
development data than when it is trained on frozen ImageNet ResNet-50 features.

Effect for one representation r (frozen or fine-tuned):
    Delta_r = MCC(APH full) - MCC(APH without O3)
Primary estimand (interaction):
    I = Delta_finetuned - Delta_frozen
Both measured on the external test set.

### 2. Data

Development (all training, model selection and tuning): Kermany paediatric chest X-ray dataset,
official training folder, split by patient into the optimisation subset (4,186 images) and the validation
subset (1,046 images) exactly as in the main study. The Kermany official test folder (624 images) is used
as an internal test only.

External test (opened once, after everything below is fixed): VinDr-PCXR (PediCXR), paediatric chest
radiographs from an independent centre in Vietnam, labelled by paediatric radiologists. Binary task matching
the development labels: images with the label "Pneumonia" (positive) versus images with "No finding"
(negative); images with any other label only are excluded. Both official partitions of VinDr-PCXR are used
as one external test set, since no model is trained on it. Patients are identified from the DICOM headers;
the unit of resampling is the patient.

### 3. Representations (fixed on development data only)

- Frozen: ImageNet ResNet-50 (IMAGENET1K_V2), global-average-pooled 2048-d features.
- Fine-tuned: ResNet-50 fine-tuned end to end on the optimisation subset with the protocol of the main study
  (AdamW, one-cycle schedule, peak 3e-4, weight decay 1e-4, 12 epochs, batch 64, class-weighted
  cross-entropy, augmentation as in the main study); epoch selected by validation MCC; three seeds (0, 1, 2).
  Weights are frozen and stored before the external data are opened.
- Standardisation: statistics of the optimisation subset only.
- Preprocessing of external images: DICOM pixel data scaled to 0-255 per image (MONOCHROME1 inverted),
  resized to 224 x 224 (frozen) or 256 x 256 then centre 224 (fine-tuned, as at validation), no other
  adaptation to the external data.

### 4. Methods compared (same budget, initial population, seeds and stopping rule)

Gradient-free, 6,795 fitness evaluations (45 x 151), same uniform initial population per run seed:
- APH full (operators O1-O4)
- APH without O3 (operators O1, O2, O4)
- APH with O3 only (current-to-pbest with SHADE adaptation and crossover)
- L-SHADE
Gradient-trained references: logistic regression, MLP trained with Adam.
20 runs per method and representation (for the fine-tuned representation: 20 runs for each of the three
fine-tuning seeds, i.e. 60 runs, analysed with the fine-tuning seed as a block).
Run time is recorded for every run. Head parameters of every run are stored before the external test.

### 5. Initialisation sensitivity (development data only, before the external test)

APH full and APH without O3 on both representations with (a) the uniform box [-1, 1] of the main study and
(b) a Glorot-scaled search space with the same box; 10 runs each, judged on validation MCC. The setting used
for the external test is chosen on validation MCC and fixed in the repository before the external test.
Classification threshold: 0.5 on the softmax probability (as in the main study), not tuned.

### 6. Primary analysis

1. For every stored head, compute predictions on the external test set once.
2. Delta_r: paired difference (same run seed) of MCC between APH full and APH without O3, averaged over runs.
3. I = Delta_finetuned - Delta_frozen.
4. 95% confidence interval of I by a cluster bootstrap that resamples external test patients with
   replacement (all images of a sampled patient kept), 5,000 resamples, with all stored predictions of the
   four gradient-free methods recomputed on each resample; run seeds held fixed.
5. Smallest effect of interest: |I| = 0.05 MCC.

Decision rule (fixed in advance):
- H1 supported if the lower 95% bound of I is above 0.
- Meaningful support if, in addition, the point estimate is at least 0.05.
- No meaningful interaction if the whole interval lies within (-0.05, +0.05).
- Otherwise inconclusive; the width of the interval is reported, and the result is presented as exploratory.

### 7. Secondary analyses

Delta_r on the internal Kermany test folder; AUROC, sensitivity and specificity on the external test;
ranking of all six methods per representation; run time; calibration (reliability plot) of the
gradient-trained references. These are descriptive and are not used to decide H1.

### 8. What will not change after the external data are opened

Labels and exclusions, models, seeds, budgets, threshold, preprocessing, the estimand, the bootstrap and the
decision rule. Any deviation will be reported as a deviation, with its reason.
