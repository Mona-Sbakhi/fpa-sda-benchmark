# Amendment 2: VinDr-PCXR (PediCXR) as a confirmatory external test

Written before any VinDr-PCXR file was downloaded or opened, and after the result of the external test on the
RSNA children (amendment 1) was known. It adds a second, confirmatory external test; it changes nothing in the
analysis of the RSNA children, which is reported as already computed.

## Models
The stored models of the first external test, unchanged: all heads and standardisation statistics of
ext_final_frozen and ext_final_ft0-2 (combined fingerprint
2be6becb497ae89d1ce8c37d367270748e6000f2e8247ae6bdd4b20977b895f6 in docs/ext_freeze_manifest.txt), the three
fine-tuned ResNet-50 weights (ft_dev/s0-2/best.pt), and the gradient-trained references exported as their fitted
coefficients. No model is retrained, tuned or selected on VinDr-PCXR.

## Data
VinDr-PCXR (PediCXR), both official partitions used together as one external test set.
- Positive: images carrying any image-level diagnosis whose name contains "pneumonia" (case-insensitive; this
  includes pneumonia, broncho-pneumonia and pleuro-pneumonia if present).
- Negative: images whose only image-level label is "No finding".
- Excluded: all other images.
- If an image has labels from several radiologists, it is positive if any radiologist assigns a pneumonia label,
  and negative only if every radiologist assigns "No finding" and nobody assigns another label.
- Unit of resampling: the patient, from the DICOM PatientID tag; if that tag is missing or unique to every image,
  the image is the unit, and this is reported.
- Preprocessing as for the RSNA images (DICOM pixel values scaled to 0-255 per image, MONOCHROME1 inverted),
  then the input pipeline of each representation.

## Analysis
Identical to the pre-registration: Delta_r, I = Delta_fine-tuned - Delta_frozen, cluster bootstrap over
patients (5,000 resamples), smallest effect of interest 0.05 MCC, threshold 0.5.

Decision rule: the categories of the pre-registration, plus one category added now because the first external
test showed an effect of this sign:
- reversed: the upper 95% bound of I is below 0 (meaningful if, in addition, I <= -0.05).
The result on VinDr-PCXR is reported whatever it is, next to the result on the RSNA children.

## Data protection
The VinDr-PCXR files are processed only on a computer of the research team, under the data use agreement.
Only aggregate results (counts, means, intervals) leave that computer.
