# Amendment 1 to the pre-registration (external test set)

Written before any model of this study was evaluated on the data named below. It replaces Section 2
("External test") of preregistration_pedicxr.md (SHA-256 652092f12a07fcf6b08c29b388350d7d07a8a50258b61ca3253f322fb9285ea5).
Everything else in the pre-registration is unchanged.

## Reason
Access to VinDr-PCXR (PediCXR) requires credentialed approval that would delay the study by weeks. The
authors decided not to request it and to use an external paediatric test set that is available now.

## New external test set
Children in the RSNA Pneumonia Detection Challenge training data (frontal chest radiographs from the NIH
Clinical Center, USA; an institution independent of the Kermany data, Guangzhou, China):
- inclusion: patient age 1-17 years in the DICOM header, label "Normal" (negative) or "Lung Opacity"
  (positive); images labelled only "No Lung Opacity / Not Normal" are excluded;
- 847 images (462 normal, 385 lung opacity) of 448 patients (NIH patient identifiers from the official RSNA
  mapping file); median age 13 years;
- the unit of resampling in the bootstrap is the NIH patient;
- preprocessing as in scripts/rsna_features.py (DICOM pixel values scaled to 0-255 per image), then the
  input pipeline of each representation without adaptation to these data.
No model of this study is trained, tuned or selected on these images; all models are trained on Kermany.

## Disclosure of prior exposure
In an earlier experiment of this project (RSNA replication before correction), 167 of these images were part
of a test set on which classifier heads trained on frozen ResNet-50 features (logistic regression, MLP,
L-SHADE, GA, SDA and the sequential hybrid) were evaluated together with adult images. No APH variant and no
fine-tuned representation was evaluated on any of these images, and the quantity estimated here (the
interaction I) was never computed on them.

## Known limitations of the replacement
- About 448 patients instead of about 9,000 studies: the confidence interval of I will be wider, and an
  inconclusive result under the pre-registered decision rule is possible.
- The children are older (median 13 years) than in the Kermany development data, and the labels are lung
  opacity rather than a clinical pneumonia diagnosis.
- 33 of the 448 patients also have images taken as adults; this does not affect the analysis, since no RSNA
  image is used for training here.
