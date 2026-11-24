# VinDr-PCXR (PediCXR) confirmatory test: record of the result

- `results_pedicxr.json` was produced by `pedicxr_kit/pedicxr_run.py run` on a computer of the research team
  (finished 2026-11-24T22:51:45Z) and contains aggregate numbers only; no image, label file or per-image
  output left that computer (PhysioNet data use agreement).
- SHA-256 of the file as received: 3d232508f8b502daedfb12e78a30bb40ac794b1b6283fb720e48df41fefcf6bb
- Data found: 7,140 images (6,050 "No finding" only, 1,090 with a pneumonia label: Pneumonia,
  Brocho-pneumonia [sic, column name in the label files], Pleuro-pneumonia), both official partitions;
  no image without a readable DICOM; no PatientID in any DICOM file, so the image is the unit of resampling
  (pre-registered fallback, amendment 2).
- Result: Delta_frozen = +0.043 [+0.035, +0.052], Delta_fine-tuned = +0.009 [+0.006, +0.013],
  I = -0.034 [-0.043, -0.025].
- Decision (amendment 2 and the original rule): the upper bound of I is below 0, i.e. the category
  "reversed"; it is not meaningful because I > -0.05, and since the whole interval lies within (-0.05, +0.05)
  the original rule classifies the result as "no meaningful interaction". The kit printed the first category
  only ("reversed"); both are reported in the paper.
- No deviation from amendment 2 occurred.
