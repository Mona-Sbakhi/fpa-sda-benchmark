# PediCXR (VinDr-PCXR) external test kit

This kit runs the pre-registered confirmatory test (docs/preregistration_amendment_2.md) on your own computer.
The VinDr-PCXR images never leave your computer: the script writes only aggregate numbers
(counts, means, confidence intervals) to `results_pedicxr.json`. Send that file only.

## 1. Install (Python 3.10 or later; a GPU is not needed)

```
python -m venv venv
venv/bin/pip install -r requirements.txt
```
On Windows use `venv\Scripts\pip` and `venv\Scripts\python`.
The first run downloads the ImageNet ResNet-50 weights from download.pytorch.org (about 100 MB).

## 2. Check the kit (no VinDr-PCXR data needed)

```
venv/bin/python pedicxr_run.py selftest
```
Expected: `selftest passed`. Optionally verify file integrity: `sha256sum -c MANIFEST.sha256`.

## 3. Download VinDr-PCXR from PhysioNet (after credentialing and signing the data use agreement)

Download the whole dataset into one folder, e.g. `~/vindr-pcxr` (DICOM images and the label CSV files).

## 4. Check that the data are found (prints counts only)

```
venv/bin/python pedicxr_run.py check --data ~/vindr-pcxr
```
It shows which label files and columns were recognised and how many images are included
(positive: any "pneumonia" label; negative: "No finding" only). If it cannot find the label files, pass them:
`--labels path/to/image_labels_train.csv path/to/image_labels_test.csv`.
Send this output if anything looks wrong, before step 5.

## 5. Run the test (once)

```
venv/bin/python pedicxr_run.py run --data ~/vindr-pcxr
```
Takes about 0.2-0.4 s per image on a normal computer (for about 9,000 images: 30-60 minutes).
At the end it writes `results_pedicxr.json`. The script refuses to run a second time once this file exists:
the test is run only once, as registered.

## 6. Send back
Only `results_pedicxr.json`. Do not send images, label files or any per-image output.

---

# بالعربي باختصار
1. ثبّت المتطلبات (الخطوة 1)، ثم شغّل `selftest` وتأكد أنه يطبع `selftest passed`.
2. بعد موافقة PhysioNet وتوقيع الاتفاقية، حمّل VinDr-PCXR كاملة في مجلد واحد على جهازك.
3. شغّل `check` وأرسل لي ما يطبعه إذا ظهر أي خطأ.
4. شغّل `run` مرة واحدة فقط.
5. أرسل ملف `results_pedicxr.json` فقط. لا ترسل أي صور أو ملفات وسوم.
