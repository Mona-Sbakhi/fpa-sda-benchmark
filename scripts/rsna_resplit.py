#!/usr/bin/env python3
"""Corrected RSNA split: adults only, grouped by the real (NIH) patient.

RSNA image identifiers are per image, and several RSNA images can come from the same NIH ChestX-ray14
patient. This script maps every image to its NIH patient with the official RSNA mapping file
(pneumonia-challenge-dataset-mappings_2018.json, published by the RSNA), drops images of patients
younger than 18 years and the few images with impossible ages (above 100; age from the DICOM header), and splits the remaining images 80/20 into a
development and a test set with a stratified split grouped by patient, so that no patient occurs in both.
The frozen ResNet-50 features are reused from rsna_features.py (feature extraction is per image).

    python scripts/rsna_resplit.py --zip rsna.zip --mapping mappings_2018.json \
        --features cache/rsna_resnet50_224.npz --out cache/rsna_adult_resnet50_224.npz

File names in the output encode the patient ("nih-<patient>-<image>.dcm"), so that the optimisation /
validation split of the experiment code also groups by patient.
"""
import argparse
import io
import json
import zipfile

import numpy as np
import pydicom
from sklearn.model_selection import StratifiedGroupKFold

ap = argparse.ArgumentParser()
ap.add_argument("--zip", required=True)
ap.add_argument("--mapping", required=True)
ap.add_argument("--features", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--min_age", type=int, default=18)
ap.add_argument("--max_age", type=int, default=100, help="ages above this are recording errors in the NIH metadata")
ap.add_argument("--test_fraction", type=float, default=0.2)
ap.add_argument("--seed", type=int, default=42)
args = ap.parse_args()

d = np.load(args.features)
F = np.concatenate([d["F_train"], d["F_test"]])
y = np.concatenate([d["y_train"], d["y_test"]])
img = [str(f).split("/")[-1][:-4] for f in list(d["train_files"]) + list(d["test_files"])]

nih = {x["subset_img_id"]: x["img_id"] for x in json.load(open(args.mapping))}
patient = np.array([nih[i].split("_")[0] for i in img])

z = zipfile.ZipFile(args.zip)
age = np.array([int(str(pydicom.dcmread(io.BytesIO(z.read(f"stage_2_train_images/{i}.dcm")),
                                        stop_before_pixels=True).PatientAge).strip()) for i in img])
keep = (age >= args.min_age) & (age <= args.max_age)
n_young, n_invalid = int((age < args.min_age).sum()), int((age > args.max_age).sum())
F, y, patient, age = F[keep], y[keep], patient[keep], age[keep]
img = [i for i, k in zip(img, keep) if k]

sg = StratifiedGroupKFold(n_splits=int(round(1 / args.test_fraction)), shuffle=True, random_state=args.seed)
tr, te = next(sg.split(F, y, patient))
assert not set(patient[tr]) & set(patient[te])
name = lambda i: f"rsna/nih-{patient[i]}-{img[i].replace('-', '')}.dcm"
info = {"images": int(keep.sum()), "excluded_under_age": n_young, "excluded_invalid_age": n_invalid, "patients": int(len(set(patient))),
        "dev_images": int(len(tr)), "dev_patients": int(len(set(patient[tr]))),
        "test_images": int(len(te)), "test_patients": int(len(set(patient[te]))),
        "dev_class_counts": np.bincount(y[tr]).tolist(), "test_class_counts": np.bincount(y[te]).tolist(),
        "age_median": float(np.median(age)), "age_range": [int(age.min()), int(age.max())]}
np.savez(args.out, F_train=F[tr], y_train=y[tr], F_test=F[te], y_test=y[te],
         train_files=np.array([name(i) for i in tr]), test_files=np.array([name(i) for i in te]),
         info=np.array(json.dumps(info)))
print(json.dumps(info, indent=1))
