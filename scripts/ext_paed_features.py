#!/usr/bin/env python3
"""External paediatric test set (pre-registration amendment 1): children (1-17 years) of the RSNA Pneumonia
training data, label Normal (0) or Lung Opacity (1), grouped by NIH patient.

Step 1 (list + frozen features):
    python scripts/ext_paed_features.py list --zip rsna.zip --mapping mappings_2018.json \
        --frozen cache/rsna_resnet50_224.npz --out cache/ext_paed.npz
Step 2 (features of a fine-tuned ResNet-50, same input pipeline as finetune_cnn.py at evaluation):
    python scripts/ext_paed_features.py finetuned --zip rsna.zip --list cache/ext_paed.npz \
        --weights ft_dev/s0/best.pt --out cache/ext_paed_ft_s0.npz
No model is evaluated here; the script only writes features.
"""
import argparse
import io
import json
import zipfile

import cv2
import numpy as np
import pandas as pd
import pydicom

ap = argparse.ArgumentParser()
ap.add_argument("step", choices=["list", "finetuned"])
ap.add_argument("--zip", required=True)
ap.add_argument("--mapping"); ap.add_argument("--frozen"); ap.add_argument("--list")
ap.add_argument("--weights"); ap.add_argument("--out", required=True)
args = ap.parse_args()
z = zipfile.ZipFile(args.zip)


def pixels(img_id):
    ds = pydicom.dcmread(io.BytesIO(z.read(f"stage_2_train_images/{img_id}.dcm")))
    a = ds.pixel_array.astype(np.float32)
    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        a = a.max() - a
    return ((a - a.min()) / max(float(a.max() - a.min()), 1e-6) * 255.0).astype(np.uint8)


if args.step == "list":
    cls = pd.read_csv(z.open("stage_2_detailed_class_info.csv")).drop_duplicates("patientId")
    cls = cls[cls["class"].isin(["Normal", "Lung Opacity"])]
    nih = {x["subset_img_id"]: x["img_id"] for x in json.load(open(args.mapping))}
    rows = []
    for pid, c in zip(cls.patientId, cls["class"]):
        age = int(str(pydicom.dcmread(io.BytesIO(z.read(f"stage_2_train_images/{pid}.dcm")),
                                      stop_before_pixels=True).PatientAge).strip())
        if 1 <= age <= 17:
            rows.append((pid, int(c == "Lung Opacity"), age, nih[pid].split("_")[0]))
    ids = np.array([r[0] for r in rows]); y = np.array([r[1] for r in rows]); age = np.array([r[2] for r in rows])
    pat = np.array([r[3] for r in rows])
    fz = np.load(args.frozen)
    fid = [str(f).split("/")[-1][:-4] for f in list(fz["train_files"]) + list(fz["test_files"])]
    F = np.concatenate([fz["F_train"], fz["F_test"]]); pos = {k: i for i, k in enumerate(fid)}
    F_frozen = F[[pos[i] for i in ids]]
    np.savez(args.out, ids=ids, y=y, age=age, patient=pat, F_frozen=F_frozen)
    print(json.dumps({"images": len(ids), "patients": len(set(pat)), "class_counts": np.bincount(y).tolist(),
                      "age_median": float(np.median(age))}))
else:
    import torch
    import torchvision
    L = np.load(args.list)
    net = torchvision.models.resnet50(weights=None); net.fc = torch.nn.Linear(2048, 2)
    net.load_state_dict(torch.load(args.weights, map_location="cpu")); net.eval()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1); std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    out = []
    for s in range(0, len(L["ids"]), 32):
        imgs = [cv2.resize(pixels(i), (256, 256), interpolation=cv2.INTER_AREA) for i in L["ids"][s:s + 32]]
        x = torch.from_numpy(np.stack(imgs)).float().div(255).unsqueeze(1)
        x = torch.nn.functional.interpolate(x, size=224, mode="bilinear", antialias=True, align_corners=False)
        x = (x.repeat(1, 3, 1, 1) - mean) / std
        with torch.inference_mode():
            f = torch.flatten(net.avgpool(net.layer4(net.layer3(net.layer2(net.layer1(
                net.maxpool(net.relu(net.bn1(net.conv1(x))))))))), 1)
        out.append(f.numpy())
    np.savez(args.out, ids=L["ids"], F=np.concatenate(out).astype(np.float32))
    print("wrote", args.out)
