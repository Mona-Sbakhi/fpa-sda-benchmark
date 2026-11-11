"""Loading the Kermany chest X-ray dataset (Kaggle: paultimothymooney/chest-xray-pneumonia).

The official Kaggle layout is used as-is:
    train/{NORMAL,PNEUMONIA}  5,216 images
    val/{NORMAL,PNEUMONIA}       16 images (merged into train; too small to be useful)
    test/{NORMAL,PNEUMONIA}    624 images (234 NORMAL / 390 PNEUMONIA) -> held out, used once

Labels: NORMAL = 0, PNEUMONIA = 1 (positive class).
Images are cached as uint8 arrays in an .npz file so preprocessing runs once.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np

CLASSES = ("NORMAL", "PNEUMONIA")
EXTS = {".jpeg", ".jpg", ".png"}


def _list_split(root: Path, split: str) -> tuple[list[Path], np.ndarray]:
    paths, labels = [], []
    for label, cls in enumerate(CLASSES):
        files = sorted(p for p in (root / split / cls).iterdir()
                       if p.suffix.lower() in EXTS and not p.name.startswith("."))
        paths += files
        labels += [label] * len(files)
    return paths, np.asarray(labels, dtype=np.int64)


def _load(paths: list[Path], size: int) -> np.ndarray:
    out = np.empty((len(paths), size, size), dtype=np.uint8)
    for i, p in enumerate(paths):
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise IOError(f"Could not read {p}")
        out[i] = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    return out


def load_dataset(root: str | Path, size: int = 150, cache_dir: str | Path = "cache") -> dict:
    """Return dict with X_train, y_train, X_test, y_test (uint8 images) and file lists."""
    root = Path(root)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    tr_p, tr_y = _list_split(root, "train")
    va_p, va_y = _list_split(root, "val")
    te_p, te_y = _list_split(root, "test")
    tr_p, tr_y = tr_p + va_p, np.concatenate([tr_y, va_y])

    key = hashlib.sha1(("|".join(map(str, tr_p + te_p)) + f"|{size}").encode()).hexdigest()[:12]
    cache = cache_dir / f"cxr_{size}px_{key}.npz"
    if cache.exists():
        d = np.load(cache, allow_pickle=False)
        X_tr, X_te = d["X_train"], d["X_test"]
    else:
        X_tr, X_te = _load(tr_p, size), _load(te_p, size)
        np.savez_compressed(cache, X_train=X_tr, X_test=X_te)

    return {
        "X_train": X_tr, "y_train": tr_y,
        "X_test": X_te, "y_test": te_y,
        "train_files": [str(p.relative_to(root)) for p in tr_p],
        "test_files": [str(p.relative_to(root)) for p in te_p],
    }


def patient_id(rel_path: str) -> str:
    """Best-effort patient identifier from Kermany file names.

    PNEUMONIA files look like 'person1234_bacteria_5678.jpeg'; NORMAL files look like
    'IM-0115-0001.jpeg' or 'NORMAL2-IM-1427-0001.jpeg'. Images of the same patient share the
    prefix before the last image counter, so they are kept in the same fold.
    """
    name = Path(rel_path).stem
    if name.startswith("person"):
        return name.split("_")[0]
    parts = name.split("-")
    return "-".join(parts[:-1]) if len(parts) > 2 else name
