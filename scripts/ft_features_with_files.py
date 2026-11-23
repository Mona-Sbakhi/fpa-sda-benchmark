#!/usr/bin/env python3
"""Add the file names (results/labels.npz order) to the features written by finetune_cnn.py, so that the
experiment code can use them like the frozen features (load_precomputed + patient-grouped split).

    python scripts/ft_features_with_files.py ft_dev/s0/features.npz cache/ft_dev_s0.npz
"""
import sys
from pathlib import Path

import numpy as np

src, dst = sys.argv[1], sys.argv[2]
L = np.load(Path(__file__).resolve().parents[1] / "results" / "labels.npz")
d = np.load(src)
assert (d["y_train"] == L["y_train"]).all() and (d["y_test"] == L["y_test"]).all()
np.savez(dst, F_train=d["F_train"].astype(np.float32), y_train=d["y_train"], F_test=d["F_test"].astype(np.float32),
         y_test=d["y_test"], train_files=L["train_files"], test_files=L["test_files"])
print("wrote", dst)
