#!/usr/bin/env python3
"""Pre-registered step 5: choose the initialisation setting per representation on VALIDATION MCC only
(mean over APH full and APH without O3), record the decision, and write the final configurations.
Rule: higher mean validation MCC wins; ties go to the box [-1, 1] of the main study."""
import glob
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def val_mcc(exp):
    v = [json.load(open(f))["val"]["mcc"] for f in glob.glob(str(ROOT / f"results/{exp}/runs/*/run_*.json"))]
    assert len(v) == 20, (exp, len(v))
    return float(np.mean(v))


decision = {}
for rep in ("frozen", "ft0"):
    box, glo = val_mcc(f"ext_init_{rep}_box"), val_mcc(f"ext_init_{rep}_glorot")
    decision[rep] = {"box_val_mcc": round(box, 4), "glorot_val_mcc": round(glo, 4),
                     "chosen": "glorot" if glo > box else "box"}
(ROOT / "docs/ext_init_decision.json").write_text(json.dumps(decision, indent=2))
print(json.dumps(decision, indent=2))

h = json.load(open(ROOT / "configs/cnn_head.json")); r = json.load(open(ROOT / "configs/cnn_redesign.json"))
ALG = {"APH": {"type": "aph"}, "APH-noO3": {"type": "aph", "operators": ["global", "local", "gravity"]},
       "APH-O3only": {"type": "aph", "operators": ["animal"]}, "L-SHADE": r["algorithms"]["L-SHADE"]}
BASE = {k: h["baselines"][k] for k in ("LogReg", "MLP-Adam")}
GL = {"type": "glorot", "mult": 2.0}
jobs = [("ext_final_frozen", "cache/resnet50_224.npz", "frozen")] + \
       [(f"ext_final_ft{s}", f"cache/ft_dev_s{s}.npz", "ft0") for s in (0, 1, 2)]
for name, feat, rep in jobs:
    c = dict(h); c["name"] = name
    c["description"] = (f"Pre-registered final runs ({name}); trained on Kermany development data only; "
                        f"initialisation '{decision[rep]['chosen']}' chosen on validation MCC; head weights saved "
                        "before the external paediatric test.")
    c["features"] = {"type": "cnn", "path": feat, "standardize": True}
    c["algorithms"] = {k: (dict(v, param_scale=GL) if decision[rep]["chosen"] == "glorot" else dict(v)) for k, v in ALG.items()}
    c["baselines"] = BASE; c["runs"] = 20; c["seed0"] = 0; c["save_weights"] = True
    json.dump(c, open(ROOT / f"configs/{name}.json", "w"), indent=2)
print("final configs written")
