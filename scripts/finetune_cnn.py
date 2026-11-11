#!/usr/bin/env python3
"""Fine-tune ImageNet ResNet-50 on the Kermany chest X-rays (GPU). Detached-friendly: logs to a file.

    python finetune_cnn.py --data cxr256.npz --out ft_out --seed 0 [--epochs 12]

Uses the benchmark's patient-grouped split (indices stored in the bundle). The epoch is chosen by
validation MCC only; the test set is evaluated once, with the selected checkpoint. Writes
metrics.json (per-epoch log, validation and test metrics, test probabilities) and features.npz
(2048-d penultimate features of the selected model for train/test images, for head-training experiments).
"""
import argparse, json, os, time
import numpy as np, torch, torch.nn as nn, torchvision

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--epochs", type=int, default=12)
ap.add_argument("--cv"); ap.add_argument("--fold", type=int, default=0); ap.add_argument("--features", default="float32")
ap.add_argument("--bs", type=int, default=64); ap.add_argument("--lr", type=float, default=3e-4)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
torch.manual_seed(a.seed); np.random.seed(a.seed)
dev = "cuda" if torch.cuda.is_available() else "cpu"
d = np.load(a.data)
Xtr, ytr, Xte, yte, opt, val = d["X_train"], d["y_train"], d["X_test"], d["y_test"], d["opt_idx"], d["val_idx"]
if a.cv:   # patient-grouped cross-validation over all images: fold `--fold` is the test fold
    f = np.load(a.cv); Xtr = np.concatenate([Xtr, Xte]); ytr = np.concatenate([ytr, yte])
    te_idx = np.where(f["fold"] == a.fold)[0]; val = np.where(f["inner_val"][a.fold])[0]
    opt = np.where((f["fold"] != a.fold) & ~f["inner_val"][a.fold])[0]
    Xte, yte = Xtr[te_idx], ytr[te_idx]
    assert not (set(f["groups"][opt]) & set(f["groups"][te_idx])) and not (set(f["groups"][val]) & set(f["groups"][te_idx]))
    assert not set(f["groups"][opt]) & set(f["groups"][val])
MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)


def batch(X, idx, train):
    x = torch.from_numpy(X[idx]).to(dev).float().div(255).unsqueeze(1)          # N,1,256,256
    if train:   # light augmentation: random crop 224 (scale jitter), small rotation, brightness/contrast
        n = len(idx); ang = (torch.rand(n, device=dev) - 0.5) * 2 * 10 * np.pi / 180
        sc = 0.85 + 0.15 * torch.rand(n, device=dev); tx = (torch.rand(n, 2, device=dev) - 0.5) * 0.1
        th = torch.zeros(n, 2, 3, device=dev)
        th[:, 0, 0] = sc * torch.cos(ang); th[:, 0, 1] = -sc * torch.sin(ang); th[:, 1, 0] = sc * torch.sin(ang)
        th[:, 1, 1] = sc * torch.cos(ang); th[:, :, 2] = tx
        grid = nn.functional.affine_grid(th, (n, 1, 224, 224), align_corners=False)
        x = nn.functional.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)
        x = (x * (0.9 + 0.2 * torch.rand(n, 1, 1, 1, device=dev)) + (torch.rand(n, 1, 1, 1, device=dev) - 0.5) * 0.1).clamp(0, 1)
    else:
        x = nn.functional.interpolate(x, size=224, mode="bilinear", antialias=True, align_corners=False)
    return (x.repeat(1, 3, 1, 1) - MEAN) / STD


def metrics(y, p):
    pr = p >= 0.5; y = y.astype(bool)
    tp = int((pr & y).sum()); tn = int((~pr & ~y).sum()); fp = int((pr & ~y).sum()); fn = int((~pr & y).sum())
    den = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return {"tn": tn, "fp": fp, "fn": fn, "tp": tp, "accuracy": (tp + tn) / len(y),
            "recall": tp / max(tp + fn, 1), "specificity": tn / max(tn + fp, 1),
            "mcc": (tp * tn - fp * fn) / den if den else 0.0}


net = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
net.fc = nn.Linear(2048, 2); net.to(dev)
cnt = np.bincount(ytr[opt]); w = torch.tensor(len(opt) / (2 * cnt), dtype=torch.float32, device=dev)
crit = nn.CrossEntropyLoss(weight=w)
optim = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
steps = a.epochs * int(np.ceil(len(opt) / a.bs))
sched = torch.optim.lr_scheduler.OneCycleLR(optim, max_lr=a.lr, total_steps=steps, pct_start=0.15)
scaler = torch.amp.GradScaler(enabled=dev == "cuda")


@torch.no_grad()
def predict(X, idx, feats=False):
    net.eval(); P, F = [], []
    for s in range(0, len(idx), 256):
        x = batch(X, idx[s:s + 256], False)
        with torch.autocast(dev, enabled=dev == "cuda"):
            f = torch.flatten(net.avgpool(net.layer4(net.layer3(net.layer2(net.layer1(net.maxpool(net.relu(net.bn1(net.conv1(x))))))))), 1)
            P.append(torch.softmax(net.fc(f).float(), 1)[:, 1].cpu().numpy())
        if feats:
            F.append(f.float().cpu().numpy())
    return (np.concatenate(P), np.concatenate(F)) if feats else np.concatenate(P)


log, best, t0 = [], (-2, -1), time.time()
rng = np.random.default_rng(a.seed)
for ep in range(a.epochs):
    net.train(); perm = rng.permutation(opt); tot = 0.0
    for s in range(0, len(perm), a.bs):
        idx = np.sort(perm[s:s + a.bs]); x = batch(Xtr, idx, True); y = torch.from_numpy(ytr[idx]).to(dev)
        with torch.autocast(dev, enabled=dev == "cuda"):
            loss = crit(net(x).float(), y)
        optim.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(optim); scaler.update(); sched.step()
        tot += loss.item() * len(idx)
    vm = metrics(ytr[val], predict(Xtr, val))
    log.append({"epoch": ep + 1, "train_loss": tot / len(opt), "val": vm, "elapsed": time.time() - t0})
    print(json.dumps(log[-1]), flush=True)
    if vm["mcc"] > best[0]:
        best = (vm["mcc"], ep + 1); torch.save(net.state_dict(), os.path.join(a.out, "best.pt"))
net.load_state_dict(torch.load(os.path.join(a.out, "best.pt")))
pte, Fte = predict(Xte, np.arange(len(yte)), True)
ptr, Ftr = predict(Xtr, np.arange(len(ytr)), True)
res = {"seed": a.seed, "epochs": a.epochs, "selected_epoch": best[1], "device": dev,
       "gpu": torch.cuda.get_device_name(0) if dev == "cuda" else None,
       "torch": torch.__version__, "torchvision": torchvision.__version__, "log": log,
       "val": metrics(ytr[val], ptr[val]), "test": metrics(yte, pte),
       "test_proba": [round(float(p), 6) for p in pte], "train_seconds": time.time() - t0}
if a.features != "none":
    np.savez(os.path.join(a.out, "features.npz"), F_train=Ftr.astype(a.features), y_train=ytr, F_test=Fte.astype(a.features),
             y_test=yte, opt_idx=opt, val_idx=val)
res["fold"] = a.fold if a.cv else None; res["n_opt"], res["n_val"], res["n_test"] = len(opt), len(val), len(yte)
os.remove(os.path.join(a.out, "best.pt"))
json.dump(res, open(os.path.join(a.out, "metrics.json"), "w"))
print("DONE", json.dumps({"val": res["val"], "test": res["test"], "selected_epoch": best[1]}), flush=True)
