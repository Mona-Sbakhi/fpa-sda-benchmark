#!/usr/bin/env python3
"""Run all algorithms x runs defined in a JSON config. Resumable: finished runs are skipped.

    python scripts/run_experiments.py configs/pca_mlp.json --workers 4
"""
import argparse
import json
import os
import sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("config")
ap.add_argument("--workers", type=int, default=os.cpu_count())
ap.add_argument("--runs", type=int, help="override number of runs (e.g. smoke tests)")
ap.add_argument("--start", type=int, default=0, help="first run index (to split runs across machines)")
ap.add_argument("--end", type=int, help="stop before this run index (default: config runs)")
ap.add_argument("--only", nargs="*", help="run only these algorithm names")
args = ap.parse_args()

# One BLAS thread per worker process; parallelism comes from running runs side by side.
if args.workers > 1:
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = "1"

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from multiprocessing import Pool  # noqa: E402

from fpasda.experiment import _init_worker, prepare, run_one, write_meta  # noqa: E402

cfg = json.loads(Path(args.config).read_text())
if args.runs:
    cfg["runs"] = args.runs
out = Path(cfg.get("output_dir", "results")) / cfg["name"]
out.mkdir(parents=True, exist_ok=True)
write_meta(cfg, out)
prepared = prepare(cfg, out)
print("data:", (out / "data_info.json").read_text())

names = list(cfg["algorithms"]) + list(cfg.get("baselines", {}))
if args.only:
    names = [n for n in names if n in args.only]
tasks = []
for run in range(args.start, args.end if args.end is not None else cfg["runs"]):
    for n in names:
        p = out / "runs" / n / f"run_{run:02d}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        if not p.exists():
            tasks.append((n, run, str(p)))
print(f"{len(tasks)} tasks to run with {args.workers} workers")

if args.workers > 1:
    with Pool(args.workers, initializer=_init_worker, initargs=(str(prepared), cfg)) as pool:
        for msg in pool.imap_unordered(run_one, tasks):
            print(msg, flush=True)
else:
    _init_worker(str(prepared), cfg)
    for t in tasks:
        print(run_one(t), flush=True)
