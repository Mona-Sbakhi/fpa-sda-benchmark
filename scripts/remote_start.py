#!/usr/bin/env python3
"""Start a detached experiment worker on a remote VM (used with `colab exec -f`).

Environment variables: BUNDLE (path to bundle.tgz), WORKDIR, CONFIG, ONLY (space-separated
algorithm names, optional), RUNS (optional), WORKERS (optional, default = CPU count).
The worker keeps running after this script returns; progress goes to $WORKDIR/worker.log.
"""
import os
import subprocess
import tarfile

wd = os.environ.get("WORKDIR", "/content/fpasda")
os.makedirs(wd, exist_ok=True)
if not os.path.exists(os.path.join(wd, "src")):
    with tarfile.open(os.environ.get("BUNDLE", "/content/bundle.tgz")) as t:
        t.extractall(wd, filter="data")
cmd = ["python", "scripts/run_experiments.py", os.environ["CONFIG"],
       "--workers", os.environ.get("WORKERS", str(os.cpu_count()))]
if os.environ.get("ONLY"):
    cmd += ["--only", *os.environ["ONLY"].split()]
if os.environ.get("RUNS"):
    cmd += ["--runs", os.environ["RUNS"]]
log = open(os.path.join(wd, "worker.log"), "a")
p = subprocess.Popen(cmd, cwd=wd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
print("started", p.pid, " ".join(cmd))
