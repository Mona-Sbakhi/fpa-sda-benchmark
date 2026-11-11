#!/usr/bin/env bash
# Package code + prepared features of one experiment so it can run on another machine
# (Colab, Kaggle) without the raw images:   scripts/make_bundle.sh results/pca_mlp_revised bundle.tgz
set -euo pipefail
res="$1"; out="${2:-bundle.tgz}"
test -f "$res/prepared.npz" || { echo "run prepare first: $res/prepared.npz missing" >&2; exit 1; }
tar -czf "$out" src scripts configs requirements-revision.txt "$res/prepared.npz" "$res/data_info.json"
echo "wrote $out ($(du -h "$out" | cut -f1))"
