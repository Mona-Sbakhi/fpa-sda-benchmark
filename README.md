# FPA–SDA benchmark

Code, configurations, per-run outputs and manuscript sources for

> M. Alsbakhi, A. Alsaqqa, M. Lubbad. *Do plant-inspired operators help to train a pneumonia classifier?
> A controlled benchmark and component analysis of a hybrid Flower Pollination–Seed Dispersal algorithm.*

The study benchmarks the Flower Pollination Algorithm (FPA), a seed-dispersal algorithm (SDA) and their
sequential hybrid against GA, PSO, DE, L-SHADE, evolution strategies, zeroth-order SGD and gradient-trained
classifiers on the paediatric chest X-ray pneumonia dataset (Kermany et al., 2018).

Archived release: https://doi.org/10.5281/zenodo.24466902

## Layout
```
src/fpasda/            data loading, features, flat MLP + fitness, optimisers, baselines, statistics
scripts/
  run_experiments.py   run every (method, run) of a config; resumable; one JSON per run
  extract_cnn_features.py   frozen ImageNet ResNet-50 features
  finetune_cnn.py      end-to-end fine-tuning (official split or patient-grouped cross-validation; GPU)
  analyze.py           per-experiment summary
  paper_assets.py      builds every table, figure and in-text number of the paper from results/
configs/               one JSON per experiment
results/               per-run outputs (metrics, convergence history, seeds, test probabilities)
paper/                 LaTeX source, generated tables and figures, compiled PDF
hybrid_fpa_sda.ipynb   the original implementation that the study re-examines
```

## Experiments
| Config | Content | Runs |
|---|---|---|
| `pca_mlp_original_setting`, `pca_mlp_revised` | original PCA pipeline (D = 138,530), raw and standardised | 10 |
| `cnn_head` | main benchmark on frozen ResNet-50 features (D = 32,818) | 50 |
| `cnn_tau_ablation` | transition point of the hybrid | 20 |
| `cnn_redesign`, `cnn_redesign_ablation` (`cnn_redesign_pilot`: first five seeds) | APH, L-SHADE, memetic variants, ablation | 50 / 20 |
| `results/finetune_resnet50`, `results/finetune_cv` | fine-tuning: official split (5 seeds), 5-fold cross-validation (3 seeds) | – |
| `cv_head_fold0..4` | head training on fine-tuned features inside the cross-validation | 10 per fold |
| `cnn_cma` (`cnn_cma_pilot`: step-size pilot, seeds 900–904) | sep-CMA-ES on the main CNN-feature task, same seeds as `cnn_head` | 50 |
| `rsna_head` | external replication on the RSNA Pneumonia dataset (14,863 adult radiographs), all methods | 20 |
| `split_B_random_image`, `split_C_random_patient` | evaluation protocol: pooled Kermany images re-split at image level (B) or by patient (C) | 20 |
| `cross_K2R`, `cross_R2K` | cross-dataset generalisation: Kermany → RSNA and RSNA → Kermany | 20 |
| `cnn_budget5x` | five times the evaluation budget (33,795 evaluations) on the main task | 20 |
| `rsna_adult`, `cross_K2R_adult`, `cross_R2K_adult` | corrected RSNA replication and cross-dataset runs: adults only, split by NIH patient (`scripts/rsna_resplit.py`) | 20 |
| `ext_init_*`, `ext_final_*` | pre-registered external test on children (development runs; stored heads in `results/ext_final_*/runs/*/run_XX.npy`) | 10 / 20 |
| `cnn_glorot` | search-space sensitivity: FPA, SDA, hybrid, GA, L-SHADE in the Glorot-scaled space | 20 |
| `mlp_weighted_kermany`, `mlp_weighted_rsna` | gradient reference trained on the same problem (class-weighted CE, early stopping on the patient-grouped validation subset) | 50 / 20 |
| `cnn_densenet` | second backbone: frozen DenseNet-121 features (`extract_cnn_features.py --arch densenet121`, D = 16,434) | 20 |
| `cnn_calib` | main methods rerun with stored validation probabilities and head weights, for `scripts/calibration_threshold.py` (reproduces `cnn_head`/`cnn_redesign`/`cnn_cma` runs 0–19 exactly) | 20 |

Further analyses: `scripts/sensitivity_extra.py` (pilot seeds, previously seen external images),
`scripts/calibration_threshold.py cnn_calib cnn_densenet` (Brier, ECE, PR-AUC, validation-chosen threshold),
`scripts/backbone_calib_tables.py` (tables of the two analyses), `scripts/gradcam_errors.py` (Grad-CAM error
analysis of the fine-tuned network; needs the fine-tuned weights and the 256-px image bundle) and
`scripts/graphical_abstract.py`.

## Reproduce
```bash
python -m venv venv && venv/bin/pip install -r requirements-revision.txt
# rebuild all tables, figures and numbers of the paper from the stored runs (no data download needed)
venv/bin/python scripts/paper_assets.py
cd paper && latexmk -pdf main.tex
```
To re-run experiments, download the dataset (Kaggle: `paultimothymooney/chest-xray-pneumonia`), set `data.root` in the
configs (they contain the absolute path used by the authors), then
```bash
venv/bin/python scripts/extract_cnn_features.py --root <chest_xray> --out cache/resnet50_224.npz
venv/bin/python scripts/run_experiments.py configs/smoke.json --workers 4      # quick end-to-end check
venv/bin/python scripts/run_experiments.py configs/cnn_head.json --workers 4
```
RSNA replication: accept the rules of the Kaggle competition `rsna-pneumonia-detection-challenge`, download its
archive, then
```bash
venv/bin/python scripts/rsna_features.py --zip <rsna archive>.zip --out cache/rsna_resnet50_224.npz
venv/bin/python scripts/run_experiments.py configs/rsna_head.json --workers 4
venv/bin/python scripts/rsna_summary.py rsna_head   # summary and paired tests
venv/bin/python scripts/rsna_table.py               # paper/tables/rsna.tex
```
Pre-registered external test on children: the registration, its amendments, the deviation record and the
fingerprints of all stored models are in `docs/external_test/`. Fine-tuning on the development data:
`KEEP_WEIGHTS=1 scripts/finetune_cnn.py` (resumable); the three fine-tuned ResNet-50 weights (about 94 MB each) are
stored in `pedicxr_kit/models/` (each under 100 MB). `pedicxr_kit/` is a self-contained kit that runs the
confirmatory VinDr-PCXR test on a local computer (see `pedicxr_kit/README.md`). External features: `scripts/ext_paed_features.py`;
analysis: `scripts/ext_analysis.py`; table: `scripts/ext_table.py`. The earlier RSNA runs `rsna_head`, `cross_K2R`
and `cross_R2K` used an image-level split and are kept for the record; the paper reports the corrected runs.

Protocol, cross-dataset and budget studies: `scripts/split_summary.py`, `scripts/cross_summary.py`, and
`scripts/extra_tables.py` (writes `paper/tables/split.tex`, `cross.tex`, `budget.tex`).
The study-design figure (`paper/figures/workflow.pdf`) is drawn by `scripts/workflow_figure.py`.

Runs are deterministic given the seed stored in each JSON. The `cv_head_fold*` experiments need the features of the
fine-tuned models, produced by `scripts/finetune_cnn.py --cv results/finetune_cv/cv_folds.npz --fold k --features float16`.

## Data
The images are not redistributed here. `results/labels.npz` holds the labels and relative file names in the order used
by the code; `results/finetune_cv/cv_folds.npz` holds the patient-grouped folds.
