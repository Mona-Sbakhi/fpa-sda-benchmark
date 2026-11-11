# FPA–SDA benchmark

Code, configurations, per-run outputs and manuscript sources for

> M. Alsbakhi, M. Lubbad, M. Alhanjouri. *Do plant-inspired operators help to train a pneumonia classifier?
> A controlled benchmark and component analysis of a hybrid Flower Pollination–Seed Dispersal algorithm.*

The study benchmarks the Flower Pollination Algorithm (FPA), a seed-dispersal algorithm (SDA) and their
sequential hybrid against GA, PSO, DE, L-SHADE, evolution strategies, zeroth-order SGD and gradient-trained
classifiers on the paediatric chest X-ray pneumonia dataset (Kermany et al., 2018).

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
Runs are deterministic given the seed stored in each JSON. The `cv_head_fold*` experiments need the features of the
fine-tuned models, produced by `scripts/finetune_cnn.py --cv results/finetune_cv/cv_folds.npz --fold k --features float16`.

## Data
The images are not redistributed here. `results/labels.npz` holds the labels and relative file names in the order used
by the code; `results/finetune_cv/cv_folds.npz` holds the patient-grouped folds.
