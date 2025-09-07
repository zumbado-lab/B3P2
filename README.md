# B3P2 — Blood–Brain Barrier Permeability Predictor

This repository contains the reproducible implementation of the project **B3P2 (A Blood–Brain Barrier Permeability Predictor for Small Molecules)**.

## Reproducibility
- Datasets: in vivo (n=328) and external validation (n=50).
- Descriptors: RDKit + Mordred 2D + 22 custom descriptors by W. Zamora.
- Models: Ridge Logistic Regression (RL), Random Forest (RF), Support Vector Machine with RBF kernel (SVM), and XGBoost (final model).
- Validation: 5-fold cross-validation, bootstrap 95% CI (n=200), Y-randomization (n=50), Brier score.
- Applicability Domain: k-NN (k=5) in z-score space, cutoff p95_train.
- Explainability: TreeSHAP.

## Installation
With conda/mamba:
```bash
mamba env create -f environment.yml
mamba activate b3p2
pip install -r requirements.txt

##Project structure
configs/         # YAML configuration files
data/raw/        # Raw datasets (bbb_in_vivo.csv, bbb_external.csv)
data/processed/  # Processed datasets (generated)
notebooks/       # Original Jupyter notebooks
src/             # Reusable Python modules
scripts/         # CLI scripts (00...07)
reports/         # Tables and figures
SI/              # Supplementary information

##Example usage
python scripts/00_compute_descriptors.py --config configs/default.yaml
python scripts/02_train_models.py --config configs/default.yaml

License
MIT License © 2025 Luis A. Zumbado Silva
