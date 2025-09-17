# B3P2 — Blood–Brain Barrier Permeability Predictor

This repository contains the full implementation of **B3P2**, a reproducible pipeline to predict the permeability of small molecules through the blood–brain barrier (BBB).  

The project started from a curated in vivo dataset and evolved into a complete modeling framework that includes preprocessing, descriptor selection, model training, validation, interpretability, and reproducibility.  

---

## Reproducibility

- **Datasets:** curated in vivo set (n = 328) and external validation set (n = 50).  
- **Descriptors:** RDKit, Mordred 2D, plus 22 custom descriptors (by W. Zamora).  
- **Models implemented:**  
  - Ridge Logistic Regression (RL)  
  - Random Forest (RF)  
  - Support Vector Machine with RBF kernel (SVM)  
  - XGBoost (XGB, final model)  
- **Validation strategy:** 5-fold CV, bootstrap 95% CI (n = 200), Y-randomization (n = 50).  
- **Metrics:** AUROC, AUPRC, MCC, Balanced Accuracy, Brier score.  
- **Applicability Domain:** kNN (k = 5) in z-score space, cutoff at p95.  
- **Interpretability:** SHAP values (TreeSHAP), importance ranking, and top 10 features.  

---

## Installation

Clone the repo and install dependencies with conda/mamba:

```bash
mamba env create -f environment.yml
mamba activate b3p2
pip install -r requirements.txt

---

##Repository structure

configs/         # YAML configuration files
data/raw/        # Raw datasets (bbb_in_vivo.csv, bbb_external.csv)
data/processed/  # Processed descriptor matrices and splits
notebooks/       # Jupyter notebooks (exploration, figures)
models/          # Trained .joblib models
reports/         # Figures, logs, tables, exports
scripts/         # Python scripts (00–05 pipeline)
    00_compute_descriptors.py
    01_preprocess_descriptors.py
    02_feature_selection.py
    03_train_and_evaluate.py
    04_model_validations.py
    05_ad_and_shap.py

---

##Usage

Run the full pipeline with:
python scripts/03_train_and_evaluate.py --config configs/default.yaml
For descriptor generation:
python scripts/00_compute_descriptors.py --config configs/default.yaml

MIT License © 2025
Luis A. Zumbado Silva
