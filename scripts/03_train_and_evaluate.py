#!/usr/bin/env python
"""
B3P2 — Train & Evaluate (RL, SVM-RBF, RF, XGBoost)
- Loads fixed splits (train80/test20) and optional external set.
- Trains the 4 models with seed=42.
- Applies deployment threshold tau (default 0.40) to probabilities.
- Prints a comparative table and writes plots + artifacts.

Inputs (defaults):
  data/processed/splits/train80.csv
  data/processed/splits/test20.csv
  data/processed/external_aligned.csv  (optional)

Outputs:
  models/{rl,svm_rbf,rf,xgb}.joblib
  reports/tables/model_comparison.csv
  reports/figures/roc_all.png
  reports/figures/calibration_all.png
"""

from __future__ import annotations
import argparse, os, json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, matthews_corrcoef,
    balanced_accuracy_score, brier_score_loss, confusion_matrix,
    roc_curve
)
from sklearn.calibration import calibration_curve


# --------------------------- helpers ---------------------------
ID, SMILES, TARGET = "ID", "SMILES", "class"

def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Smiles" in df.columns and "SMILES" not in df.columns:
        df = df.rename(columns={"Smiles": "SMILES"})
    return df

def feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in [ID, SMILES, TARGET]]

def to_labels(y_prob: np.ndarray, tau: float) -> np.ndarray:
    return (y_prob >= tau).astype(int)

def metrics_block(y_true: np.ndarray, y_prob: np.ndarray, tau: float) -> dict:
    y_pred = to_labels(y_prob, tau)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred),
        "BalancedAcc": balanced_accuracy_score(y_true, y_pred),
        "MCC": matthews_corrcoef(y_true, y_pred),
        "AUROC": roc_auc_score(y_true, y_prob),
        "AUPRC": average_precision_score(y_true, y_prob),
        "Brier": brier_score_loss(y_true, y_prob),
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
    }

def ensure_dirs(*paths: Path) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


# --------------------------- main ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/processed/splits/train80.csv")
    ap.add_argument("--test", default="data/processed/splits/test20.csv")
    ap.add_argument("--external", default="data/processed/external_aligned.csv")
    ap.add_argument("--models_dir", default="models")
    ap.add_argument("--reports_dir", default="reports")
    ap.add_argument("--tau", type=float, default=0.40)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    train_path = Path(args.train)
    test_path = Path(args.test)
    ext_path = Path(args.external)

    models_dir = Path(args.models_dir)
    figures_dir = Path(args.reports_dir) / "figures"
    tables_dir = Path(args.reports_dir) / "tables"
    logs_dir = Path(args.reports_dir) / "logs"
    ensure_dirs(models_dir, figures_dir, tables_dir, logs_dir)

    # Load data
    train = load_csv(train_path)
    test = load_csv(test_path)
    X_cols = feature_cols(train)

    X_train, y_train = train[X_cols].to_numpy(), train[TARGET].astype(int).to_numpy()
    X_test,  y_test  = test[X_cols].to_numpy(),  test[TARGET].astype(int).to_numpy()

    # External (optional)
    has_external = ext_path.exists()
    if has_external:
        ext = load_csv(ext_path)
        # align columns if user committed external with metadata first
        ext = ext[[c for c in train.columns if c in ext.columns] + [c for c in train.columns if c not in ext.columns]]
        X_ext = ext[X_cols].to_numpy()
        y_ext = ext[TARGET].astype(int).to_numpy()

    # Models
    models = {
        "rl": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(solver="liblinear", max_iter=5000, random_state=args.seed)),
        ]),
        "svm_rbf": Pipeline([
            ("scaler", StandardScaler()),
            ("svc", SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=args.seed)),
        ]),
        "rf": RandomForestClassifier(
            n_estimators=600, max_depth=None, min_samples_split=2, min_samples_leaf=1,
            max_features="sqrt", bootstrap=True, class_weight="balanced",
            random_state=args.seed, n_jobs=-1
        ),
        "xgb": XGBClassifier(
            n_estimators=600, max_depth=5, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9,
            reg_lambda=1.0, reg_alpha=0.0,
            random_state=args.seed, eval_metric="logloss", n_jobs=-1,
            tree_method="hist"
        ),
    }

    # Train
    for name, mdl in models.items():
        mdl.fit(X_train, y_train)
        joblib.dump(mdl, models_dir / f"{name}.joblib")

    # Evaluate (test and external) with fixed tau
    records = []
    roc_curves = {}
    calib_points = {}

    for name, mdl in models.items():
        # probas
        yprob_test = mdl.predict_proba(X_test)[:, 1]
        mb_test = metrics_block(y_test, yprob_test, tau=args.tau)
        mb_test.update({"Model": name, "Split": "Test20"})
        records.append(mb_test)

        # curves
        fpr, tpr, _ = roc_curve(y_test, yprob_test)
        roc_curves[name] = (fpr, tpr)
        frac_pos, mean_pred = calibration_curve(y_test, yprob_test, n_bins=10, strategy="uniform")
        calib_points[name] = (mean_pred, frac_pos)

        if has_external:
            yprob_ext = mdl.predict_proba(X_ext)[:, 1]
            mb_ext = metrics_block(y_ext, yprob_ext, tau=args.tau)
            mb_ext.update({"Model": name, "Split": "External"})
            records.append(mb_ext)

    # Table
    df = pd.DataFrame.from_records(records)
    # Order columns
    cols = ["Model","Split","Accuracy","Precision","Recall","F1","BalancedAcc",
            "MCC","AUROC","AUPRC","Brier","TP","FP","FN","TN"]
    df = df[cols]
    df.sort_values(by=["Split","AUROC"], ascending=[True, False], inplace=True)
    df.to_csv(tables_dir / "model_comparison.csv", index=False)

    # Print concise table
    print("\n=== Model comparison (tau = {:.2f}) ===".format(args.tau))
    print(df.to_string(index=False))

    # ROC combined (Test20)
    plt.figure(figsize=(5.2,4.2))
    for name, (fpr, tpr) in roc_curves.items():
        auc = df[(df.Model==name) & (df.Split=="Test20")]["AUROC"].values[0]
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    plt.plot([0,1],[0,1],"k--", linewidth=1)
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC — Test20")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "roc_all.png", dpi=300)
    plt.close()

    # Calibration combined (Test20)
    plt.figure(figsize=(5.2,4.2))
    plt.plot([0,1],[0,1],"--", linewidth=1, color="gray", label="Perfect calibration")
    for name, (mp, fp) in calib_points.items():
        plt.plot(mp, fp, marker="o", linewidth=1.2, label=name)
    plt.xlabel("Predicted probability"); plt.ylabel("Observed frequency")
    plt.title("Calibration — Test20")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "calibration_all.png", dpi=300)
    plt.close()

    # Manifest
    manifest = {
        "seed": args.seed,
        "tau": args.tau,
        "train": str(train_path),
        "test": str(test_path),
        "external": str(ext_path) if has_external else None,
        "models": [str(models_dir / f"{k}.joblib") for k in models.keys()],
        "table": str(tables_dir / "model_comparison.csv"),
        "figures": [str(figures_dir / "roc_all.png"), str(figures_dir / "calibration_all.png")],
    }
    (logs_dir / "train_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("\nArtifacts written:")
    for k, v in manifest.items():
        print(f"- {k}: {v}")


if __name__ == "__main__":
    main()
