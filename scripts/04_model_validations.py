#!/usr/bin/env python
"""
B3P2 — Internal Validations (RL, SVM-RBF, RF, XGB)
- RL: Test/External metrics, CV-5 & CV-10 (train), External bootstrap 95% CI, External Y-randomization.
- SVM-RBF: External CV-5, External bootstrap 95% CI, External Y-randomization.
- RF: External CV-5, External bootstrap 95% CI, External Y-randomization.
- XGB: Train CV-5 (OOF) metrics + bootstrap 95% CI + Y-randomization (on CV-OOF).
Outputs:
  reports/tables/*.csv
  reports/figures/*.png
"""

from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    roc_auc_score, average_precision_score, balanced_accuracy_score, f1_score,
    accuracy_score, precision_score, recall_score, matthews_corrcoef,
    brier_score_loss, confusion_matrix, roc_curve
)
from sklearn.calibration import calibration_curve
from sklearn.model_selection import StratifiedKFold, cross_val_predict

ID, SMILES, TARGET = "ID", "SMILES", "class"

# --------------------------- utils ---------------------------
def load_csv(p: Path) -> pd.DataFrame:
    df = pd.read_csv(p)
    if "Smiles" in df.columns and "SMILES" not in df.columns:
        df = df.rename(columns={"Smiles":"SMILES"})
    return df

def feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in [ID, SMILES, TARGET]]

def to_labels(y_prob: np.ndarray, tau: float) -> np.ndarray:
    return (y_prob >= tau).astype(int)

def metric_block(y_true: np.ndarray, y_prob: np.ndarray, tau: float) -> dict:
    y_pred = to_labels(y_prob, tau)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    spec = tn/(tn+fp) if (tn+fp)>0 else np.nan
    return dict(
        Accuracy=accuracy_score(y_true, y_pred),
        Precision=precision_score(y_true, y_pred, zero_division=0),
        Recall=recall_score(y_true, y_pred),
        F1=f1_score(y_true, y_pred, zero_division=0),
        BalancedAcc=balanced_accuracy_score(y_true, y_pred),
        MCC=matthews_corrcoef(y_true, y_pred),
        AUROC=roc_auc_score(y_true, y_prob),
        AUPRC=average_precision_score(y_true, y_prob),
        Brier=brier_score_loss(y_true, y_prob),
        TP=tp, FP=fp, FN=fn, TN=tn
    )

def bootstrap_ci(y: np.ndarray, yprob: np.ndarray, tau: float, n_boot: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = len(y)
    rows = []
    keys = ["Accuracy","Precision","Recall","F1","BalancedAcc","MCC","AUROC","AUPRC","Brier"]
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        mb = metric_block(y[idx], yprob[idx], tau)
        rows.append({k: mb[k] for k in keys})
    df = pd.DataFrame(rows)
    out = pd.DataFrame({
        "mean": df.mean(),
        "lo": df.quantile(0.025),
        "hi": df.quantile(0.975),
    })
    out.index.name = "Metric"
    return out.reset_index()

def save_confusion(cm: np.ndarray, title: str, outpath: Path):
    tn, fp, fn, tp = cm.ravel()
    M = np.array([[tn, fp],[fn, tp]])
    fig, ax = plt.subplots(figsize=(4.2,3.6))
    im = ax.imshow(M, cmap="Blues")
    for (i,j), v in np.ndenumerate(M):
        ax.text(j, i, int(v), ha="center", va="center")
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(["Pred BBB−","Pred BBB+"])
    ax.set_yticklabels(["True BBB−","True BBB+"])
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()

def save_roc(y: np.ndarray, yprob: np.ndarray, title: str, outpath: Path):
    fpr, tpr, _ = roc_curve(y, yprob)
    auc = roc_auc_score(y, yprob)
    plt.figure(figsize=(4.8,4.0))
    plt.plot(fpr, tpr, label=f"AUC={auc:.3f}")
    plt.plot([0,1],[0,1],"k--", linewidth=1)
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title(title); plt.legend()
    plt.tight_layout(); plt.savefig(outpath, dpi=300); plt.close()

def save_calibration(y: np.ndarray, yprob: np.ndarray, title: str, outpath: Path):
    frac_pos, mean_pred = calibration_curve(y, yprob, n_bins=10, strategy="uniform")
    plt.figure(figsize=(4.8,4.0))
    plt.plot([0,1],[0,1],"--", color="gray", linewidth=1, label="Perfect calibration")
    plt.plot(mean_pred, frac_pos, marker="o", linewidth=1.2, label="Model")
    plt.xlabel("Predicted probability"); plt.ylabel("Observed frequency")
    plt.title(title); plt.legend()
    plt.tight_layout(); plt.savefig(outpath, dpi=300); plt.close()

def make_rl(seed: int) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(solver="liblinear", max_iter=5000, random_state=seed)),
    ])

def clone_svm_from_pipeline(p: Pipeline, seed: int) -> Pipeline:
    svc = p.named_steps["svc"]
    return Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(kernel=svc.kernel, C=svc.C, gamma=svc.gamma, probability=True, random_state=seed)),
    ])

def clone_rf_from_best(rf_best: RandomForestClassifier, seed: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=rf_best.n_estimators,
        max_depth=rf_best.max_depth,
        min_samples_split=rf_best.min_samples_split,
        min_samples_leaf=rf_best.min_samples_leaf,
        max_features=rf_best.max_features,
        bootstrap=rf_best.bootstrap,
        class_weight=rf_best.class_weight,
        random_state=seed, n_jobs=-1
    )

# --------------------------- main ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/processed/splits/train80.csv")
    ap.add_argument("--test",  default="data/processed/splits/test20.csv")
    ap.add_argument("--external", default="data/processed/external_aligned.csv")
    ap.add_argument("--models_dir", default="models")
    ap.add_argument("--reports_dir", default="reports")
    ap.add_argument("--tau", type=float, default=0.40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_boot", type=int, default=500)
    ap.add_argument("--n_yrand", type=int, default=50)
    args = ap.parse_args()

    train = load_csv(Path(args.train))
    test  = load_csv(Path(args.test))
    Xc = feature_cols(train)
    X_train, y_train = train[Xc].to_numpy(), train[TARGET].astype(int).to_numpy()
    X_test,  y_test  = test[Xc].to_numpy(),  test[TARGET].astype(int).to_numpy()

    # External optional
    ext_path = Path(args.external)
    has_ext = ext_path.exists()
    if has_ext:
        ext = load_csv(ext_path)
        # align columns
        missing = [c for c in Xc if c not in ext.columns]
        for c in missing:
            ext[c] = np.nan
        cols_ext = Xc + ([TARGET] if TARGET in ext.columns else [])
        ext = ext[cols_ext]
        if TARGET in ext.columns:
            X_ext = ext[Xc].to_numpy()
            y_ext = ext[TARGET].astype(int).to_numpy()
        else:
            has_ext = False

    models_dir = Path(args.models_dir)
    reports = Path(args.reports_dir)
    figs = reports / "figures"
    tabs = reports / "tables"
    logs = reports / "logs"
    for d in [figs, tabs, logs]:
        d.mkdir(parents=True, exist_ok=True)

    # ------------ Load trained models ------------
    paths = {
        "rl": models_dir / "rl.joblib",
        "svm": models_dir / "svm_rbf.joblib",
        "rf": models_dir / "rf.joblib",
        "xgb": models_dir / "xgb.joblib",
    }
    # Backward-compat: accept previous names from 03
    if not paths["svm"].exists():
        alt = models_dir / "svm_rbf.joblib"
        if alt.exists(): paths["svm"] = alt
    if not paths["rf"].exists():
        alt = models_dir / "rf.joblib"
        if alt.exists(): paths["rf"] = alt
        alt2 = models_dir / "rf_model_optimized.joblib"
        if alt2.exists(): paths["rf"] = alt2
    if not paths["xgb"].exists():
        alt = models_dir / "xgb.joblib"
        if alt.exists(): paths["xgb"] = alt
        alt2 = models_dir / "xgb_model.joblib"
        if alt2.exists(): paths["xgb"] = alt2

    rl  = joblib.load(paths["rl"])  if paths["rl"].exists()  else make_rl(args.seed).fit(X_train, y_train)
    svm = joblib.load(paths["svm"]) if paths["svm"].exists() else Pipeline([("scaler", StandardScaler()),
                                                                            ("svc", SVC(kernel="rbf", C=1.0, gamma="scale",
                                                                                        probability=True, random_state=args.seed))]).fit(X_train, y_train)
    rf  = joblib.load(paths["rf"])  if paths["rf"].exists()  else RandomForestClassifier(
        n_estimators=600, max_depth=None, min_samples_split=2, min_samples_leaf=1,
        max_features="sqrt", bootstrap=True, class_weight="balanced",
        random_state=args.seed, n_jobs=-1).fit(X_train, y_train)
    xgb = joblib.load(paths["xgb"]) if paths["xgb"].exists() else XGBClassifier(
        n_estimators=600, max_depth=5, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
        random_state=args.seed, eval_metric="logloss", tree_method="hist", n_jobs=-1).fit(X_train, y_train)

    tau = args.tau
    seed = args.seed

    # ================= RL =================
    # Test
    yprob_test = rl.predict_proba(X_test)[:,1]
    mb = metric_block(y_test, yprob_test, tau)
    df_rl_test = pd.DataFrame([{**mb, "Model":"RL", "Split":"Test20", "tau":tau}])
    save_confusion(confusion_matrix(y_test, to_labels(yprob_test, tau)), "RL — Confusion (Test20)", figs/"rl_conf_test.png")
    save_roc(y_test, yprob_test, "RL — ROC (Test20)", figs/"rl_roc_test.png")
    save_calibration(y_test, yprob_test, "RL — Calibration (Test20)", figs/"rl_cal_test.png")

    # External
    df_rl_ext = pd.DataFrame()
    if has_ext:
        yprob_ext = rl.predict_proba(X_ext)[:,1]
        mb_ext = metric_block(y_ext, yprob_ext, tau)
        df_rl_ext = pd.DataFrame([{**mb_ext, "Model":"RL", "Split":"External", "tau":tau}])
        save_confusion(confusion_matrix(y_ext, to_labels(yprob_ext, tau)), "RL — Confusion (External)", figs/"rl_conf_ext.png")
        save_roc(y_ext, yprob_ext, "RL — ROC (External)", figs/"rl_roc_ext.png")
        save_calibration(y_ext, yprob_ext, "RL — Calibration (External)", figs/"rl_cal_ext.png")

    # CV on train (5 and 10)
    df_rl_cv = []
    for k in [5, 10]:
        cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
        yprob_oof = cross_val_predict(rl, train[Xc], train[TARGET], cv=cv, method="predict_proba")[:,1]
        mb_cv = metric_block(train[TARGET].to_numpy().astype(int), yprob_oof, tau)
        df_rl_cv.append({**mb_cv, "Model":"RL", "Split":f"Train_CV{k}", "tau":tau})
    df_rl_cv = pd.DataFrame(df_rl_cv)

    # Bootstrap 95% CI (External)
    if has_ext:
        ci_rl_ext = bootstrap_ci(y_ext, yprob_ext, tau, n_boot=args.n_boot, seed=seed)
        ci_rl_ext.insert(0, "Model", "RL")
        ci_rl_ext.insert(1, "Split", "External")
        ci_rl_ext.to_csv(tabs/"rl_external_bootstrap_ci.csv", index=False)

        # Y-Randomization (External)
        rng = np.random.default_rng(seed)
        aucs = []
        for i in range(args.n_yrand):
            y_perm = rng.permutation(y_train)
            mdl = make_rl(seed + i + 1)
            mdl.fit(X_train, y_perm)
            ypp = mdl.predict_proba(X_ext)[:,1]
            aucs.append(roc_auc_score(y_ext, ypp))
        df_yr = pd.DataFrame({"AUROC_perm":aucs})
        df_yr.to_csv(tabs/"rl_external_yrandom.csv", index=False)

    # Save RL summary
    pd.concat([df_rl_test, df_rl_ext, df_rl_cv], ignore_index=True).to_csv(tabs/"rl_validations.csv", index=False)

    # ================= SVM-RBF =================
    df_svm = pd.DataFrame()
    if has_ext:
        yprob_cv_ext = cross_val_predict(svm, X_ext, y_ext, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=seed),
                                         method="predict_proba")[:,1]
        mb_cv_ext = metric_block(y_ext, yprob_cv_ext, tau)
        df_svm = pd.DataFrame([{**mb_cv_ext, "Model":"SVM-RBF", "Split":"External_CV5", "tau":tau}])
        df_svm.to_csv(tabs/"svm_external_cv5.csv", index=False)

        # Bootstrap External (with real model on full train)
        yprob_ext_real = svm.predict_proba(X_ext)[:,1]
        ci_svm_ext = bootstrap_ci(y_ext, yprob_ext_real, tau, n_boot=args.n_boot, seed=seed)
        ci_svm_ext.insert(0, "Model", "SVM-RBF")
        ci_svm_ext.insert(1, "Split", "External")
        ci_svm_ext.to_csv(tabs/"svm_external_bootstrap_ci.csv", index=False)

        # Y-Randomization External
        rng = np.random.default_rng(seed)
        aucs = []
        for i in range(args.n_yrand):
            y_perm = rng.permutation(y_train)
            model = clone_svm_from_pipeline(svm, seed + i + 1)
            model.fit(X_train, y_perm)
            ypp = model.predict_proba(X_ext)[:,1]
            aucs.append(roc_auc_score(y_ext, ypp))
        pd.DataFrame({"AUROC_perm":aucs}).to_csv(tabs/"svm_external_yrandom.csv", index=False)

    # ================= RF =================
    df_rf = pd.DataFrame()
    if has_ext:
        yprob_cv_ext = cross_val_predict(rf, X_ext, y_ext, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=seed),
                                         method="predict_proba")[:,1]
        mb_cv_ext = metric_block(y_ext, yprob_cv_ext, tau)
        df_rf = pd.DataFrame([{**mb_cv_ext, "Model":"RF", "Split":"External_CV5", "tau":tau}])
        df_rf.to_csv(tabs/"rf_external_cv5.csv", index=False)

        # Bootstrap External (real rf)
        yprob_ext_real = rf.predict_proba(X_ext)[:,1]
        ci_rf_ext = bootstrap_ci(y_ext, yprob_ext_real, tau, n_boot=args.n_boot, seed=seed)
        ci_rf_ext.insert(0, "Model", "RF")
        ci_rf_ext.insert(1, "Split", "External")
        ci_rf_ext.to_csv(tabs/"rf_external_bootstrap_ci.csv", index=False)

        # Y-Randomization External
        rng = np.random.default_rng(seed)
        aucs = []
        for i in range(args.n_yrand):
            y_perm = rng.permutation(y_train)
            model = clone_rf_from_best(rf, seed + i + 1)
            model.fit(X_train, y_perm)
            ypp = model.predict_proba(X_ext)[:,1]
            aucs.append(roc_auc_score(y_ext, ypp))
        pd.DataFrame({"AUROC_perm":aucs}).to_csv(tabs/"rf_external_yrandom.csv", index=False)

    # ================= XGB (CV5 on train, OOF) =================
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    oof = cross_val_predict(xgb, train[Xc], train[TARGET], cv=skf, method="predict_proba")[:,1]
    auroc = roc_auc_score(y_train, oof)
    auprc = average_precision_score(y_train, oof)
    ba    = balanced_accuracy_score(y_train, to_labels(oof, tau))
    f1    = f1_score(y_train, to_labels(oof, tau), zero_division=0)
    df_xgb_cv = pd.DataFrame([{
        "Model":"XGB", "Split":"Train_CV5_OOF", "tau":tau,
        "AUROC":auroc, "AUPRC":auprc, "BalancedAcc":ba, "F1":f1
    }])
    df_xgb_cv.to_csv(tabs/"xgb_train_cv5_oof.csv", index=False)

    # Bootstrap 95% CI on OOF
    ci_xgb = bootstrap_ci(y_train, oof, tau, n_boot=args.n_boot, seed=seed)
    ci_xgb.insert(0, "Model", "XGB")
    ci_xgb.insert(1, "Split", "Train_CV5_OOF")
    ci_xgb.to_csv(tabs/"xgb_train_cv5_oof_bootstrap_ci.csv", index=False)

    # Y-Randomization on CV (perm train labels; compute OOF on perm labels)
    rng = np.random.default_rng(seed)
    aucs = []
    for i in range(args.n_yrand):
        y_perm = rng.permutation(y_train)
        oof_p = cross_val_predict(xgb, X_train, y_perm, cv=skf, method="predict_proba")[:,1]
        try:
            aucs.append(roc_auc_score(y_perm, oof_p))
        except ValueError:
            pass
    pd.DataFrame({"AUROC_perm":aucs}).to_csv(tabs/"xgb_train_cv5_yrandom.csv", index=False)

    # Summary manifest
    manifest = {
        "tau": tau, "seed": seed, "n_boot": args.n_boot, "n_yrand": args.n_yrand,
        "train": str(Path(args.train)), "test": str(Path(args.test)),
        "external": str(ext_path) if has_ext else None,
        "tables": sorted([str(p) for p in tabs.glob("*.csv")]),
        "figures": sorted([str(p) for p in figs.glob("*.png")]),
    }
    (logs/"validations_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("\n[OK] Validations complete. See reports/tables and reports/figures.")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
