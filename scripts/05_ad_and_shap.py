#!/usr/bin/env python
"""
B3P2 — Applicability Domain (kNN) + SHAP (XGB) + 5 representative molecules
- AD: k=5 in z-score space; threshold = p95 of train mean kNN distances.
- SHAP: TreeExplainer over train80; export top-10 and full importances.
- Representatives: TP, TN, FP, FN prototypes + borderline near τ=0.40 (External).
Outputs:
  reports/tables/ad_summary.csv
  reports/tables/ad_distances_test20.csv
  reports/tables/ad_distances_external.csv            (if external)
  reports/figures/Fig_AD_test20_external.png|pdf      (if external)
  reports/tables/shap_importances_full.csv
  reports/tables/shap_top10.csv
  reports/figures/Fig_SHAP_top10_barh.png|pdf
  reports/tables/external_representatives_5.csv       (if external & model)
  reports/tables/external_representatives_5_summary.csv (if external & RDKit)
  reports/figures/Fig_representatives_5.png           (if RDKit succeeds)
"""

from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import confusion_matrix

import shap

# Optional: RDKit (figures + properties for 5 reps)
try:
    from rdkit import Chem
    from rdkit.Chem import Draw, Descriptors, Crippen, Lipinski, rdMolDescriptors
    HAS_RDKIT = True
except Exception:
    HAS_RDKIT = False

ID, SMILES, TARGET = "ID", "SMILES", "class"
TAU = 0.40
K_AD = 5
RANDOM_STATE = 42

def load_csv(p: Path) -> pd.DataFrame:
    df = pd.read_csv(p)
    if "Smiles" in df.columns and "SMILES" not in df.columns:
        df = df.rename(columns={"Smiles":"SMILES"})
    return df

def feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in [ID, SMILES, TARGET]]

def proba_pos(model, X):
    return model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else model.predict(X).astype(float)

def ensure_dirs(base: Path):
    (base/"figures").mkdir(parents=True, exist_ok=True)
    (base/"tables").mkdir(parents=True, exist_ok=True)

def set_ieee_style():
    plt.rcParams.update({
        "figure.dpi": 300, "savefig.dpi": 600,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "font.size": 7, "font.family": "serif",
        "font.serif": ["DejaVu Serif","Liberation Serif","Times","Times New Roman"],
        "axes.labelsize": 7, "axes.titlesize": 7,
        "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "figure.constrained_layout.use": True,
    })

def compute_ad(scaler: StandardScaler, X_tr: np.ndarray, X_q: np.ndarray, k: int = K_AD):
    Xs_tr = scaler.transform(X_tr)
    Xs_q  = scaler.transform(X_q)
    nn = NearestNeighbors(n_neighbors=k+1, metric="euclidean").fit(Xs_tr)
    d_tr, _ = nn.kneighbors(Xs_tr, return_distance=True)
    d_tr_mean = d_tr[:, 1:].mean(axis=1)
    p95 = float(np.percentile(d_tr_mean, 95))
    d_q, _ = nn.kneighbors(Xs_q, return_distance=True)
    d_q_mean = d_q[:, :k].mean(axis=1)
    in_ad = (d_q_mean <= p95).astype(int)
    return p95, d_q_mean, in_ad, Xs_tr, Xs_q

def save_ad_hist(d_te_mean, d_ex_mean, p95, figs_dir: Path):
    set_ieee_style()
    FIG_W, FIG_H = 3.5, 5.8
    XMAX = float(np.nanpercentile(np.concatenate([d_te_mean, d_ex_mean]), 99)) if len(d_ex_mean)>0 else 15.0
    XMAX = max(5.0, min(15.0, XMAX))

    def hist_panel(ax, data, xmax):
        bins = np.linspace(0, xmax, 21)
        ax.hist(np.clip(data, 0, xmax), bins=bins, edgecolor="black", linewidth=0.6, antialiased=True)
        ax.axvline(p95, color="red", linestyle="--", linewidth=1.0, label=f"p95 train = {p95:.3f}")
        ax.set_xlim(0, xmax)
        ax.set_xlabel("Mean distance to kNN (k=5)")
        ax.set_ylabel("Count")
        ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.6)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(FIG_W, FIG_H))
    hist_panel(ax1, d_te_mean, XMAX);  ax1.legend(frameon=False, loc="upper right")
    hist_panel(ax2, d_ex_mean, XMAX);  ax2.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    png = figs_dir / "Fig_AD_test20_external.png"
    pdf = figs_dir / "Fig_AD_test20_external.pdf"
    fig.savefig(png, bbox_inches="tight"); fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

def select_prototype(Xs: np.ndarray, mask: np.ndarray):
    if mask.sum() == 0: return None
    Xsub = Xs[mask]
    centroid = Xsub.mean(axis=0)
    d = np.linalg.norm(Xsub - centroid, axis=1)
    return int(np.argmin(d))

def pick_borderline(yprob: np.ndarray, in_ad: np.ndarray, tau: float = TAU):
    mask = (in_ad == 1)
    if mask.sum() == 0: mask = np.ones_like(in_ad, dtype=bool)
    idx = int(np.argmin(np.abs(yprob[mask] - tau)))
    return int(np.arange(len(yprob))[mask][idx])

def rdkit_props(m):
    return dict(
        MW   = Descriptors.MolWt(m),
        TPSA = rdMolDescriptors.CalcTPSA(m),
        logP = Crippen.MolLogP(m),
        HBD  = Lipinski.NumHDonors(m),
        HBA  = Lipinski.NumHAcceptors(m),
        RotB = Lipinski.NumRotatableBonds(m),
        Rings = rdMolDescriptors.CalcNumRings(m),
        AromAtoms = sum(1 for a in m.GetAtoms() if a.GetIsAromatic()),
        HeteroAtoms = sum(1 for a in m.GetAtoms() if a.GetAtomicNum() not in (1,6)),
    )

def main():
    # ---- paths
    train_p = Path("data/processed/splits/train80.csv")
    test_p  = Path("data/processed/splits/test20.csv")
    ext_p   = Path("data/processed/external_aligned.csv")
    models_dir = Path("models")
    reports = Path("reports"); ensure_dirs(reports)
    figs = reports/"figures"; tabs = reports/"tables"

    # ---- data
    train = load_csv(train_p)
    test  = load_csv(test_p)
    Xc = feature_cols(train)
    X_tr = train[Xc].to_numpy()
    X_te = test[Xc].to_numpy()

    # ---- scaler & AD on test
    scaler = StandardScaler().fit(X_tr)
    p95_train, d_te_mean, in_ad_te, Xs_tr, Xs_te = compute_ad(scaler, X_tr, X_te, k=K_AD)

    # ---- external (optional)
    has_ext = ext_p.exists()
    d_ex_mean = np.array([])
    in_ad_ex  = np.array([], dtype=int)
    if has_ext:
        ext = load_csv(ext_p)
        # align columns if needed (keep order of train feats)
        miss = [c for c in Xc if c not in ext.columns]
        for c in miss: ext[c] = np.nan
        ext = ext[Xc + ([TARGET] if TARGET in ext.columns else [])]
        if TARGET not in ext.columns:
            has_ext = False
        else:
            X_ex = ext[Xc].to_numpy()
            y_ex = ext[TARGET].astype(int).to_numpy()
            _, d_ex_mean, in_ad_ex, _, Xs_ex = compute_ad(scaler, X_tr, X_ex, k=K_AD)
            # AD hist figure
            save_ad_hist(d_te_mean, d_ex_mean, p95_train, figs)

    # ---- export AD tables + summary
    ad_sum = []
    ad_sum.append({"Split":"Test20", "p95_train": p95_train,
                   "coverage_inAD": float(in_ad_te.mean()), "n": len(in_ad_te)})
    pd.DataFrame({"ID": np.arange(len(d_te_mean)), "mean_knn_dist": d_te_mean, "in_AD": in_ad_te})\
      .to_csv(tabs/"ad_distances_test20.csv", index=False)

    if has_ext:
        ad_sum.append({"Split":"External", "p95_train": p95_train,
                       "coverage_inAD": float(in_ad_ex.mean()), "n": len(in_ad_ex)})
        pd.DataFrame({"ID": np.arange(len(d_ex_mean)), "mean_knn_dist": d_ex_mean, "in_AD": in_ad_ex})\
          .to_csv(tabs/"ad_distances_external.csv", index=False)

    pd.DataFrame(ad_sum).to_csv(tabs/"ad_summary.csv", index=False)

    # ---- load XGB model
    xgb_path = None
    for cand in ["xgb.joblib","xgb_model.joblib"]:
        if (models_dir/cand).exists(): xgb_path = models_dir/cand; break
    if xgb_path is None:
        print("[WARN] XGB model not found in models/. Skipping SHAP and representatives.")
        return
    xgb = joblib.load(xgb_path)

    # ---- SHAP over train80
    X_train = train[Xc].to_numpy()
    explainer = shap.TreeExplainer(xgb)
    shap_values = explainer.shap_values(X_train)  # (n_samples, n_features)
    mean_abs = np.abs(shap_values).mean(axis=0)
    imp_full = pd.DataFrame({"feature": Xc, "mean_abs_shap": mean_abs}).sort_values("mean_abs_shap", ascending=False)
    imp_full.to_csv(tabs/"shap_importances_full.csv", index=False)

    top_k = 10
    top = imp_full.head(top_k)
    top.to_csv(tabs/"shap_top10.csv", index=False)

    # ---- SHAP barh figure
    set_ieee_style()
    FIG_W, FIG_H = 3.5, 3.0
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.barh(np.arange(len(top))[::-1], top["mean_abs_shap"].values[::-1], height=0.7)
    ax.set_yticks(np.arange(len(top))[::-1])
    ax.set_yticklabels(top["feature"].values[::-1])
    ax.set_xlabel("Mean |SHAP|")
    ax.grid(True, axis="x", linestyle="--", linewidth=0.5, alpha=0.6)
    fig.tight_layout()
    fig.savefig(figs/"Fig_SHAP_top10_barh.png", bbox_inches="tight")
    fig.savefig(figs/"Fig_SHAP_top10_barh.pdf", bbox_inches="tight")
    plt.close(fig)

    # ---- Representatives (External only)
    if has_ext:
        X_ex = ext[Xc].to_numpy()
        y_ex = ext[TARGET].astype(int).to_numpy()
        yprob_ex = proba_pos(xgb, X_ex)
        yhat = (yprob_ex >= TAU).astype(int)

        # categories
        tn, fp, fn, tp = confusion_matrix(y_ex, yhat).ravel()
        TP_mask = (y_ex == 1) & (yhat == 1)
        TN_mask = (y_ex == 0) & (yhat == 0)
        FP_mask = (y_ex == 0) & (yhat == 1)
        FN_mask = (y_ex == 1) & (yhat == 0)

        # standardize external (reuse scaler)
        Xs_ex = StandardScaler().fit_transform(X_ex)  # display-only; for prototypes use same scaler as train for AD:
        # Prefer the AD scaler for consistency:
        Xs_ex = (X_ex - scaler.mean_) / scaler.scale_

        # prototype per category (closest to centroid)
        selected = []
        for cat, msk, tag in [("TP", TP_mask, "prototype_TP"),
                              ("TN", TN_mask, "prototype_TN"),
                              ("FP", FP_mask, "prototype_FP"),
                              ("FN", FN_mask, "prototype_FN")]:
            rel = select_prototype(Xs_ex, msk)
            if rel is not None:
                abs_idx = np.arange(len(X_ex))[msk][rel]
                selected.append((cat, tag, int(abs_idx)))

        # borderline near TAU among In-AD (or all if none In-AD)
        bl_abs = pick_borderline(yprob_ex, in_ad_ex if len(in_ad_ex)>0 else np.ones(len(yprob_ex), dtype=int), TAU)
        selected.append(("BORDER", "borderline_tau", bl_abs))

        # deduplicate in selection order
        seen = set(); rows = []
        ids  = (ext[ID].astype(str).values if ID in ext.columns else np.arange(len(ext)).astype(str))
        smis = ext[SMILES].astype(str).values

        for cat, why, idx in selected:
            if idx in seen: continue
            seen.add(idx)
            rows.append({
                "Category": cat, "rationale": why, ID: ids[idx], SMILES: smis[idx],
                "proba": float(yprob_ex[idx]), "pred": int(yhat[idx]), "true": int(y_ex[idx]),
                "in_AD": int(in_ad_ex[idx]) if len(in_ad_ex)>0 else np.nan,
                "dist_mean_k": float(d_ex_mean[idx]) if len(d_ex_mean)>0 else np.nan,
                "p95_train_AD": p95_train
            })

        reps = pd.DataFrame(rows)
        # keep requested order
        order_map = {"TP":0, "TN":1, "FP":2, "FN":3, "BORDER":4}
        reps["__o"] = reps["Category"].map(order_map).fillna(99)
        reps = reps.sort_values(["__o","Category","rationale"]).drop(columns="__o")
        reps.to_csv(tabs/"external_representatives_5.csv", index=False)

        # optional: RDKit summary + grid image
        if HAS_RDKIT and len(reps) > 0:
            def safe_inchikey(m):
                try: return Chem.MolToInchiKey(m)
                except Exception: return "NA"

            props_rows, mols, legends = [], [], []
            for _, r in reps.iterrows():
                smi = str(r[SMILES]); m = Chem.MolFromSmiles(smi)
                if m is None:
                    props_rows.append({**r.to_dict(), "InChIKey":"NA",
                                       "MW":np.nan,"TPSA":np.nan,"logP":np.nan,"HBD":np.nan,"HBA":np.nan,
                                       "RotB":np.nan,"Rings":np.nan,"AromAtoms":np.nan,"HeteroAtoms":np.nan})
                    mols.append(Chem.MolFromSmiles("C"))
                    legends.append(f'{r["Category"]} · {r["rationale"]}\nID {r[ID]} · invalid SMILES')
                    continue
                p = rdkit_props(m); inchikey = safe_inchikey(m)
                props_rows.append({**r.to_dict(), "InChIKey":inchikey, **p})
                legends.append(f'{r["Category"]} · {r["rationale"]}\nID {r[ID]} · p={r["proba"]:.2f} · In-AD={int(r["in_AD"]) if not np.isnan(r["in_AD"]) else "NA"}')
                mols.append(m)

            pd.DataFrame(props_rows).to_csv(tabs/"external_representatives_5_summary.csv", index=False)

            try:
                img = Draw.MolsToGridImage(mols, molsPerRow=5, subImgSize=(280,280), legends=legends, useSVG=False)
                out_png = reports/"figures"/"Fig_representatives_5.png"
                img.save(str(out_png))
            except Exception as e:
                print("[WARN] RDKit drawing failed:", e)

    print("[OK] AD, SHAP, and representatives completed.")
    print(f"Tables -> {tabs}")
    print(f"Figures -> {figs}")

if __name__ == "__main__":
    main()
