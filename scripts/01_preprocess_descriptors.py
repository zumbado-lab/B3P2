#!/usr/bin/env python
"""
B3P2 — Descriptor preprocessing (cleaning + variance + correlation + VIF)

Inputs (CSV, after 00 step):
  - --original  data/processed/originalfull.csv
                (fallbacks: original_descriptors_cleaned.csv)
  - --external  data/processed/externalfull.csv
                (fallbacks: external_descriptors_aligned.csv)

What it does
  1) Cleans numeric features (drop ±inf; drop all-NaN columns). Preserves ID/SMILES/class/logBB.
  2) Writes a short README and a protected-descriptors list before filtering.
  3) Filters non-protected descriptors by:
     a) zero variance
     b) pairwise correlation ≥ --corr-thresh (default 0.90)
     c) iterative VIF ≤ --vif-thresh (default 5.0)
  4) Exports:
     - data/processed/filtering_steps/step1_variance.csv
     - data/processed/filtering_steps/step2_corr.csv
     - data/processed/filtering_steps/step3_vif.csv
     - data/processed/original_filtered.csv
     - data/processed/external_filtered_aligned.csv
     - data/processed/filtering_steps/protected_descriptors.txt
     - data/processed/filtering_steps/retained_features.txt
     - data/processed/filtering_steps/variance_removed.txt
     - data/processed/filtering_steps/corr_removed.txt
     - data/processed/filtering_steps/vif_removed.txt
     - data/processed/filtering_steps/README_filters.txt
"""

from __future__ import annotations
import argparse, os
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold


# ------------------------------ I/O helpers ------------------------------
def _find_existing(*candidates: str) -> Path:
    for c in candidates:
        p = Path(c)
        if p.exists():
            return p
    raise FileNotFoundError(f"None of these exist: {candidates}")


def _detect_base_cols(df: pd.DataFrame) -> List[str]:
    base = []
    for c in ("ID", "SMILES", "class", "logBB", "SMILES_canonical"):
        if c in df.columns:
            base.append(c)
    if "ID" not in base:
        df.insert(0, "ID", np.arange(1, len(df) + 1))
        base.insert(0, "ID")
    if "SMILES" not in base and "SMILES_canonical" in base:
        df.insert(1, "SMILES", df["SMILES_canonical"])
        base.insert(1, "SMILES")
    return base


def _numeric_clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(axis=1, how="all")
    return df


def _write_list(path: Path, items: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(items) + ("\n" if items else ""), encoding="utf-8")


# ------------------------------ Filtering steps ------------------------------
def step_variance(df: pd.DataFrame, base_cols: List[str], protected: List[str]) -> (pd.DataFrame, List[str]):
    feat_cols = [c for c in df.columns if c not in base_cols]
    prot = [c for c in feat_cols if c in protected]
    to_filter = [c for c in feat_cols if c not in protected]

    X = df[to_filter].copy()
    X = X.select_dtypes(include=[np.number])
    selector = VarianceThreshold(threshold=0.0)
    keep_mask = selector.fit(X).get_support()
    kept = list(X.columns[keep_mask])

    removed = sorted(list(set(to_filter) - set(kept)))
    final_cols = base_cols + prot + kept
    out = df[final_cols].copy()
    return out, removed


def step_corr(df: pd.DataFrame, base_cols: List[str], protected: List[str], thresh: float) -> (pd.DataFrame, List[str]):
    feat_cols = [c for c in df.columns if c not in base_cols]
    prot = [c for c in feat_cols if c in protected]
    to_check = [c for c in feat_cols if c not in protected]

    if len(to_check) == 0:
        return df, []

    corr = df[to_check].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

    to_drop = set()
    for col in upper.columns:
        if any(upper[col] >= thresh):
            to_drop.add(col)

    kept = [c for c in to_check if c not in to_drop]
    final_cols = base_cols + prot + kept
    out = df[final_cols].copy()
    removed = sorted(list(to_drop))
    return out, removed


def _vif_diag(X: pd.DataFrame) -> pd.Series:
    # Standardize
    Z = (X - X.mean()) / (X.std(ddof=0) + 1e-12)
    # Correlation matrix and pseudo-inverse
    C = np.corrcoef(Z.values, rowvar=False)
    C_inv = np.linalg.pinv(C)
    diag = np.diag(C_inv)
    # VIF = diag(inv(corr))
    return pd.Series(diag, index=X.columns)


def step_vif(df: pd.DataFrame, base_cols: List[str], protected: List[str], thresh: float) -> (pd.DataFrame, List[str]):
    feat_cols = [c for c in df.columns if c not in base_cols]
    prot = [c for c in feat_cols if c in protected]
    work = [c for c in feat_cols if c not in protected]
    removed: List[str] = []

    X = df[work].copy()
    # Drop any remaining NaNs
    X = X.replace([np.inf, -np.inf], np.nan).dropna(axis=0)

    # Remove zero-variance if sneaked in
    nz = X.columns[X.std(ddof=0) > 0]
    X = X[nz].copy()

    while True:
        if X.shape[1] <= 1:
            break
        vif = _vif_diag(X)
        worst = vif.drop(labels=[c for c in vif.index if c in protected], errors="ignore").sort_values(ascending=False)
        if worst.empty or worst.iloc[0] <= thresh:
            break
        drop_col = worst.index[0]
        removed.append(drop_col)
        X = X.drop(columns=[drop_col])

    kept = list(X.columns)
    final_cols = base_cols + prot + kept
    out = df[final_cols].copy()
    return out, removed


# ------------------------------ CLI ------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--original", default=None, help="CSV with original descriptors (default: originalfull.csv or original_descriptors_cleaned.csv)")
    ap.add_argument("--external", default=None, help="CSV with external descriptors (default: externalfull.csv or external_descriptors_aligned.csv)")
    ap.add_argument("--outdir", default="data/processed", help="Base output directory")
    ap.add_argument("--corr-thresh", type=float, default=0.90)
    ap.add_argument("--vif-thresh", type=float, default=5.0)
    ap.add_argument("--protected", nargs="*", default=["XH_strength","CH_strength","MinAbsEStateIndex","VSA_EState3","BCUTp-1l"],
                    help="Always-keep descriptors (by name)")
    args = ap.parse_args()

    out_base = Path(args.outdir)
    steps_dir = out_base / "filtering_steps"
    steps_dir.mkdir(parents=True, exist_ok=True)

    # Resolve inputs
    original_csv = Path(args.original) if args.original else _find_existing(
        "data/processed/originalfull.csv",
        "data/processed/original_descriptors_cleaned.csv",
    )
    external_csv = Path(args.external) if args.external else _find_existing(
        "data/processed/externalfull.csv",
        "data/processed/external_descriptors_aligned.csv",
    )

    # Load and basic clean
    orig = pd.read_csv(original_csv)
    ext  = pd.read_csv(external_csv)
    orig = _numeric_clean(orig)
    ext  = _numeric_clean(ext)

    # Base cols
    base_cols_o = _detect_base_cols(orig)
    base_cols_e = _detect_base_cols(ext)  # ensure presence in external too

    # Write README and protected list before filtering
    readme = (
        "B3P2 descriptor preprocessing\n"
        f"- Protected descriptors (kept in all filters): {', '.join(args.protected)}\n"
        f"- Zero-variance removal on non-protected features\n"
        f"- Pairwise correlation threshold: {args.corr_thresh}\n"
        f"- Iterative VIF threshold: {args.vif_thresh}\n"
        "Outputs: step1_variance.csv, step2_corr.csv, step3_vif.csv, original_filtered.csv, external_filtered_aligned.csv\n"
    )
    (steps_dir / "README_filters.txt").write_text(readme, encoding="utf-8")
    _write_list(steps_dir / "protected_descriptors.txt", args.protected)

    # Step 1: variance on original
    step1, removed_var = step_variance(orig, base_cols_o, args.protected)
    step1.to_csv(steps_dir / "step1_variance.csv", index=False)
    _write_list(steps_dir / "variance_removed.txt", removed_var)

    # Step 2: correlation on original
    step2, removed_corr = step_corr(step1, base_cols_o, args.protected, args.corr_thresh)
    step2.to_csv(steps_dir / "step2_corr.csv", index=False)
    _write_list(steps_dir / "corr_removed.txt", removed_corr)

    # Step 3: VIF on original
    step3, removed_vif = step_vif(step2, base_cols_o, args.protected, args.vif_thresh)
    step3.to_csv(steps_dir / "step3_vif.csv", index=False)
    _write_list(steps_dir / "vif_removed.txt", removed_vif)

    # Final original
    final_feat_cols = [c for c in step3.columns if c not in base_cols_o]
    _write_list(steps_dir / "retained_features.txt", final_feat_cols)
    step3.to_csv(out_base / "original_filtered.csv", index=False)

    # Align external to final original features (same set & order)
    ext_meta = ext[base_cols_e].copy()
    # Ensure external contains all metadata columns present in original base (ID/SMILES/class/logBB)
    for c in base_cols_o:
        if c not in ext_meta.columns and c in ext.columns:
            ext_meta[c] = ext[c]
        if c not in ext_meta.columns:
            if c == "class":
                ext_meta[c] = np.nan
            elif c == "logBB":
                ext_meta[c] = np.nan
            elif c == "SMILES":
                ext_meta[c] = ext.get("SMILES_canonical", pd.Series([np.nan]*len(ext)))
            else:
                ext_meta[c] = np.nan

    missing = [c for c in final_feat_cols if c not in ext.columns]
    ext_feats = ext.reindex(columns=final_feat_cols, fill_value=np.nan)
    ext_final = pd.concat([ext_meta.reset_index(drop=True), ext_feats.reset_index(drop=True)], axis=1)
    ext_final = ext_final[step3.columns]  # exact order
    ext_final.to_csv(out_base / "external_filtered_aligned.csv", index=False)

    print("[OK] Step1 variance:", (steps_dir / "step1_variance.csv"))
    print("[OK] Step2 corr:", (steps_dir / "step2_corr.csv"))
    print("[OK] Step3 vif:", (steps_dir / "step3_vif.csv"))
    print("[OK] Final original:", (out_base / "original_filtered.csv"))
    print("[OK] Final external aligned:", (out_base / "external_filtered_aligned.csv"))
    if missing:
        print(f"[NOTE] External lacked {len(missing)} features; filled with NaN (listed in retained_features.txt order).")


if __name__ == "__main__":
    main()
