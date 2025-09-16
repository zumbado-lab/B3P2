#!/usr/bin/env python
"""
B3P2 — Descriptor computation (2D): Mordred + RDKit classics
- Reads ORIGINAL and EXTERNAL CSVs (SMILES + ID + class/logBB)
- Canonicalizes and validates SMILES
- Computes ~1600 Mordred 2D + full RDKit descriptor set
- Keeps ID, SMILES, target; cleans numeric (drop ±inf; drop all-NaN columns)
- Aligns EXTERNAL columns to ORIGINAL reference (same set & order)
- Optionally merges WZR extra descriptors for each split
- Emits descriptor spec (column list), versions, checksums

Outputs (default: data/processed/):
  - original_descriptors_cleaned.csv / .parquet
  - external_descriptors_cleaned.csv / .parquet
  - external_descriptors_aligned.csv / .parquet
  - originalfull.csv / externalfull.csv        (if WZR provided)
  - descriptor_columns.txt
  - versions.json
  - checksums.json
"""

from __future__ import annotations
import argparse, json, os, sys, time, hashlib, platform
from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import Descriptors as RDDesc
from rdkit.Chem.rdmolfiles import MolToSmiles

from mordred import Calculator, descriptors


# -------------------- utils --------------------
def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonicalize_smiles(s: str) -> Optional[str]:
    try:
        mol = Chem.MolFromSmiles(s)
        return MolToSmiles(mol, canonical=True) if mol is not None else None
    except Exception:
        return None


def rdkit_classic_2d(mol: Chem.Mol) -> dict:
    out = {}
    for name, fn in RDDesc.descList:
        try:
            out[name] = float(fn(mol))
        except Exception:
            out[name] = np.nan
    return out


def compute_block(smiles: List[str], ignore_3d: bool = True) -> pd.DataFrame:
    cano = [canonicalize_smiles(s) for s in smiles]
    valid_mask = [c is not None for c in cano]
    idx_valid = [i for i, v in enumerate(valid_mask) if v]
    mols = [Chem.MolFromSmiles(cano[i]) for i in idx_valid]

    # Mordred (2D only)
    calc = Calculator(descriptors, ignore_3D=ignore_3d)
    mordred_df = calc.pandas(mols)
    mordred_df.index = idx_valid

    # RDKit classics
    rd_rows = [rdkit_classic_2d(m) for m in mols]
    rd_df = pd.DataFrame(rd_rows, index=idx_valid)

    # Merge and reinsert into original order
    X = pd.concat([mordred_df, rd_df], axis=1)
    # force numeric
    for c in X.columns:
        if not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = pd.to_numeric(X[c], errors="coerce")

    full = pd.DataFrame(index=range(len(smiles)), columns=X.columns, dtype=float)
    full.loc[idx_valid, :] = X.values
    full.insert(0, "SMILES_canonical", cano)
    full.insert(1, "valid_smiles", valid_mask)
    return full


def clean_numeric_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(axis=1, how="all")
    return df


def detect_smiles_column(df: pd.DataFrame) -> str:
    for k in ("SMILES", "smiles"):
        if k in df.columns:
            return k
    raise ValueError("SMILES column not found (expected 'SMILES' or 'smiles').")


def detect_id_column(df: pd.DataFrame, hint: Optional[str]) -> str:
    if hint and hint in df.columns:
        return hint
    for k in ("ID", "compound_id", "ID Num Propio", "id"):
        if k in df.columns:
            return k
    raise ValueError("ID column not found; pass --id-col-original/--id-col-external.")


def write_spec(columns: List[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(columns) + "\n", encoding="utf-8")


# -------------------- core --------------------
def process_dataset(
    csv_path: Path,
    id_col_hint: Optional[str],
    target_col: str,
    keep_logbb: bool = True,
) -> tuple[pd.DataFrame, str, str]:
    raw = pd.read_csv(csv_path)
    smi_col = detect_smiles_column(raw)
    id_col = detect_id_column(raw, id_col_hint)

    smiles = raw[smi_col].astype(str).tolist()
    X = compute_block(smiles, ignore_3d=True)

    # drop invalid SMILES rows
    X = X[X["valid_smiles"]].drop(columns=["valid_smiles"]).reset_index(drop=True)

    # base/meta
    out = X.copy()
    out.insert(1, "SMILES", raw.loc[X.index, smi_col].reset_index(drop=True))
    out.insert(0, "ID", raw.loc[X.index, id_col].reset_index(drop=True))

    if target_col in raw.columns:
        out[target_col] = raw.loc[X.index, target_col].reset_index(drop=True)
    if keep_logbb and "logBB" in raw.columns:
        out["logBB"] = raw.loc[X.index, "logBB"].reset_index(drop=True)

    # numeric cleaning
    meta_cols = [c for c in ("ID", "SMILES", "SMILES_canonical", target_col, "logBB") if c in out.columns]
    feats = out.drop(columns=meta_cols, errors="ignore")
    feats = clean_numeric_frame(feats)
    # drop columns that are all-NaN after cleaning
    feats = feats.dropna(axis=1, how="all")

    # keep only numeric features + meta
    out = pd.concat([out[meta_cols], feats], axis=1)
    return out, id_col, smi_col


def align_external_to_reference(ext_df: pd.DataFrame, ref_df: pd.DataFrame) -> pd.DataFrame:
    ref_cols = ref_df.columns.tolist()
    for col in ref_cols:
        if col not in ext_df.columns:
            ext_df[col] = np.nan
    aligned = ext_df[ref_cols]
    return aligned


def safe_merge_wzr(main_df: pd.DataFrame, wzr_df: pd.DataFrame, id_col_name: str) -> pd.DataFrame:
    if "compound_id" in wzr_df.columns and id_col_name != "compound_id":
        wzr_df = wzr_df.rename(columns={"compound_id": id_col_name})
    # Avoid duplicating ID if present in wzr
    wzr_df = wzr_df.drop(columns=[id_col_name], errors="ignore")
    merged = pd.concat([main_df.reset_index(drop=True), wzr_df.reset_index(drop=True)], axis=1)
    # cleanup accidental duplicates like ID.1 or class_x/class_y
    dup_id = [c for c in merged.columns if c.startswith(f"{id_col_name}.")]
    merged = merged.drop(columns=dup_id, errors="ignore")
    if "class_x" in merged.columns:
        merged = merged.drop(columns=["class_y"], errors="ignore").rename(columns={"class_x": "class"})
    return merged


# -------------------- cli --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--original", default="data/raw/original_clean.csv")
    ap.add_argument("--external", default="data/raw/external_clean.csv")
    ap.add_argument("--id-col-original", default=None)
    ap.add_argument("--id-col-external", default=None)
    ap.add_argument("--target", default="class")
    ap.add_argument("--outdir", default="data/processed")
    ap.add_argument("--wzr-original", default=None, help="Optional CSV with extra WZR descriptors for ORIGINAL")
    ap.add_argument("--wzr-external", default=None, help="Optional CSV with extra WZR descriptors for EXTERNAL")
    args = ap.parse_args()

    np.random.seed(42)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    orig_df, orig_id, _ = process_dataset(Path(args.original), args.id_col_original, args.target)
    ext_df,  ext_id,  _ = process_dataset(Path(args.external),  args.id_col_external, args.target)

    # Save cleaned (pre-alignment)
    orig_csv = outdir / "original_descriptors_cleaned.csv"
    ext_csv  = outdir / "external_descriptors_cleaned.csv"
    orig_parq = outdir / "original_descriptors_cleaned.parquet"
    ext_parq  = outdir / "external_descriptors_cleaned.parquet"

    orig_df.to_csv(orig_csv, index=False); orig_df.to_parquet(orig_parq, index=False)
    ext_df.to_csv(ext_csv, index=False);   ext_df.to_parquet(ext_parq, index=False)
    print(f"[OK] {orig_csv}  shape={orig_df.shape}")
    print(f"[OK] {ext_csv}   shape={ext_df.shape}")

    # Align external to original reference (same set/order)
    aligned_ext = align_external_to_reference(ext_df.copy(), orig_df)
    aligned_csv = outdir / "external_descriptors_aligned.csv"
    aligned_parq = outdir / "external_descriptors_aligned.parquet"
    aligned_ext.to_csv(aligned_csv, index=False); aligned_ext.to_parquet(aligned_parq, index=False)
    print(f"[OK] {aligned_csv}  shape={aligned_ext.shape}")

    # Descriptor spec (features only, stable)
    meta_cols = [c for c in ("ID", "SMILES", "SMILES_canonical", args.target, "logBB") if c in orig_df.columns]
    spec_cols = [c for c in orig_df.columns if c not in meta_cols]
    spec_path = outdir / "descriptor_columns.txt"
    write_spec(spec_cols, spec_path)
    print(f"[OK] descriptor spec -> {spec_path}  (n={len(spec_cols)})")

    # Optional: merge WZR extra descriptors
    if args.wzr_original:
        wzr_o = pd.read_csv(args.wzr_original)
        original_full = safe_merge_wzr(orig_df, wzr_o, orig_id)
        (outdir / "originalfull.csv").write_text(original_full.to_csv(index=False))
        print("[OK] originalfull.csv (with WZR) written")
    if args.wzr_external:
        wzr_e = pd.read_csv(args.wzr_external)
        external_full = safe_merge_wzr(aligned_ext, wzr_e, ext_id)
        (outdir / "externalfull.csv").write_text(external_full.to_csv(index=False))
        print("[OK] externalfull.csv (with WZR) written")

    # Versions / checksums
    versions = {
        "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": sys.version, "platform": platform.platform(),
    }
    try:
        import rdkit; versions["rdkit"] = rdkit.__version__
    except Exception: versions["rdkit"] = "unknown"
    try:
        import mordred; versions["mordred"] = getattr(mordred, "__version__", "unknown")
    except Exception: versions["mordred"] = "unknown"
    (outdir / "versions.json").write_text(json.dumps(versions, indent=2))

    checks = {
        "original_csv_sha256": sha256(Path(args.original)),
        "external_csv_sha256": sha256(Path(args.external)),
        "orig_desc_sha256": sha256(orig_parq),
        "ext_desc_sha256": sha256(ext_parq),
        "aligned_ext_sha256": sha256(aligned_parq),
        "descriptor_spec": str(spec_path),
    }
    (outdir / "checksums.json").write_text(json.dumps(checks, indent=2))
    print("[DONE] descriptors computed and saved.")


if __name__ == "__main__":
    main()
