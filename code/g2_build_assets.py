#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build frozen G2 raw node assets. No y is stored."""

from __future__ import annotations

import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import RDLogger

import g2_core as core


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_table", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--smiles_col", default="canonical_SMILES_RDKit")
    p.add_argument("--disable_rdkit_warnings", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if args.disable_rdkit_warnings:
        RDLogger.DisableLog("rdApp.*")

    inp = Path(args.input_table).resolve()
    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(inp, encoding="utf-8-sig", low_memory=False).reset_index(drop=True)
    if args.smiles_col not in df.columns:
        raise KeyError(args.smiles_col)

    smiles = df[args.smiles_col].astype(str).tolist()
    X = core.build_compact6_raw(smiles)
    if X.shape != (len(df), 6):
        raise RuntimeError(f"Unexpected six-descriptor set shape: {X.shape}")

    x_path = out / "g2_compact6_raw.npy"
    np.save(x_path, X)

    rowmap = pd.DataFrame({
        "global_row_index": np.arange(len(df), dtype=int),
        "SMILES": smiles,
    })
    rowmap_path = out / "g2_rowmap.csv"
    rowmap.to_csv(rowmap_path, index=False, encoding="utf-8-sig")

    meta = {
        "asset_type": "G2_frozen_raw_node_assets",
        "contains_y": False,
        "n_rows": int(len(df)),
        "n_features": 6,
        "feature_names": core.COMPACT6,
        "smiles_sha256": core.smiles_hash(smiles),
        "schema_hash": core.schema_hash(),
        "x_sha256": core.file_sha256(x_path),
        "rowmap_sha256": core.file_sha256(rowmap_path),
        "input_table": str(inp),
    }
    (out / "g2_asset_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out / "g2_frozen_spec.json").write_text(
        json.dumps(core.spec(), indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"[DONE] {x_path}")
    print(f"[rows] {len(df)}")
    print(f"[features] {core.COMPACT6}")
    print(f"[schema] {core.schema_hash()}")
    print("[contains_y] False")


if __name__ == "__main__":
    main()
