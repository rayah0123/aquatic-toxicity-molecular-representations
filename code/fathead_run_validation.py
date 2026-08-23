#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fathead_run_validation.py

Validation-only production runner for the frozen Fathead-minnow robustness
experiment. It reuses the Daphnia-frozen representation/model specifications.

NO frozen TEST metrics are evaluated by this runner.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd


EXPECTED_N = 821
SPLIT_SEEDS = [42, 123, 777, 2025, 3407]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda")
    p.add_argument("--skip_existing", action="store_true")
    return p.parse_args()


def run(cmd):
    print("\n[RUN]", " ".join(map(str, cmd)), flush=True)
    subprocess.run(cmd, check=True)


def audit_input(input_table: Path, split_root: Path):
    d = pd.read_csv(input_table, encoding="utf-8-sig", low_memory=False)

    if len(d) != EXPECTED_N:
        raise RuntimeError(f"Expected Fish N={EXPECTED_N}, found {len(d)}")

    required = {
        "canonical_SMILES_RDKit",
        "smiles_connectivity",
        "Tox_original",
    }
    missing = required - set(d.columns)
    if missing:
        raise KeyError(f"Missing Fish input columns: {sorted(missing)}")

    if d["canonical_SMILES_RDKit"].nunique(dropna=False) != EXPECTED_N:
        raise RuntimeError("Fish exact canonical SMILES are not unique.")

    if d["Tox_original"].isna().any():
        raise RuntimeError("Missing Fish target values.")

    for seed in SPLIT_SEEDS:
        p = split_root / f"split_seed{seed}" / "split_membership.csv"
        if not p.exists():
            raise FileNotFoundError(p)

        m = pd.read_csv(p, encoding="utf-8-sig")
        if len(m) != EXPECTED_N:
            raise RuntimeError(f"Split {seed}: expected {EXPECTED_N} rows, got {len(m)}")

        vc = m["split"].astype(str).str.lower().value_counts()
        if set(vc.index) != {"train", "val", "test"}:
            raise RuntimeError(f"Split {seed}: invalid membership labels {vc.to_dict()}")

    print("[INPUT/SPLIT QC] PASS")
    print(f"[Fish N] {EXPECTED_N}")
    print("[Frozen TEST evaluated] NO")


def main():
    args = parse_args()

    script_dir = Path(__file__).resolve().parent
    root = script_dir.parent

    input_table = root / "03_processed" / "TEST" / "TEST_Fathead_96h_LC50_modeling_pool.csv"
    split_root = root / "04_splits" / "TEST_Fathead" / "scaffold_split_murcko_topology_refined"

    asset_root = root / "02_intermediate" / "TEST" / "LC50" / "model_assets"
    g1_asset_dir = asset_root / "G1"
    g2_asset_dir = asset_root / "G2"
    y_dir = asset_root / "target"

    result_root = root / "07_results" / "Fathead"
    rf_out = result_root / "rf_validation"
    g1_out = result_root / "g1_validation"
    g2_out = result_root / "g2_validation"

    for p in [g1_asset_dir, g2_asset_dir, y_dir, rf_out, g1_out, g2_out]:
        p.mkdir(parents=True, exist_ok=True)

    audit_input(input_table, split_root)

    # 1) RF validation: frozen Morgan-RF + Descriptor-RF specifications.
    rf_script = script_dir / "fathead_rf_validation.py"
    if not rf_script.exists():
        raise FileNotFoundError(rf_script)

    run([
        sys.executable, str(rf_script),
        "--input_table", str(input_table),
        "--split_root", str(split_root),
        "--out_dir", str(rf_out),
        "--disable_rdkit_warnings",
    ])

    # 2) Build G1 unlabeled molecular-graph asset and target-only y asset.
    run([
        sys.executable,
        str(script_dir / "g1_build_graph_assets.py"),
        "--input_table", str(input_table),
        "--out_dir", str(g1_asset_dir),
        "--smiles_col", "canonical_SMILES_RDKit",
        "--disable_rdkit_warnings",
    ])

    run([
        sys.executable,
        str(script_dir / "g1_prepare_targets.py"),
        "--input_table", str(input_table),
        "--out_dir", str(y_dir),
        "--smiles_col", "canonical_SMILES_RDKit",
        "--y_col", "Tox_original",
    ])

    g1_graph = g1_asset_dir / "g1_graph_assets.pt"
    y_csv = y_dir / "g1_y_experimental.csv"

    # 3) G1 validation.
    # Use the corrected public G1 validation trainer.
    g1_cmd = [
        sys.executable,
        str(script_dir / "g1_run_multisplit.py"),
        "--trainer", "g1_train_validation.py",
        "--graph_asset", str(g1_graph),
        "--y_csv", str(y_csv),
        "--split_root", str(split_root),
        "--out_dir", str(g1_out),
        "--device", args.device,
    ]
    if args.skip_existing:
        g1_cmd.append("--skip_existing")
    run(g1_cmd)

    # 4) Build G2 raw six-descriptor set asset.
    run([
        sys.executable,
        str(script_dir / "g2_build_assets.py"),
        "--input_table", str(input_table),
        "--out_dir", str(g2_asset_dir),
        "--smiles_col", "canonical_SMILES_RDKit",
        "--disable_rdkit_warnings",
    ])

    g2_x = g2_asset_dir / "g2_compact6_raw.npy"
    g2_rowmap = g2_asset_dir / "g2_rowmap.csv"

    # 5) G2 + descriptor-only validation.
    g2_cmd = [
        sys.executable,
        str(script_dir / "g2_run_multisplit.py"),
        "--trainer", "g2_train_validation.py",
        "--x_raw", str(g2_x),
        "--rowmap_csv", str(g2_rowmap),
        "--y_csv", str(y_csv),
        "--split_root", str(split_root),
        "--out_dir", str(g2_out),
        "--conditions", "g2", "self",
        "--device", args.device,
    ]
    if args.skip_existing:
        g2_cmd.append("--skip_existing")
    run(g2_cmd)

    print("\n[FATHEAD MINNOW VALIDATION COMPLETE]")
    print(f"RF summary : {rf_out / '02_rf_hierarchical_summary.csv'}")
    print(f"G1 summary : {g1_out / 'g1_validation_hierarchical_summary.csv'}")
    print(f"G2 summary : {g2_out / 'g2_validation_hierarchical_summary.csv'}")
    print("[Frozen TEST evaluated] NO")


if __name__ == "__main__":
    main()
