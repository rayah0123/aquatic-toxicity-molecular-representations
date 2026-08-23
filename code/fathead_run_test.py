#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""One-shot frozen TEST evaluation for the Fathead minnow robustness endpoint."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def run(cmd):
    print("\n[RUN]", " ".join(map(str, cmd)), flush=True)
    subprocess.run(cmd, check=True)


def main():
    script_dir = Path(__file__).resolve().parent
    root = script_dir.parent

    input_table = root / "03_processed" / "TEST" / "TEST_Fathead_96h_LC50_modeling_pool.csv"
    split_root = root / "04_splits" / "TEST_Fathead" / "scaffold_split_murcko_topology_refined"

    asset_root = root / "02_intermediate" / "TEST" / "LC50" / "model_assets"
    g1_graph = asset_root / "G1" / "g1_graph_assets.pt"
    y_csv = asset_root / "target" / "g1_y_experimental.csv"
    g2_x = asset_root / "G2" / "g2_compact6_raw.npy"
    g2_rowmap = asset_root / "G2" / "g2_rowmap.csv"

    result_root = root / "07_results" / "Fathead"
    rf_val = result_root / "rf_validation"
    g1_val = result_root / "g1_validation"
    g2_val = result_root / "g2_validation"

    rf_summary = rf_val / "02_rf_hierarchical_summary.csv"
    g1_summary = g1_val / "g1_validation_hierarchical_summary.csv"
    g2_summary = g2_val / "g2_validation_hierarchical_summary.csv"
    g1_manifest = g1_val / "g1_model_manifest.csv"
    g2_manifest = g2_val / "g2_model_manifest.csv"

    required = [
        input_table, rf_summary, g1_summary, g2_summary,
        g1_manifest, g2_manifest, g1_graph, y_csv, g2_x, g2_rowmap,
    ]
    for p in required:
        if not p.exists():
            raise FileNotFoundError(p)

    d = pd.read_csv(input_table, encoding="utf-8-sig", low_memory=False)
    if len(d) != 821:
        raise RuntimeError(f"Expected Fish N=821, found {len(d)}")

    g1m = pd.read_csv(g1_manifest, encoding="utf-8-sig")
    g2m = pd.read_csv(g2_manifest, encoding="utf-8-sig")
    if len(g1m) != 25:
        raise RuntimeError(f"Expected 25 G1 frozen bundles, found {len(g1m)}")
    if len(g2m) != 50:
        raise RuntimeError(f"Expected 50 G2/self frozen bundles, found {len(g2m)}")
    if "test_evaluated" in g1m.columns and g1m["test_evaluated"].astype(str).str.lower().isin(["true", "1"]).any():
        raise RuntimeError("G1 manifest indicates TEST was already evaluated during validation.")
    if "test_evaluated" in g2m.columns and g2m["test_evaluated"].astype(str).str.lower().isin(["true", "1"]).any():
        raise RuntimeError("G2 manifest indicates TEST was already evaluated during validation.")

    freeze = {
        "stage": "Fathead pre-TEST freeze",
        "n_total": 821,
        "scientific_spec_changed_after_validation": False,
        "test_opened_before_this_runner": False,
        "files": {str(p.relative_to(root)): sha256(p) for p in required},
    }
    freeze_path = result_root / "fathead_test_freeze.json"
    freeze_path.write_text(json.dumps(freeze, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[PRE-TEST FREEZE] {freeze_path}")

    run([
        sys.executable, str(script_dir / "fathead_baseline_evaluate_test.py"),
        "--input_table", str(input_table),
        "--split_root", str(split_root),
        "--out_dir", str(result_root / "baseline_test"),
    ])

    run([
        sys.executable, str(script_dir / "daphnia_rf_evaluate_test.py"),
        "--input_table", str(input_table),
        "--split_root", str(split_root),
        "--out_dir", str(result_root / "rf_test"),
        "--disable_rdkit_warnings",
    ])

    run([
        sys.executable, str(script_dir / "g1_evaluate_test.py"),
        "--manifest", str(g1_manifest),
        "--graph_asset", str(g1_graph),
        "--y_csv", str(y_csv),
        "--split_root", str(split_root),
        "--out_dir", str(result_root / "g1_test"),
        "--device", "cuda",
    ])

    run([
        sys.executable, str(script_dir / "g2_evaluate_test.py"),
        "--manifest", str(g2_manifest),
        "--x_raw", str(g2_x),
        "--rowmap_csv", str(g2_rowmap),
        "--y_csv", str(y_csv),
        "--split_root", str(split_root),
        "--out_dir", str(result_root / "g2_test"),
        "--device", "cuda",
    ])

    print("\n[FATHEAD MINNOW TEST COMPLETE]")
    print(result_root / "baseline_test" / "02_baseline_frozen_test_hierarchical_summary.csv")
    print(result_root / "rf_test" / "03_rf_frozen_test_hierarchical_summary.csv")
    print(result_root / "g1_test" / "g1_frozen_test_hierarchical_summary.csv")
    print(result_root / "g2_test" / "g2_frozen_test_hierarchical_summary.csv")
    print("[STATUS] Fish TEST is now OPENED and evaluated. Do not retune from these results.")


if __name__ == "__main__":
    main()
