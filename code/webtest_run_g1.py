#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate G1 on the fixed WebTEST cohort and structural splits using
the established G1 architecture and hyperparameters.

Run 5 structural splits by 5 model seeds for validation; evaluate held-out
TEST only after all 25 model bundles are available. Graph assets and completed
validation runs may be reused.
This G1 comparison followed inspection of other WebTEST TEST results;
G1 is not tuned using those results."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EXPECTED_N = 119276
SPLIT_SEEDS = [42, 123, 777, 2025, 3407]
MODEL_SEEDS = [42, 123, 777, 2025, 3407]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--skip_test_if_summary_exists",
        action="store_true",
        help="Do not reevaluate TEST if the final G1 TEST summary already exists.",
    )
    return p.parse_args()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def run(cmd):
    print("\n[RUN]", " ".join(map(str, cmd)), flush=True)
    subprocess.run(cmd, check=True)


def require(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def make_target_asset(input_table: Path, out_csv: Path, meta_json: Path):
    d = pd.read_csv(input_table, encoding="utf-8-sig", low_memory=False).reset_index(drop=True)
    if len(d) != EXPECTED_N:
        raise RuntimeError(f"Expected WebTEST N={EXPECTED_N}, found {len(d)}")
    for c in ["SMILES_RDKit", "target_neglog10_molar"]:
        if c not in d.columns:
            raise KeyError(f"Missing required WebTEST column: {c}")

    y = pd.to_numeric(d["target_neglog10_molar"], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(y).all():
        raise RuntimeError("Non-finite target_neglog10_molar values found.")

    out = pd.DataFrame({
        "global_row_index": np.arange(len(d), dtype=np.int64),
        "SMILES": d["SMILES_RDKit"].astype(str).to_numpy(),
        "y": y,
    })
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False, encoding="utf-8-sig")

    meta = {
        "asset": "G1 WebTEST surrogate target",
        "status": "DETERMINISTIC_TARGET_ASSET",
        "source_table": str(input_table),
        "source_table_sha256": sha256(input_table),
        "n_rows": int(len(out)),
        "smiles_source_column": "SMILES_RDKit",
        "target_source_column": "target_neglog10_molar",
        "target_definition": "-log10(LC50 [mol/L])",
        "label_provenance": "WebTEST Daphnia LC50 consensus-only surrogate labels",
        "global_row_index": "0..N-1, exact frozen-table row order",
        "target_csv_sha256": sha256(out_csv),
    }
    meta_json.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


def validate_manifest(manifest: Path):
    m = pd.read_csv(manifest, encoding="utf-8-sig", low_memory=False)
    if len(m) != 25:
        raise RuntimeError(f"Expected 25 G1 validation bundles, found {len(m)}")

    if {"split_seed", "model_seed"}.issubset(m.columns):
        pairs = set(zip(m["split_seed"].astype(int), m["model_seed"].astype(int)))
        expected = {(s, ms) for s in SPLIT_SEEDS for ms in MODEL_SEEDS}
        if pairs != expected:
            missing = sorted(expected - pairs)
            extra = sorted(pairs - expected)
            raise RuntimeError(
                f"Manifest run-grid mismatch. missing={missing[:5]}, extra={extra[:5]}"
            )

    if "test_evaluated" in m.columns:
        opened = (
            m["test_evaluated"].astype(str).str.strip().str.lower()
            .isin(["true", "1", "yes"])
        )
        if opened.any():
            raise RuntimeError("Validation manifest indicates TEST evaluation in training bundles.")
    return m


def main():
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    root = script_dir.parent

    input_table = (
        root / "03_processed" / "WebTEST"
        / "WebTEST_Daphnia_consensus_unique_connectivity_frozen.csv"
    )
    split_root = (
        root / "04_splits" / "WebTEST_Daphnia"
        / "scaffold_split_murcko_topology_refined"
    )

    feature_dir = root / "05_features" / "G1_WebTEST_revised"
    result_root = root / "07_results" / "WebTEST_revised"
    val_dir = result_root / "g1_validation"
    test_dir = result_root / "g1_test"

    graph_asset = feature_dir / "g1_graph_assets.pt"
    target_csv = feature_dir / "g1_y_webtest_surrogate.csv"
    target_meta = feature_dir / "g1_y_webtest_surrogate_meta.json"
    manifest = val_dir / "g1_model_manifest.csv"
    val_summary = val_dir / "g1_validation_hierarchical_summary.csv"
    test_summary = test_dir / "g1_frozen_test_hierarchical_summary.csv"

    scripts = {
        "builder": script_dir / "g1_build_graph_assets.py",
        "trainer": script_dir / "g1_train_validation.py",
        "multi": script_dir / "g1_run_multisplit.py",
        "test": script_dir / "g1_evaluate_test.py",
        "core": script_dir / "g1_core.py",
    }

    require(input_table)
    require(split_root)
    for p in scripts.values():
        require(p)
    for seed in SPLIT_SEEDS:
        require(split_root / f"split_seed{seed}" / "split_membership.csv")

    feature_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    print("[1/5] G1 graph asset", flush=True)
    if not graph_asset.exists():
        run([
            sys.executable, str(scripts["builder"]),
            "--input_table", str(input_table),
            "--out_dir", str(feature_dir),
            "--smiles_col", "SMILES_RDKit",
            "--disable_rdkit_warnings",
        ])
    else:
        print(f"[SKIP existing] {graph_asset}", flush=True)
    require(graph_asset)

    print("[2/5] WebTEST surrogate target asset", flush=True)
    make_target_asset(input_table, target_csv, target_meta)
    require(target_csv)

    provenance = {
        "stage": "WebTEST revised G1 late-added benchmark completion",
        "status": "LATE_ADDED_FIXED_BENCHMARK",
        "reason": (
            "G1 was part of the manuscript representation comparison but was not "
            "re-evaluated when the revised WebTEST analysis was rebuilt around the "
            "G2-vs-no-edge relational experiment."
        ),
        "other_WebTEST_TEST_results_had_already_been_observed": True,
        "G1_specification_newly_tuned_for_WebTEST": False,
        "G1_specification_source": "frozen G1 production pipeline",
        "validation_before_G1_TEST_evaluation": True,
        "post_TEST_retuning_allowed": False,
        "input_table": str(input_table),
        "input_sha256": sha256(input_table),
        "graph_asset": str(graph_asset),
        "graph_asset_sha256": sha256(graph_asset),
        "target_asset": str(target_csv),
        "target_asset_sha256": sha256(target_csv),
        "split_seeds": SPLIT_SEEDS,
        "model_seeds": MODEL_SEEDS,
    }
    provenance_path = val_dir / "g1_webtest_provenance.json"
    provenance_path.write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("[3/5] G1 validation: 5 structural splits x 5 model seeds", flush=True)
    run([
        sys.executable, str(scripts["multi"]),
        "--trainer", scripts["trainer"].name,
        "--graph_asset", str(graph_asset),
        "--y_csv", str(target_csv),
        "--split_root", str(split_root),
        "--out_dir", str(val_dir),
        "--split_seeds", *[str(x) for x in SPLIT_SEEDS],
        "--model_seeds", *[str(x) for x in MODEL_SEEDS],
        "--device", args.device,
        "--skip_existing",
    ])

    require(manifest)
    require(val_summary)
    validate_manifest(manifest)

    print("[4/5] Frozen G1 TEST evaluation", flush=True)
    if args.skip_test_if_summary_exists and test_summary.exists():
        print(f"[SKIP existing TEST summary] {test_summary}", flush=True)
    else:
        run([
            sys.executable, str(scripts["test"]),
            "--manifest", str(manifest),
            "--graph_asset", str(graph_asset),
            "--y_csv", str(target_csv),
            "--split_root", str(split_root),
            "--out_dir", str(test_dir),
            "--device", args.device,
        ])

    require(test_summary)

    print("[5/5] COMPLETE", flush=True)
    print(f"Validation summary : {val_summary}", flush=True)
    print(f"Frozen TEST summary: {test_summary}", flush=True)
    print(f"Provenance         : {provenance_path}", flush=True)
    print("\n[G1 WEBTEST FROZEN TEST SUMMARY]", flush=True)
    print(pd.read_csv(test_summary, encoding="utf-8-sig").to_string(index=False), flush=True)
    print(
        "\n[IMPORTANT] Do not tune G1 after this TEST result. "
        "Next step is aggregation into manuscript Table 2.",
        flush=True,
    )


if __name__ == "__main__":
    main()
