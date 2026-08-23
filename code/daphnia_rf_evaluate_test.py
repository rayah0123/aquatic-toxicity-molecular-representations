#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
daphnia_rf_evaluate_test.py

Frozen TEST evaluation for already-frozen RF representations:
- Morgan-RF
- Descriptor-RF

This script imports rf_baselines_validation and reuses its exact
representation builders, TRAIN-only descriptor preprocessing, RF factory,
split loader, and regression metrics. No RF hyperparameter tuning occurs here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from rdkit import RDLogger, rdBase

import rf_baselines as rfcore


DEFAULT_SPLIT_SEEDS = [42, 123, 777, 2025, 3407]
DEFAULT_MODEL_SEEDS = [42, 123, 777, 2025, 3407]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input_table",
        default="../03_processed/TEST/TEST_Daphnia_modeling_pool_no_ADORE_overlap.csv",
    )
    p.add_argument(
        "--split_root",
        default="../04_splits/TEST_Daphnia/scaffold_split_murcko_topology_refined",
    )
    p.add_argument(
        "--out_dir",
        default="../07_results/Daphnia/rf_test",
    )
    p.add_argument("--smiles_col", default="canonical_SMILES_RDKit")
    p.add_argument("--y_col", default="Tox_original")
    p.add_argument("--split_seeds", nargs="*", type=int, default=DEFAULT_SPLIT_SEEDS)
    p.add_argument("--model_seeds", nargs="*", type=int, default=DEFAULT_MODEL_SEEDS)

    p.add_argument("--radius", type=int, default=4)
    p.add_argument("--n_bits", type=int, default=2048)
    p.add_argument("--n_estimators", type=int, default=500)
    p.add_argument("--max_depth", type=int, default=None)
    p.add_argument("--min_samples_leaf", type=int, default=1)
    p.add_argument("--max_features", type=float, default=1.0)
    p.add_argument("--n_jobs", type=int, default=-1)

    p.add_argument("--disable_rdkit_warnings", action="store_true")
    return p.parse_args()


def resolve(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (Path(__file__).resolve().parent / p).resolve()


def assert_frozen_settings(args):
    expected = {
        "radius": 4,
        "n_bits": 2048,
        "n_estimators": 500,
        "max_depth": None,
        "min_samples_leaf": 1,
        "max_features": 1.0,
    }
    got = {
        "radius": args.radius,
        "n_bits": args.n_bits,
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "min_samples_leaf": args.min_samples_leaf,
        "max_features": args.max_features,
    }
    if got != expected:
        raise RuntimeError(
            "Frozen RF specification changed.\n"
            f"Expected: {expected}\nGot: {got}\n"
            "Do not tune RF settings after TEST is opened."
        )


def make_hierarchical_summary(run_df):
    metrics = [
        "test_RMSE", "test_MAE", "test_R2", "test_Pearson", "test_Spearman",
        "test_bias_pred_minus_true", "test_pred_to_true_sd_ratio",
        "test_calibration_slope_ytrue_on_ypred",
    ]
    splitmeans = (
        run_df.groupby(["model", "split_seed"], as_index=False)[metrics]
        .mean(numeric_only=True)
    )
    counts = (
        run_df.groupby(["model", "split_seed"], as_index=False)[
            ["n_total", "n_train", "n_val", "n_test"]
        ]
        .first()
    )
    splitmeans = counts.merge(splitmeans, on=["model", "split_seed"], how="left")

    rows = []
    for model, g in splitmeans.groupby("model"):
        row = {
            "model": model,
            "n_total": int(g["n_total"].iloc[0]),
            "n_structural_splits": int(g["split_seed"].nunique()),
            "model_seeds_per_split": int(
                run_df.loc[run_df["model"].eq(model), "model_seed"].nunique()
            ),
            "train_n_min": int(g["n_train"].min()),
            "train_n_max": int(g["n_train"].max()),
            "val_n_min": int(g["n_val"].min()),
            "val_n_max": int(g["n_val"].max()),
            "test_n_min": int(g["n_test"].min()),
            "test_n_max": int(g["n_test"].max()),
        }
        for m in metrics:
            row[f"{m}_splitmean_mean"] = float(g[m].mean())
            row[f"{m}_splitmean_std"] = float(g[m].std(ddof=1))
            row[f"{m}_splitmean_median"] = float(g[m].median())
        rows.append(row)

    summary = (
        pd.DataFrame(rows)
        .sort_values("test_RMSE_splitmean_mean")
        .reset_index(drop=True)
    )
    return summary, splitmeans


def main():
    args = parse_args()
    assert_frozen_settings(args)

    if args.disable_rdkit_warnings:
        RDLogger.DisableLog("rdApp.*")

    input_table = resolve(args.input_table)
    split_root = resolve(args.split_root)
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_table, encoding="utf-8-sig", low_memory=False).reset_index(drop=True)
    for c in [args.smiles_col, args.y_col]:
        if c not in df.columns:
            raise KeyError(f"Missing required column: {c}")

    smiles = df[args.smiles_col].astype(str).tolist()
    y_all = pd.to_numeric(df[args.y_col], errors="raise").to_numpy(dtype=np.float64)
    if not np.isfinite(y_all).all():
        raise ValueError("Non-finite target values found.")

    print("[1/3] Building Count Morgan r4/2048 ...", flush=True)
    X_morgan = rfcore.build_count_morgan(smiles, radius=args.radius, n_bits=args.n_bits)

    print("[2/3] Building RDKit descriptor pool ...", flush=True)
    X_desc_raw, descriptor_names, descriptor_global_qc = rfcore.build_rdkit_descriptor_raw(smiles)
    descriptor_global_qc.to_csv(
        out_dir / "00_descriptor_global_raw_qc.csv",
        index=False,
        encoding="utf-8-sig",
    )

    rf_args = SimpleNamespace(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        max_features=args.max_features,
        n_jobs=args.n_jobs,
    )

    run_rows = []
    pred_rows = []
    descriptor_qc_rows = []
    split_count_rows = []

    split_seeds = [int(x) for x in args.split_seeds]
    model_seeds = [int(x) for x in args.model_seeds]
    total = len(split_seeds) * len(model_seeds) * 2
    done = 0

    print("[3/3] Frozen TEST RF evaluation:", flush=True)

    for split_seed in split_seeds:
        idx = rfcore.load_split_indices(split_root, split_seed, len(df))
        tr, va, te = idx["train"], idx["val"], idx["test"]

        split_count_rows.append({
            "split_seed": int(split_seed),
            "n_total": int(len(df)),
            "n_train": int(len(tr)),
            "n_val": int(len(va)),
            "n_test": int(len(te)),
        })

        y_train = y_all[tr]
        y_test = y_all[te]

        kept_idx, train_medians, kept_names, dqc = (
            rfcore.fit_descriptor_preprocessor_train_only(
                X_desc_raw, tr, descriptor_names
            )
        )
        X_desc_train = rfcore.transform_descriptors(
            X_desc_raw, tr, kept_idx, train_medians
        )
        X_desc_test = rfcore.transform_descriptors(
            X_desc_raw, te, kept_idx, train_medians
        )

        descriptor_qc_rows.append({
            "split_seed": int(split_seed),
            **dqc,
            "kept_descriptor_names": "|".join(kept_names),
        })

        X_morgan_train = X_morgan[tr]
        X_morgan_test = X_morgan[te]

        for model_seed in model_seeds:
            rf_morgan = rfcore.make_rf(rf_args, model_seed)
            rf_morgan.fit(X_morgan_train, y_train)
            pred_morgan = rf_morgan.predict(X_morgan_test)
            mm = rfcore.regression_metrics(y_test, pred_morgan)

            run_rows.append({
                "model": "Morgan-RF",
                "split_seed": int(split_seed),
                "model_seed": int(model_seed),
                "n_total": int(len(df)),
                "n_train": int(len(tr)),
                "n_val": int(len(va)),
                "n_test": int(len(te)),
                "n_features": int(X_morgan_train.shape[1]),
                **{f"test_{k}": v for k, v in mm.items()},
            })
            for pos, gi in enumerate(te):
                pred_rows.append({
                    "model": "Morgan-RF",
                    "split_seed": int(split_seed),
                    "model_seed": int(model_seed),
                    "global_row_index": int(gi),
                    "y_true": float(y_test[pos]),
                    "y_pred": float(pred_morgan[pos]),
                })

            done += 1
            print(f"{int(round(100*done/total))}%", end=" ", flush=True)

            rf_desc = rfcore.make_rf(rf_args, model_seed)
            rf_desc.fit(X_desc_train, y_train)
            pred_desc = rf_desc.predict(X_desc_test)
            md = rfcore.regression_metrics(y_test, pred_desc)

            run_rows.append({
                "model": "Descriptor-RF",
                "split_seed": int(split_seed),
                "model_seed": int(model_seed),
                "n_total": int(len(df)),
                "n_train": int(len(tr)),
                "n_val": int(len(va)),
                "n_test": int(len(te)),
                "n_features": int(X_desc_train.shape[1]),
                **{f"test_{k}": v for k, v in md.items()},
            })
            for pos, gi in enumerate(te):
                pred_rows.append({
                    "model": "Descriptor-RF",
                    "split_seed": int(split_seed),
                    "model_seed": int(model_seed),
                    "global_row_index": int(gi),
                    "y_true": float(y_test[pos]),
                    "y_pred": float(pred_desc[pos]),
                })

            done += 1
            print(
                f"{int(round(100*done/total))}%",
                end=" " if done < total else "\n",
                flush=True,
            )

    run_df = pd.DataFrame(run_rows)
    pred_df = pd.DataFrame(pred_rows)
    desc_qc_df = pd.DataFrame(descriptor_qc_rows)
    counts_df = pd.DataFrame(split_count_rows)
    summary_df, splitmeans_df = make_hierarchical_summary(run_df)

    run_df.to_csv(out_dir / "01_rf_frozen_test_all_runs.csv", index=False, encoding="utf-8-sig")
    splitmeans_df.to_csv(out_dir / "02_rf_frozen_test_splitmeans.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(out_dir / "03_rf_frozen_test_hierarchical_summary.csv", index=False, encoding="utf-8-sig")
    pred_df.to_csv(out_dir / "04_rf_frozen_test_predictions.csv", index=False, encoding="utf-8-sig")
    desc_qc_df.to_csv(out_dir / "05_descriptor_trainonly_preprocess_qc.csv", index=False, encoding="utf-8-sig")
    counts_df.to_csv(out_dir / "06_split_sample_counts.csv", index=False, encoding="utf-8-sig")

    meta = {
        "module": "daphnia_rf_evaluate_test",
        "scientific_source_module": "rf_baselines_validation",
        "n_total": int(len(df)),
        "split_seeds": split_seeds,
        "model_seeds": model_seeds,
        "test_evaluated": True,
        "model_selection_or_tuning_in_this_script": False,
        "morgan": {
            "type": "Count Morgan",
            "radius": int(args.radius),
            "n_bits": int(args.n_bits),
            "includeChirality": False,
        },
        "descriptor": {
            "raw": "RDKit Descriptors.CalcMolDescriptors",
            "preprocessing": "TRAIN-only all-missing removal + median imputation + zero-variance removal; no scaling",
        },
        "rf": {
            "n_estimators": int(args.n_estimators),
            "max_depth": args.max_depth,
            "min_samples_leaf": int(args.min_samples_leaf),
            "max_features": float(args.max_features),
            "bootstrap": True,
            "hyperparameter_tuning": False,
        },
        "summary_policy": "mean of 5 RF model seeds within each structural split, then summary across 5 structural splits",
        "rdkit_version": rdBase.rdkitVersion,
    }
    (out_dir / "rf_test_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"[DONE] {out_dir}")
    print(f"[N] total={len(df)}")
    print("[TEST evaluated] YES")
    print("[model tuning] NO")


if __name__ == "__main__":
    main()
