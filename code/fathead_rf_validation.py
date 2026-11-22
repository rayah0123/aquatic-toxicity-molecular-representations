#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate random-forest and train-mean/median baselines on the fixed
316-chemical modeling table and structural validation splits.

Count Morgan: radius 4, 2048 dimensions, no chirality.
RDKit descriptors: remove train-all-missing/constant columns, impute with
TRAIN medians, and apply the resulting transformation to validation.
RF configuration: 500 trees, max_features=1.0, min_samples_leaf=1,
bootstrap=True, unrestricted depth; five model seeds.
Summaries average model seeds within each split before aggregating splits.
The held-out TEST partition is not evaluated here."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from rdkit import Chem, rdBase, RDLogger
from rdkit.Chem import Descriptors, rdFingerprintGenerator

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


DEFAULT_SPLIT_SEEDS = [42, 123, 777, 2025, 3407]
DEFAULT_MODEL_SEEDS = [42, 123, 777, 2025, 3407]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Validation-only random-forest baselines: mean/median + Morgan-RF + RDKit-descriptor-RF"
    )
    p.add_argument(
        "--input_table",
        default="../03_processed/TEST/TEST_Fathead_96h_LC50_modeling_pool.csv",
    )
    p.add_argument(
        "--split_root",
        default="../04_splits/TEST_Fathead/scaffold_split_murcko_topology_refined",
    )
    p.add_argument(
        "--out_dir",
        default="../07_results/Fathead/rf_validation",
    )
    p.add_argument(
        "--reference_g2_summary",
        default=None,
        help="Optional reference G2 summary. Read-only; G2 is not rerun.",
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


def resolve_path(path_str: str) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p
    script_dir = Path(__file__).resolve().parent
    return (script_dir / p).resolve()


def load_split_indices(split_root: Path, seed: int, n_rows: int) -> Dict[str, np.ndarray]:
    path = split_root / f"split_seed{seed}" / "split_membership.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing split membership: {path}")

    m = pd.read_csv(path, encoding="utf-8-sig")
    if "split" not in m.columns:
        raise KeyError(f"Missing 'split' column in {path}")

    idx_col = None
    for candidate in ["original_row_index", "row_idx", "index"]:
        if candidate in m.columns:
            idx_col = candidate
            break
    if idx_col is None:
        raise KeyError(
            f"Could not find row-index column in {path}. "
            "Expected one of: original_row_index, row_idx, index"
        )

    out: Dict[str, np.ndarray] = {}
    for name in ["train", "val", "test"]:
        arr = (
            pd.to_numeric(
                m.loc[m["split"].astype(str).str.lower().eq(name), idx_col],
                errors="raise",
            )
            .astype(int)
            .to_numpy()
        )
        out[name] = np.sort(arr)

    all_idx = np.concatenate([out["train"], out["val"], out["test"]])
    if len(all_idx) != n_rows:
        raise RuntimeError(f"split seed {seed}: assigned={len(all_idx)} expected={n_rows}")
    if len(np.unique(all_idx)) != n_rows:
        raise RuntimeError(f"split seed {seed}: duplicate row assignment")
    if all_idx.min() < 0 or all_idx.max() >= n_rows:
        raise RuntimeError(f"split seed {seed}: row index out of range")

    return out


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    pearson = float(pd.Series(y_true).corr(pd.Series(y_pred), method="pearson"))
    spearman = float(pd.Series(y_true).corr(pd.Series(y_pred), method="spearman"))

    bias = float(np.mean(y_pred - y_true))
    true_std = float(np.std(y_true, ddof=1)) if len(y_true) > 1 else np.nan
    pred_std = float(np.std(y_pred, ddof=1)) if len(y_pred) > 1 else np.nan
    pred_to_true_sd_ratio = (
        float(pred_std / true_std)
        if np.isfinite(true_std) and true_std > 0 and np.isfinite(pred_std)
        else np.nan
    )

    # Calibration diagnostic: y_true = intercept + slope * y_pred
    # slope > 1 often indicates prediction-range compression.
    if len(y_true) >= 2 and np.isfinite(pred_std) and pred_std > 0:
        slope, intercept = np.polyfit(y_pred, y_true, deg=1)
        slope = float(slope)
        intercept = float(intercept)
    else:
        slope, intercept = np.nan, np.nan

    return {
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "Pearson": pearson,
        "Spearman": spearman,
        "bias_pred_minus_true": bias,
        "true_std": true_std,
        "pred_std": pred_std,
        "pred_to_true_sd_ratio": pred_to_true_sd_ratio,
        "calibration_slope_ytrue_on_ypred": slope,
        "calibration_intercept_ytrue_on_ypred": intercept,
        "pred_min": float(np.min(y_pred)),
        "pred_max": float(np.max(y_pred)),
        "true_min": float(np.min(y_true)),
        "true_max": float(np.max(y_true)),
    }


def build_count_morgan(
    smiles: Sequence[str], radius: int, n_bits: int
) -> np.ndarray:
    gen = rdFingerprintGenerator.GetMorganGenerator(
        radius=radius,
        fpSize=n_bits,
        includeChirality=False,
    )
    X = np.zeros((len(smiles), n_bits), dtype=np.float32)

    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            raise ValueError(f"Invalid SMILES at row {i}: {smi}")
        fp = gen.GetCountFingerprint(mol)
        for bit_idx, count in fp.GetNonzeroElements().items():
            bit_idx = int(bit_idx)
            if 0 <= bit_idx < n_bits:
                X[i, bit_idx] = float(count)

    return X


def build_rdkit_descriptor_raw(
    smiles: Sequence[str],
) -> Tuple[np.ndarray, List[str], pd.DataFrame]:
    """
    Build raw full RDKit descriptor matrix.
    No imputation / dropping is done here.
    Non-finite values become NaN.
    """
    rows: List[Dict[str, float]] = []

    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            raise ValueError(f"Invalid SMILES at row {i}: {smi}")

        d = Descriptors.CalcMolDescriptors(mol, missingVal=np.nan, silent=True)
        clean: Dict[str, float] = {}
        for k, v in d.items():
            try:
                fv = float(v)
            except Exception:
                fv = np.nan
            if not np.isfinite(fv):
                fv = np.nan
            clean[str(k)] = fv
        rows.append(clean)

    ddf = pd.DataFrame(rows, dtype=float)
    descriptor_names = ddf.columns.astype(str).tolist()
    X = ddf.to_numpy(dtype=np.float64)

    per_descriptor_qc = pd.DataFrame({
        "descriptor": descriptor_names,
        "n_nonfinite_all": np.sum(~np.isfinite(X), axis=0).astype(int),
        "finite_fraction_all": np.mean(np.isfinite(X), axis=0).astype(float),
    })

    return X, descriptor_names, per_descriptor_qc


def fit_descriptor_preprocessor_train_only(
    X_raw: np.ndarray,
    train_idx: np.ndarray,
    descriptor_names: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray, List[str], Dict]:
    """
    Returns:
      kept_col_idx
      train_medians corresponding to kept cols
      kept_names
      qc dict
    """
    Xtr = np.asarray(X_raw[train_idx], dtype=np.float64)

    finite_any = np.isfinite(Xtr).any(axis=0)
    cols_1 = np.where(finite_any)[0]
    if len(cols_1) == 0:
        raise RuntimeError("All RDKit descriptor columns are all-missing on TRAIN.")

    X1 = Xtr[:, cols_1]
    medians = np.nanmedian(X1, axis=0)

    # Defensive: np.nanmedian can still return NaN only if all values are NaN,
    # but such columns were already removed.
    good_median = np.isfinite(medians)
    cols_2 = cols_1[good_median]
    medians_2 = medians[good_median]

    X2 = np.asarray(X_raw[train_idx][:, cols_2], dtype=np.float64)
    bad = ~np.isfinite(X2)
    if bad.any():
        X2 = X2.copy()
        rr, cc = np.where(bad)
        X2[rr, cc] = medians_2[cc]

    # Drop zero-variance columns based on TRAIN only.
    variances = np.var(X2, axis=0)
    nonzero_var = np.isfinite(variances) & (variances > 0)
    kept_col_idx = cols_2[nonzero_var]
    kept_medians = medians_2[nonzero_var]
    kept_names = [str(descriptor_names[i]) for i in kept_col_idx]

    qc = {
        "n_descriptor_raw": int(X_raw.shape[1]),
        "n_drop_train_all_missing": int(np.sum(~finite_any)),
        "n_drop_nonfinite_train_median": int(np.sum(~good_median)),
        "n_drop_train_zero_variance": int(np.sum(~nonzero_var)),
        "n_descriptor_final": int(len(kept_col_idx)),
        "n_train_rows": int(len(train_idx)),
    }

    return kept_col_idx, kept_medians, kept_names, qc


def transform_descriptors(
    X_raw: np.ndarray,
    row_idx: np.ndarray,
    kept_col_idx: np.ndarray,
    train_medians: np.ndarray,
) -> np.ndarray:
    X = np.asarray(X_raw[row_idx][:, kept_col_idx], dtype=np.float64)
    bad = ~np.isfinite(X)
    if bad.any():
        X = X.copy()
        rr, cc = np.where(bad)
        X[rr, cc] = train_medians[cc]
    if not np.isfinite(X).all():
        raise RuntimeError("Non-finite descriptor values remain after TRAIN-fitted imputation.")
    return X.astype(np.float32, copy=False)


def make_rf(args: argparse.Namespace, random_state: int) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=int(args.n_estimators),
        max_depth=args.max_depth,
        min_samples_leaf=int(args.min_samples_leaf),
        max_features=float(args.max_features),
        bootstrap=True,
        random_state=int(random_state),
        n_jobs=int(args.n_jobs),
    )


def target_stats(y: np.ndarray) -> Dict[str, float]:
    y = np.asarray(y, dtype=float)
    return {
        "n": int(len(y)),
        "mean": float(np.mean(y)),
        "std": float(np.std(y, ddof=1)) if len(y) > 1 else np.nan,
        "min": float(np.min(y)),
        "q05": float(np.quantile(y, 0.05)),
        "q25": float(np.quantile(y, 0.25)),
        "median": float(np.median(y)),
        "q75": float(np.quantile(y, 0.75)),
        "q95": float(np.quantile(y, 0.95)),
        "max": float(np.max(y)),
    }


def hierarchical_summary(run_df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "val_RMSE", "val_MAE", "val_R2", "val_Pearson", "val_Spearman",
        "val_bias_pred_minus_true",
        "val_pred_to_true_sd_ratio",
        "val_calibration_slope_ytrue_on_ypred",
    ]

    model_seed_rows = run_df.loc[run_df["model_seed"].notna()].copy()
    fixed_rows = run_df.loc[run_df["model_seed"].isna()].copy()

    rows: List[Dict] = []

    # Stochastic RF models: average model seeds within structural split first.
    if len(model_seed_rows):
        split_means = (
            model_seed_rows.groupby(["split_seed", "model"], as_index=False)[metric_cols].mean()
        )
        for model, g in split_means.groupby("model"):
            row = {
                "model": model,
                "n_structural_splits": int(g["split_seed"].nunique()),
                "model_seeds_per_split": int(
                    model_seed_rows.loc[model_seed_rows["model"].eq(model), "model_seed"].nunique()
                ),
            }
            for m in metric_cols:
                row[f"{m}_splitmean_mean"] = float(g[m].mean())
                row[f"{m}_splitmean_std"] = float(g[m].std(ddof=1)) if len(g) > 1 else np.nan
                row[f"{m}_splitmean_median"] = float(g[m].median())
            rows.append(row)

    # Deterministic baselines: one row per structural split.
    if len(fixed_rows):
        for model, g in fixed_rows.groupby("model"):
            row = {
                "model": model,
                "n_structural_splits": int(g["split_seed"].nunique()),
                "model_seeds_per_split": 0,
            }
            for m in metric_cols:
                row[f"{m}_splitmean_mean"] = float(g[m].mean())
                row[f"{m}_splitmean_std"] = float(g[m].std(ddof=1)) if len(g) > 1 else np.nan
                row[f"{m}_splitmean_median"] = float(g[m].median())
            rows.append(row)

    out = pd.DataFrame(rows)
    return out.sort_values("val_RMSE_splitmean_mean", ascending=True).reset_index(drop=True)


def model_seed_variability(run_df: pd.DataFrame) -> pd.DataFrame:
    d = run_df.loc[run_df["model_seed"].notna()].copy()
    rows = []
    for (split_seed, model), g in d.groupby(["split_seed", "model"]):
        vals = g["val_RMSE"].to_numpy(dtype=float)
        rows.append({
            "split_seed": int(split_seed),
            "model": model,
            "n_model_seeds": int(len(vals)),
            "val_RMSE_modelseed_mean": float(np.mean(vals)),
            "val_RMSE_modelseed_std": float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan,
            "val_RMSE_modelseed_min": float(np.min(vals)),
            "val_RMSE_modelseed_max": float(np.max(vals)),
        })
    return pd.DataFrame(rows)


def build_comparison_with_reference_g2(
    rf_summary: pd.DataFrame,
    reference_g2_path: Path | None,
) -> pd.DataFrame:
    left = pd.DataFrame({
        "source": "RF_validation",
        "model": rf_summary["model"],
        "relation_mode": np.nan,
        "topk": np.nan,
        "val_RMSE_mean": rf_summary["val_RMSE_splitmean_mean"],
        "val_MAE_mean": rf_summary["val_MAE_splitmean_mean"],
        "val_R2_mean": rf_summary["val_R2_splitmean_mean"],
        "val_Pearson_mean": rf_summary["val_Pearson_splitmean_mean"],
        "val_Spearman_mean": rf_summary["val_Spearman_splitmean_mean"],
    })

    frames = [left]
    if reference_g2_path is not None and reference_g2_path.exists():
        p = pd.read_csv(reference_g2_path, encoding="utf-8-sig")
        right = pd.DataFrame({
            "source": "reference_G2",
            "model": "G2_Morgan2048_GraphSAGE",
            "relation_mode": p["relation_mode"],
            "topk": p["topk"],
            "val_RMSE_mean": p["val_RMSE_splitmean_mean"],
            "val_MAE_mean": p["val_MAE_splitmean_mean"],
            "val_R2_mean": p["val_R2_splitmean_mean"],
            "val_Pearson_mean": p["val_Pearson_splitmean_mean"],
            "val_Spearman_mean": p["val_Spearman_splitmean_mean"],
        })
        frames.append(right)

    out = pd.concat(frames, ignore_index=True)
    return out.sort_values("val_RMSE_mean", ascending=True).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    if args.disable_rdkit_warnings:
        RDLogger.DisableLog("rdApp.*")

    input_table = resolve_path(args.input_table)
    split_root = resolve_path(args.split_root)
    out_dir = resolve_path(args.out_dir)
    reference_g2_path = resolve_path(args.reference_g2_summary) if args.reference_g2_summary else None
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_table, encoding="utf-8-sig", low_memory=False).reset_index(drop=True)

    for col in [args.smiles_col, args.y_col]:
        if col not in df.columns:
            raise KeyError(f"Missing required column: {col}")

    smiles = df[args.smiles_col].astype(str).tolist()
    y_all = pd.to_numeric(df[args.y_col], errors="raise").to_numpy(dtype=np.float64)
    if not np.isfinite(y_all).all():
        raise ValueError("Non-finite target values found.")

    # ---------- Deterministic representation construction ----------
    print("[1/4] Building Count Morgan r4/2048 ...", flush=True)
    X_morgan = build_count_morgan(smiles, radius=args.radius, n_bits=args.n_bits)

    print("[2/4] Calculating raw RDKit descriptors ...", flush=True)
    X_desc_raw, descriptor_names, descriptor_global_qc = build_rdkit_descriptor_raw(smiles)
    descriptor_global_qc.to_csv(
        out_dir / "00B_descriptor_global_raw_qc.csv",
        index=False,
        encoding="utf-8-sig",
    )

    split_seeds = [int(x) for x in args.split_seeds]
    model_seeds = [int(x) for x in args.model_seeds]

    target_rows: List[Dict] = []
    run_rows: List[Dict] = []
    pred_rows: List[Dict] = []
    descriptor_qc_rows: List[Dict] = []

    # Overall y distribution only; TEST-specific y distribution is not used.
    target_rows.append({
        "split_seed": np.nan,
        "partition": "all_821",
        **target_stats(y_all),
        "note": "overall target distribution only",
    })

    total_blocks = len(split_seeds) * len(model_seeds) * 2
    block_i = 0

    print("[3/4] Validation-only RF runs:", flush=True)

    for split_seed in split_seeds:
        idx = load_split_indices(split_root, split_seed, len(df))
        train_idx, val_idx, test_idx = idx["train"], idx["val"], idx["test"]

        y_train = y_all[train_idx]
        y_val = y_all[val_idx]

        target_rows.append({
            "split_seed": int(split_seed),
            "partition": "train",
            **target_stats(y_train),
            "note": "used for training",
        })
        target_rows.append({
            "split_seed": int(split_seed),
            "partition": "val",
            **target_stats(y_val),
            "note": "validation-only; frozen TEST not scored",
        })

        # ---------- deterministic baselines ----------
        mean_pred = np.full(len(val_idx), float(np.mean(y_train)), dtype=float)
        median_pred = np.full(len(val_idx), float(np.median(y_train)), dtype=float)

        for model_name, pred in [
            ("TrainMean", mean_pred),
            ("TrainMedian", median_pred),
        ]:
            m = regression_metrics(y_val, pred)
            run_rows.append({
                "split_seed": int(split_seed),
                "model_seed": np.nan,
                "model": model_name,
                "n_train": int(len(train_idx)),
                "n_val": int(len(val_idx)),
                "n_test_frozen_not_evaluated": int(len(test_idx)),
                "test_evaluated": False,
                **{f"val_{k}": v for k, v in m.items()},
            })
            for pos, global_idx in enumerate(val_idx):
                pred_rows.append({
                    "split_seed": int(split_seed),
                    "model_seed": np.nan,
                    "model": model_name,
                    "global_row_index": int(global_idx),
                    "y_true": float(y_val[pos]),
                    "y_pred": float(pred[pos]),
                })

        # ---------- TRAIN-only descriptor preprocessing ----------
        kept_idx, train_medians, kept_names, dqc = fit_descriptor_preprocessor_train_only(
            X_desc_raw, train_idx, descriptor_names
        )
        X_desc_train = transform_descriptors(
            X_desc_raw, train_idx, kept_idx, train_medians
        )
        X_desc_val = transform_descriptors(
            X_desc_raw, val_idx, kept_idx, train_medians
        )

        descriptor_qc_rows.append({
            "split_seed": int(split_seed),
            **dqc,
            "kept_descriptor_names": "|".join(kept_names),
        })

        X_morgan_train = X_morgan[train_idx]
        X_morgan_val = X_morgan[val_idx]

        for model_seed in model_seeds:
            # Morgan-RF
            rf_morgan = make_rf(args, model_seed)
            rf_morgan.fit(X_morgan_train, y_train)
            pred_morgan = rf_morgan.predict(X_morgan_val)
            mm = regression_metrics(y_val, pred_morgan)

            run_rows.append({
                "split_seed": int(split_seed),
                "model_seed": int(model_seed),
                "model": "Morgan-RF",
                "n_train": int(len(train_idx)),
                "n_val": int(len(val_idx)),
                "n_test_frozen_not_evaluated": int(len(test_idx)),
                "test_evaluated": False,
                "n_features": int(X_morgan_train.shape[1]),
                **{f"val_{k}": v for k, v in mm.items()},
            })
            for pos, global_idx in enumerate(val_idx):
                pred_rows.append({
                    "split_seed": int(split_seed),
                    "model_seed": int(model_seed),
                    "model": "Morgan-RF",
                    "global_row_index": int(global_idx),
                    "y_true": float(y_val[pos]),
                    "y_pred": float(pred_morgan[pos]),
                })

            block_i += 1
            pct = int(round(100 * block_i / max(total_blocks, 1)))
            print(f"{pct}%", end=" ", flush=True)

            # Descriptor-RF
            rf_desc = make_rf(args, model_seed)
            rf_desc.fit(X_desc_train, y_train)
            pred_desc = rf_desc.predict(X_desc_val)
            md = regression_metrics(y_val, pred_desc)

            run_rows.append({
                "split_seed": int(split_seed),
                "model_seed": int(model_seed),
                "model": "Descriptor-RF",
                "n_train": int(len(train_idx)),
                "n_val": int(len(val_idx)),
                "n_test_frozen_not_evaluated": int(len(test_idx)),
                "test_evaluated": False,
                "n_features": int(X_desc_train.shape[1]),
                **{f"val_{k}": v for k, v in md.items()},
            })
            for pos, global_idx in enumerate(val_idx):
                pred_rows.append({
                    "split_seed": int(split_seed),
                    "model_seed": int(model_seed),
                    "model": "Descriptor-RF",
                    "global_row_index": int(global_idx),
                    "y_true": float(y_val[pos]),
                    "y_pred": float(pred_desc[pos]),
                })

            block_i += 1
            pct = int(round(100 * block_i / max(total_blocks, 1)))
            print(f"{pct}%", end=" " if block_i < total_blocks else "\n", flush=True)

    target_df = pd.DataFrame(target_rows)
    run_df = pd.DataFrame(run_rows)
    pred_df = pd.DataFrame(pred_rows)
    descriptor_qc_df = pd.DataFrame(descriptor_qc_rows)

    summary_df = hierarchical_summary(run_df)
    variability_df = model_seed_variability(run_df)
    comparison_df = build_comparison_with_reference_g2(summary_df, reference_g2_path)

    target_df.to_csv(out_dir / "00_target_summary.csv", index=False, encoding="utf-8-sig")
    run_df.to_csv(out_dir / "01_rf_all_runs.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(out_dir / "02_rf_hierarchical_summary.csv", index=False, encoding="utf-8-sig")
    pred_df.to_csv(out_dir / "03_rf_validation_predictions.csv", index=False, encoding="utf-8-sig")
    descriptor_qc_df.to_csv(out_dir / "04_descriptor_trainonly_preprocess_qc.csv", index=False, encoding="utf-8-sig")
    variability_df.to_csv(out_dir / "05_rf_model_seed_variability.csv", index=False, encoding="utf-8-sig")
    comparison_df.to_csv(out_dir / "06_optional_reference_g2_comparison.csv", index=False, encoding="utf-8-sig")

    meta = {
        "module": "fathead_rf_validation",
        "input_table": str(input_table),
        "split_root": str(split_root),
        "reference_g2_summary_read_only": str(reference_g2_path) if reference_g2_path is not None else None,
        "n_chemicals": int(len(df)),
        "split_seeds": split_seeds,
        "model_seeds": model_seeds,
        "target_col": args.y_col,
        "smiles_col": args.smiles_col,
        "test_partition_evaluated": False,
        "morgan": {
            "radius": int(args.radius),
            "n_bits": int(args.n_bits),
            "includeChirality": False,
            "type": "Count Morgan",
        },
        "rdkit_descriptor_preprocessing": {
            "raw_calculation": "Descriptors.CalcMolDescriptors",
            "nonfinite_to_nan": True,
            "drop_train_all_missing": True,
            "median_impute": "TRAIN medians only",
            "drop_zero_variance": "TRAIN only",
            "scaling": "none (RF)",
        },
        "rf": {
            "n_estimators": int(args.n_estimators),
            "max_depth": args.max_depth,
            "min_samples_leaf": int(args.min_samples_leaf),
            "max_features": float(args.max_features),
            "bootstrap": True,
            "n_jobs": int(args.n_jobs),
            "hyperparameter_tuning": False,
        },
        "versions": {
            "rdkit": rdBase.rdkitVersion,
        },
        "summary_policy": "model-seed mean within structural split, then summary across structural splits",
    }
    (out_dir / "rf_validation_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    lines = [
        "[RANDOM-FOREST VALIDATION BASELINES]",
        "",
        f"Input chemicals                    : {len(df)}",
        f"Structural split seeds             : {split_seeds}",
        f"RF model seeds                     : {model_seeds}",
        f"Target                             : {args.y_col}",
        "Frozen TEST evaluated             : NO",
        f"Morgan                             : Count Morgan radius={args.radius}, n_bits={args.n_bits}, chirality=False",
        f"RF                                 : {args.n_estimators} trees, max_depth={args.max_depth}, min_leaf={args.min_samples_leaf}, max_features={args.max_features}",
        "Descriptor preprocessing           : TRAIN-only median imputation + TRAIN-only zero-variance removal",
        "",
        "[RF / deterministic baseline hierarchical validation summary]",
        summary_df.to_string(index=False),
        "",
        "[Combined comparison with optional reference G2 summary, if supplied]",
        comparison_df.to_string(index=False),
        "",
        "[Interpretation rule]",
        "- Do NOT call the final G2 performance acceptable yet.",
        "- First compare Morgan-RF and Descriptor-RF against G2 under the same frozen splits.",
        "- If RF baselines are much better than G2, diagnose GNN optimization/implementation before proceeding.",
        "- If all methods are similarly weak, diagnose target/split/data limitation rather than tuning the split.",
        "- TEST remains frozen regardless of validation outcome.",
    ]
    (out_dir / "rf_validation_summary.txt").write_text("\n".join(lines), encoding="utf-8")

    print(f"[DONE] {out_dir}")


if __name__ == "__main__":
    main()
