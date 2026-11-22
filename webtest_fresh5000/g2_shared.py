#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Shared helpers for the six-descriptor GraphSAGE comparison.

This module contains only the preprocessing, metric, and hierarchical-summary
logic required by the large WebTEST G2/descriptor-only training pipeline.
All data-dependent preprocessing is fit on the training partition only.
"""
from __future__ import annotations
from typing import Dict, Sequence
import numpy as np
import pandas as pd
import g2_core as base


def fit_descriptor_preprocessor_train_only(
    X_raw: np.ndarray, train_idx: np.ndarray, descriptor_names: Sequence[str]
) -> Dict:
    Xtr = np.asarray(X_raw[train_idx], dtype=np.float64)
    finite_any = np.isfinite(Xtr).any(axis=0)
    cols1 = np.where(finite_any)[0]
    if len(cols1) == 0:
        raise RuntimeError("All descriptors are training-all-missing.")

    X1 = Xtr[:, cols1]
    med1 = np.nanmedian(X1, axis=0)
    good_med = np.isfinite(med1)
    cols2 = cols1[good_med]
    med2 = med1[good_med]

    X2 = np.asarray(X_raw[train_idx][:, cols2], dtype=np.float64)
    bad = ~np.isfinite(X2)
    if bad.any():
        X2 = X2.copy()
        rr, cc = np.where(bad)
        X2[rr, cc] = med2[cc]

    var2 = np.var(X2, axis=0)
    nonzero = np.isfinite(var2) & (var2 > 0)
    kept_cols = cols2[nonzero]
    kept_medians = med2[nonzero]

    Xk = np.asarray(X_raw[train_idx][:, kept_cols], dtype=np.float64)
    bad = ~np.isfinite(Xk)
    if bad.any():
        Xk = Xk.copy()
        rr, cc = np.where(bad)
        Xk[rr, cc] = kept_medians[cc]

    mean = np.mean(Xk, axis=0)
    std = np.std(Xk, axis=0, ddof=0)
    good_std = np.isfinite(std) & (std > 0)
    final_cols = kept_cols[good_std]
    final_medians = kept_medians[good_std]
    final_mean = mean[good_std]
    final_std = std[good_std]
    final_names = [str(descriptor_names[i]) for i in final_cols]

    return {
        "kept_cols": final_cols.astype(int),
        "medians": final_medians.astype(float),
        "mean": final_mean.astype(float),
        "std": final_std.astype(float),
        "names": final_names,
        "qc": {
            "n_descriptor_raw": int(X_raw.shape[1]),
            "n_drop_train_all_missing": int(np.sum(~finite_any)),
            "n_drop_nonfinite_train_median": int(np.sum(~good_med)),
            "n_drop_train_zero_variance_pre_z": int(np.sum(~nonzero)),
            "n_drop_train_zero_variance_post_impute": int(np.sum(~good_std)),
            "n_descriptor_final": int(len(final_cols)),
        },
    }


def transform_descriptor_rows(X_raw: np.ndarray, row_idx: np.ndarray, pp: Dict) -> np.ndarray:
    cols = pp["kept_cols"]
    med = pp["medians"]
    mean = pp["mean"]
    std = pp["std"]
    X = np.asarray(X_raw[row_idx][:, cols], dtype=np.float64)
    bad = ~np.isfinite(X)
    if bad.any():
        X = X.copy()
        rr, cc = np.where(bad)
        X[rr, cc] = med[cc]
    X = (X - mean) / std
    if not np.isfinite(X).all():
        raise RuntimeError("Non-finite values remain after descriptor standardization.")
    return X.astype(np.float32, copy=False)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return base.regression_metrics(y_true, y_pred)


def hierarchical_summary(run_df: pd.DataFrame) -> pd.DataFrame:
    metrics = ["val_RMSE", "val_MAE", "val_R2", "val_Pearson", "val_Spearman"]
    split_means = (
        run_df.groupby(
            ["split_seed", "condition", "relation_mode", "topk"],
            dropna=False, as_index=False,
        )[metrics].mean()
    )
    rows = []
    for keys, g in split_means.groupby(
        ["condition", "relation_mode", "topk"], dropna=False
    ):
        condition, mode, topk = keys
        row = {
            "condition": condition,
            "relation_mode": mode,
            "topk": topk,
            "n_structural_splits": int(g["split_seed"].nunique()),
            "model_seeds_per_split": int(run_df["model_seed"].nunique()),
        }
        for metric in metrics:
            row[f"{metric}_splitmean_mean"] = float(g[metric].mean())
            row[f"{metric}_splitmean_std"] = float(g[metric].std(ddof=1)) if len(g) > 1 else np.nan
            row[f"{metric}_splitmean_median"] = float(g[metric].median())
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["val_RMSE_splitmean_mean", "condition"],
        ascending=[True, True], na_position="last",
    ).reset_index(drop=True)
