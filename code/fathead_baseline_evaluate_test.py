#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Frozen TEST baselines for Fathead minnow. No training or tuning."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


SEEDS = [42, 123, 777, 2025, 3407]


def parse_args():
    p = argparse.ArgumentParser()
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
        default="../07_results/Fathead/baseline_test",
    )
    p.add_argument("--y_col", default="Tox_original")
    return p.parse_args()


def resolve(s):
    p = Path(s)
    return p if p.is_absolute() else (Path(__file__).resolve().parent / p).resolve()


def metrics(y, pred):
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y, pred))),
        "MAE": float(mean_absolute_error(y, pred)),
        "R2": float(r2_score(y, pred)),
        "Pearson": float(pd.Series(y).corr(pd.Series(pred), method="pearson")),
        "Spearman": float(pd.Series(y).corr(pd.Series(pred), method="spearman")),
    }


def load_idx(path, split):
    m = pd.read_csv(path, encoding="utf-8-sig")
    ic = next(
        (c for c in ["original_row_index", "row_idx", "index", "global_row_index"] if c in m.columns),
        None,
    )
    if ic is None:
        raise KeyError(f"No row-index column: {path}")
    return np.sort(
        pd.to_numeric(
            m.loc[m["split"].astype(str).str.lower().eq(split), ic],
            errors="raise",
        ).astype(int).to_numpy()
    )


def main():
    a = parse_args()
    inp, split_root, out = resolve(a.input_table), resolve(a.split_root), resolve(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    d = pd.read_csv(inp, encoding="utf-8-sig", low_memory=False).reset_index(drop=True)
    y = pd.to_numeric(d[a.y_col], errors="raise").to_numpy(float)
    if len(d) != 821 or not np.isfinite(y).all():
        raise RuntimeError("Unexpected Fish input or target.")

    rows = []
    for seed in SEEDS:
        sp = split_root / f"split_seed{seed}" / "split_membership.csv"
        tr, va, te = load_idx(sp, "train"), load_idx(sp, "val"), load_idx(sp, "test")
        for model, center in [
            ("TrainMean", float(np.mean(y[tr]))),
            ("TrainMedian", float(np.median(y[tr]))),
        ]:
            pred = np.full(len(te), center, dtype=float)
            mm = metrics(y[te], pred)
            rows.append({
                "model": model,
                "split_seed": seed,
                "n_total": len(d),
                "n_train": len(tr),
                "n_val": len(va),
                "n_test": len(te),
                **{f"test_{k}": v for k, v in mm.items()},
            })

    rdf = pd.DataFrame(rows)
    rdf.to_csv(out / "01_baseline_frozen_test_by_split.csv", index=False, encoding="utf-8-sig")

    metric_cols = ["test_RMSE", "test_MAE", "test_R2", "test_Pearson", "test_Spearman"]
    summary = []
    for model, g in rdf.groupby("model"):
        row = {
            "model": model,
            "n_total": int(g["n_total"].iloc[0]),
            "n_structural_splits": int(len(g)),
            "train_n_min": int(g["n_train"].min()),
            "train_n_max": int(g["n_train"].max()),
            "val_n_min": int(g["n_val"].min()),
            "val_n_max": int(g["n_val"].max()),
            "test_n_min": int(g["n_test"].min()),
            "test_n_max": int(g["n_test"].max()),
        }
        for c in metric_cols:
            row[f"{c}_splitmean_mean"] = float(g[c].mean())
            row[f"{c}_splitmean_std"] = float(g[c].std(ddof=1))
            row[f"{c}_splitmean_median"] = float(g[c].median())
        summary.append(row)

    pd.DataFrame(summary).to_csv(
        out / "02_baseline_frozen_test_hierarchical_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print(f"[DONE] {out}")
    print("[NOTE] Frozen TEST baselines evaluated; no model fitting occurred.")


if __name__ == "__main__":
    main()
