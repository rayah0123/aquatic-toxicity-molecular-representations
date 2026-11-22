#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P10K_evaluate_WebTEST_RF_frozen_TEST_resume.py

Frozen TEST evaluation for the already-fixed WebTEST RF baselines.

Models
------
1) Morgan-RF
   - Count Morgan radius=4, 2048 dimensions, chirality=False
2) Descriptor-RF
   - full RDKit 2D descriptor pool
   - TRAIN-only preprocessing:
       non-finite -> NaN
       drop TRAIN-all-missing columns
       TRAIN-median imputation
       drop TRAIN-zero-variance columns

Frozen RF configuration
-----------------------
n_estimators=500
max_depth=None
min_samples_leaf=1
max_features=1.0
bootstrap=True
model seeds = [42,123,777,2025,3407]

IMPORTANT
---------
- This script does NOT tune anything.
- Validation metrics are NOT used for model selection here.
- Each RF is refit with the exact frozen TRAIN partition/configuration/seed used
  in validation, then applied once to the frozen TEST partition.
- TEST never enters descriptor preprocessing or model fitting.
- One split_seed x model_seed x model task is checkpointed atomically.
- Re-running the same command skips completed tasks and resumes remaining work.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from rdkit import Chem, rdBase, RDLogger
from rdkit.Chem import Descriptors, rdFingerprintGenerator

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parents[1]

INPUT_TABLE = (
    ROOT / "03_processed" / "WebTEST"
    / "WebTEST_Daphnia_consensus_unique_connectivity_frozen.csv"
)
SPLIT_ROOT = (
    ROOT / "04_splits" / "WebTEST_Daphnia"
    / "scaffold_split_murcko_topology_refined"
)
OUT = (
    ROOT / "07_results" / "WebTEST_revised"
    / "P10K_RF_frozen_TEST"
)

SMILES_COL = "SMILES_RDKit"
Y_COL = "target_neglog10_molar"

SPLIT_SEEDS = [42, 123, 777, 2025, 3407]
MODEL_SEEDS = [42, 123, 777, 2025, 3407]

RADIUS = 4
N_BITS = 2048

N_ESTIMATORS = 500
MAX_DEPTH = None
MIN_SAMPLES_LEAF = 1
MAX_FEATURES = 1.0
N_JOBS = -1

EXPECTED_N = 119276


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)


def atomic_write_json(path: Path, obj: dict) -> None:
    atomic_write_text(path, json.dumps(obj, indent=2, ensure_ascii=False))


def atomic_write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    try:
        os.close(fd)
        df.to_csv(tmp_name, index=False, encoding="utf-8-sig")
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)


def atomic_save_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=path.stem + ".", suffix=".npz", dir=str(path.parent)
    )
    os.close(fd)
    try:
        np.savez_compressed(tmp_name, **arrays)
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)


def load_split_indices(split_root: Path, seed: int, n_rows: int) -> Dict[str, np.ndarray]:
    path = split_root / f"split_seed{seed}" / "split_membership.csv"
    if not path.exists():
        raise FileNotFoundError(path)

    m = pd.read_csv(path, encoding="utf-8-sig")
    if "split" not in m.columns:
        raise KeyError(f"Missing split column: {path}")

    idx_col = None
    for candidate in ["original_row_index", "row_idx", "index"]:
        if candidate in m.columns:
            idx_col = candidate
            break
    if idx_col is None:
        raise KeyError(f"Could not identify row-index column: {path}")

    out = {}
    for name in ["train", "val", "test"]:
        arr = pd.to_numeric(
            m.loc[m["split"].astype(str).str.lower().eq(name), idx_col],
            errors="raise",
        ).astype(int).to_numpy()
        out[name] = np.sort(arr)

    all_idx = np.concatenate([out["train"], out["val"], out["test"]])
    if len(all_idx) != n_rows:
        raise RuntimeError(f"seed={seed}: assigned={len(all_idx)} expected={n_rows}")
    if len(np.unique(all_idx)) != n_rows:
        raise RuntimeError(f"seed={seed}: duplicate row assignment")
    if all_idx.min() < 0 or all_idx.max() >= n_rows:
        raise RuntimeError(f"seed={seed}: row index out of range")
    return out


def build_count_morgan(smiles: Sequence[str]) -> np.ndarray:
    gen = rdFingerprintGenerator.GetMorganGenerator(
        radius=RADIUS,
        fpSize=N_BITS,
        includeChirality=False,
    )
    X = np.zeros((len(smiles), N_BITS), dtype=np.float32)
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            raise ValueError(f"Invalid SMILES at row {i}: {smi}")
        fp = gen.GetCountFingerprint(mol)
        for bit_idx, count in fp.GetNonzeroElements().items():
            bit_idx = int(bit_idx)
            if 0 <= bit_idx < N_BITS:
                X[i, bit_idx] = float(count)
    return X


def build_rdkit_descriptor_raw(
    smiles: Sequence[str],
) -> Tuple[np.ndarray, List[str]]:
    rows: List[Dict[str, float]] = []
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            raise ValueError(f"Invalid SMILES at row {i}: {smi}")
        d = Descriptors.CalcMolDescriptors(mol, missingVal=np.nan, silent=True)
        clean = {}
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
    return ddf.to_numpy(dtype=np.float64), ddf.columns.astype(str).tolist()


def fit_descriptor_preprocessor_train_only(
    X_raw: np.ndarray,
    train_idx: np.ndarray,
    descriptor_names: Sequence[str],
):
    Xtr = np.asarray(X_raw[train_idx], dtype=np.float64)

    finite_any = np.isfinite(Xtr).any(axis=0)
    cols_1 = np.where(finite_any)[0]
    if len(cols_1) == 0:
        raise RuntimeError("All descriptor columns are missing on TRAIN.")

    X1 = Xtr[:, cols_1]
    medians = np.nanmedian(X1, axis=0)

    good_median = np.isfinite(medians)
    cols_2 = cols_1[good_median]
    medians_2 = medians[good_median]

    X2 = np.asarray(X_raw[train_idx][:, cols_2], dtype=np.float64)
    bad = ~np.isfinite(X2)
    if bad.any():
        X2 = X2.copy()
        rr, cc = np.where(bad)
        X2[rr, cc] = medians_2[cc]

    variances = np.var(X2, axis=0)
    nonzero_var = np.isfinite(variances) & (variances > 0)

    kept_col_idx = cols_2[nonzero_var]
    kept_medians = medians_2[nonzero_var]
    kept_names = [str(descriptor_names[i]) for i in kept_col_idx]

    return kept_col_idx, kept_medians, kept_names


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
        raise RuntimeError("Non-finite descriptor values remain.")
    return X.astype(np.float32, copy=False)


def make_rf(seed: int) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        max_features=MAX_FEATURES,
        bootstrap=True,
        random_state=int(seed),
        n_jobs=N_JOBS,
    )


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
        "Pearson": float(pd.Series(y_true).corr(pd.Series(y_pred), method="pearson")),
        "Spearman": float(pd.Series(y_true).corr(pd.Series(y_pred), method="spearman")),
        "bias_pred_minus_true": float(np.mean(y_pred - y_true)),
    }


def task_stem(split_seed: int, model_seed: int, model: str) -> str:
    safe_model = model.replace("-", "_")
    return f"split{split_seed}_modelseed{model_seed}_{safe_model}"


def task_done(checkpoint_dir: Path, stem: str) -> bool:
    meta = checkpoint_dir / f"{stem}.json"
    pred = checkpoint_dir / f"{stem}.npz"
    if not meta.exists() or not pred.exists():
        return False
    try:
        obj = json.loads(meta.read_text(encoding="utf-8"))
    except Exception:
        return False
    return obj.get("status") == "DONE"


def load_all_checkpoints(checkpoint_dir: Path):
    run_rows = []
    pred_frames = []

    for meta_path in sorted(checkpoint_dir.glob("split*_*.json")):
        obj = json.loads(meta_path.read_text(encoding="utf-8"))
        if obj.get("status") != "DONE":
            continue

        stem = meta_path.stem
        npz_path = checkpoint_dir / f"{stem}.npz"
        if not npz_path.exists():
            raise RuntimeError(f"Missing prediction checkpoint for {stem}")

        run_rows.append(obj["result"])

        z = np.load(npz_path)
        n = len(z["global_row_index"])
        pred_frames.append(pd.DataFrame({
            "split_seed": np.repeat(int(obj["result"]["split_seed"]), n),
            "model_seed": np.repeat(int(obj["result"]["model_seed"]), n),
            "model": np.repeat(str(obj["result"]["model"]), n),
            "global_row_index": z["global_row_index"].astype(np.int64),
            "y_true": z["y_true"].astype(np.float64),
            "y_pred": z["y_pred"].astype(np.float64),
        }))

    run_df = pd.DataFrame(run_rows)
    pred_df = (
        pd.concat(pred_frames, ignore_index=True)
        if pred_frames else pd.DataFrame()
    )
    return run_df, pred_df


def hierarchical_summary(run_df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "test_RMSE", "test_MAE", "test_R2",
        "test_Pearson", "test_Spearman",
        "test_bias_pred_minus_true",
    ]

    split_means = (
        run_df.groupby(["split_seed", "model"], as_index=False)[metric_cols]
        .mean()
    )

    rows = []
    for model, g in split_means.groupby("model"):
        row = {
            "model": model,
            "n_structural_splits": int(g["split_seed"].nunique()),
            "model_seeds_per_split": int(
                run_df.loc[run_df["model"].eq(model), "model_seed"].nunique()
            ),
        }
        for m in metric_cols:
            row[f"{m}_splitmean_mean"] = float(g[m].mean())
            row[f"{m}_splitmean_std"] = (
                float(g[m].std(ddof=1)) if len(g) > 1 else np.nan
            )
            row[f"{m}_splitmean_median"] = float(g[m].median())
        rows.append(row)

    return (
        pd.DataFrame(rows)
        .sort_values("test_RMSE_splitmean_mean")
        .reset_index(drop=True)
    )


def main():
    RDLogger.DisableLog("rdApp.*")

    for p in [INPUT_TABLE, SPLIT_ROOT]:
        if not p.exists():
            raise FileNotFoundError(p)

    OUT.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = OUT / "_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(
        INPUT_TABLE,
        encoding="utf-8-sig",
        low_memory=False,
    ).reset_index(drop=True)

    if len(df) != EXPECTED_N:
        raise RuntimeError(f"Expected N={EXPECTED_N}, got {len(df)}")

    for col in [SMILES_COL, Y_COL]:
        if col not in df.columns:
            raise KeyError(col)

    smiles = df[SMILES_COL].astype(str).tolist()
    y_all = pd.to_numeric(df[Y_COL], errors="raise").to_numpy(dtype=np.float64)
    if not np.isfinite(y_all).all():
        raise ValueError("Non-finite targets found.")

    meta = {
        "module": "P10K_evaluate_WebTEST_RF_frozen_TEST_resume",
        "status": "FROZEN_TEST_EVALUATION",
        "n_chemicals": int(len(df)),
        "input_table": str(INPUT_TABLE),
        "split_root": str(SPLIT_ROOT),
        "target_col": Y_COL,
        "smiles_col": SMILES_COL,
        "split_seeds": SPLIT_SEEDS,
        "model_seeds": MODEL_SEEDS,
        "models": ["Morgan-RF", "Descriptor-RF"],
        "morgan": {
            "type": "Count Morgan",
            "radius": RADIUS,
            "n_bits": N_BITS,
            "includeChirality": False,
        },
        "descriptor_preprocessing": (
            "TRAIN-only all-missing removal, TRAIN median imputation, "
            "TRAIN-only zero-variance removal"
        ),
        "rf": {
            "n_estimators": N_ESTIMATORS,
            "max_depth": MAX_DEPTH,
            "min_samples_leaf": MIN_SAMPLES_LEAF,
            "max_features": MAX_FEATURES,
            "bootstrap": True,
            "n_jobs": N_JOBS,
            "hyperparameter_tuning_after_validation": False,
        },
        "test_policy": (
            "Refit exact frozen TRAIN/config/seed; predict TEST once. "
            "TEST is never used for preprocessing, fitting, or model selection."
        ),
        "checkpoint_resume": True,
        "versions": {"rdkit": rdBase.rdkitVersion},
    }
    atomic_write_json(OUT / "00_meta.json", meta)

    print("[1/4] Building Count Morgan r4/2048 ...", flush=True)
    X_morgan = build_count_morgan(smiles)

    print("[2/4] Calculating raw RDKit descriptors ...", flush=True)
    X_desc_raw, descriptor_names = build_rdkit_descriptor_raw(smiles)

    total_tasks = len(SPLIT_SEEDS) * len(MODEL_SEEDS) * 2
    completed = 0
    for split_seed in SPLIT_SEEDS:
        for model_seed in MODEL_SEEDS:
            for model in ["Morgan-RF", "Descriptor-RF"]:
                stem = task_stem(split_seed, model_seed, model)
                if task_done(checkpoint_dir, stem):
                    completed += 1

    print(
        f"[3/4] Frozen TEST RF: {completed}/{total_tasks} tasks already complete.",
        flush=True,
    )

    task_i = 0

    for split_seed in SPLIT_SEEDS:
        idx = load_split_indices(SPLIT_ROOT, split_seed, len(df))
        train_idx = idx["train"]
        test_idx = idx["test"]

        y_train = y_all[train_idx]
        y_test = y_all[test_idx]

        # Descriptor preprocessing is fit strictly on TRAIN.
        kept_idx, train_medians, kept_names = (
            fit_descriptor_preprocessor_train_only(
                X_desc_raw, train_idx, descriptor_names
            )
        )
        X_desc_train = transform_descriptors(
            X_desc_raw, train_idx, kept_idx, train_medians
        )
        X_desc_test = transform_descriptors(
            X_desc_raw, test_idx, kept_idx, train_medians
        )

        X_morgan_train = X_morgan[train_idx]
        X_morgan_test = X_morgan[test_idx]

        for model_seed in MODEL_SEEDS:
            for model in ["Morgan-RF", "Descriptor-RF"]:
                task_i += 1
                stem = task_stem(split_seed, model_seed, model)

                if task_done(checkpoint_dir, stem):
                    print(
                        f"[SKIP {task_i}/{total_tasks}] "
                        f"split={split_seed} seed={model_seed} model={model}",
                        flush=True,
                    )
                    continue

                print(
                    f"[RUN  {task_i}/{total_tasks}] "
                    f"split={split_seed} seed={model_seed} model={model}",
                    flush=True,
                )

                if model == "Morgan-RF":
                    Xtr, Xte = X_morgan_train, X_morgan_test
                else:
                    Xtr, Xte = X_desc_train, X_desc_test

                rf = make_rf(model_seed)
                rf.fit(Xtr, y_train)
                pred = rf.predict(Xte)

                mm = metrics(y_test, pred)

                result = {
                    "split_seed": int(split_seed),
                    "model_seed": int(model_seed),
                    "model": model,
                    "n_total": int(len(df)),
                    "n_train": int(len(train_idx)),
                    "n_test": int(len(test_idx)),
                    "n_features": int(Xtr.shape[1]),
                    **{f"test_{k}": v for k, v in mm.items()},
                }

                atomic_save_npz(
                    checkpoint_dir / f"{stem}.npz",
                    global_row_index=test_idx.astype(np.int64),
                    y_true=y_test.astype(np.float64),
                    y_pred=np.asarray(pred, dtype=np.float64),
                )
                atomic_write_json(
                    checkpoint_dir / f"{stem}.json",
                    {
                        "status": "DONE",
                        "result": result,
                        "descriptor_names_if_applicable": (
                            kept_names if model == "Descriptor-RF" else None
                        ),
                    },
                )

                run_df, _ = load_all_checkpoints(checkpoint_dir)
                if len(run_df):
                    run_df = run_df.sort_values(
                        ["split_seed", "model", "model_seed"]
                    ).reset_index(drop=True)
                    atomic_write_csv(
                        OUT / "01_completed_run_metrics.csv",
                        run_df,
                    )

    run_df, pred_df = load_all_checkpoints(checkpoint_dir)

    if len(run_df) != total_tasks:
        raise RuntimeError(
            f"Only {len(run_df)}/{total_tasks} tasks completed. "
            "Re-run the same command to resume."
        )

    run_df = run_df.sort_values(
        ["split_seed", "model", "model_seed"]
    ).reset_index(drop=True)
    pred_df = pred_df.sort_values(
        ["split_seed", "model", "model_seed", "global_row_index"]
    ).reset_index(drop=True)

    summary_df = hierarchical_summary(run_df)

    atomic_write_csv(OUT / "01_completed_run_metrics.csv", run_df)
    atomic_write_csv(OUT / "02_RF_frozen_TEST_predictions.csv", pred_df)
    atomic_write_csv(OUT / "03_RF_frozen_TEST_summary.csv", summary_df)

    lines = [
        "[P10K WEBTEST RF FROZEN TEST]",
        "",
        f"Input chemicals        : {len(df)}",
        f"Structural split seeds : {SPLIT_SEEDS}",
        f"RF model seeds         : {MODEL_SEEDS}",
        "Hyperparameter retuning: NO",
        "TEST used in fitting   : NO",
        "Checkpoint/resume      : YES",
        "",
        "[Frozen TEST hierarchical summary]",
        summary_df.to_string(index=False),
        "",
        "[Interpretation boundary]",
        "- These are the frozen TEST results for the previously fixed RF baselines.",
        "- Do not retune RF settings after observing these TEST results.",
        "- Chemical-level predictions are saved for the later model-choice condition map.",
    ]
    atomic_write_text(OUT / "P10K_summary.txt", "\n".join(lines))

    print("[4/4] COMPLETE", flush=True)
    print(summary_df.to_string(index=False), flush=True)
    print(f"[DONE] {OUT}", flush=True)


if __name__ == "__main__":
    main()
