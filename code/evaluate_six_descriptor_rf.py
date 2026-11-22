#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate a matched six-descriptor random-forest baseline across Daphnia,
ADORE, fathead minnow, and WebTEST.

Descriptors: MolLogP, MolWt, TPSA, NumHAcceptors, NumHDonors,
NumRotatableBonds. All data-dependent preprocessing uses TRAIN only.
Use the fixed RF configuration (500 trees, max_features=1.0,
min_samples_leaf=1, bootstrap=True, unrestricted depth).
Each context/split/seed task saves its predictions separately; existing
model outputs are not modified. TEST/external outcomes are not used for fitting."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from rdkit import Chem, RDLogger, rdBase
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parents[1]

SPLIT_SEEDS = [42, 123, 777, 2025, 3407]
MODEL_SEEDS = [42, 123, 777, 2025, 3407]

N_ESTIMATORS = 500
MAX_DEPTH = None
MIN_SAMPLES_LEAF = 1
MAX_FEATURES = 1.0
N_JOBS = -1

COMPACT6 = [
    "MolLogP",
    "MolWt",
    "TPSA",
    "NumHAcceptors",
    "NumHDonors",
    "NumRotatableBonds",
]

DAPHNIA = (
    ROOT / "03_processed" / "TEST"
    / "TEST_Daphnia_modeling_pool_no_ADORE_overlap.csv"
)
DAPHNIA_SPLITS = (
    ROOT / "04_splits" / "TEST_Daphnia"
    / "scaffold_split_murcko_topology_refined"
)

FATHEAD = (
    ROOT / "03_processed" / "TEST"
    / "TEST_Fathead_96h_LC50_modeling_pool.csv"
)
FATHEAD_SPLITS = (
    ROOT / "04_splits" / "TEST_Fathead"
    / "scaffold_split_murcko_topology_refined"
)

ADORE = (
    ROOT / "03_processed" / "ADORE"
    / "ADORE_Daphnia_48h_LC50_external_chemical_level.csv"
)

WEBTEST = (
    ROOT / "03_processed" / "WebTEST"
    / "WebTEST_Daphnia_consensus_unique_connectivity_frozen.csv"
)
WEBTEST_SPLITS = (
    ROOT / "04_splits" / "WebTEST_Daphnia"
    / "scaffold_split_murcko_topology_refined"
)

OUT_ROOT = (
    ROOT / "07_results" / "matched_controls"
    / "six_descriptor_rf"
)

CONTEXTS = {
    "Daphnia_TEST": {
        "train_table": DAPHNIA,
        "query_table": DAPHNIA,
        "split_root": DAPHNIA_SPLITS,
        "train_smiles_col": "canonical_SMILES_RDKit",
        "query_smiles_col": "canonical_SMILES_RDKit",
        "train_y_col": "Tox_original",
        "query_y_col": "Tox_original",
        "query_partition": "test",
        "metric_prefix": "test",
        "expected_train_n": 316,
        "expected_query_n": 316,
    },
    "ADORE_external": {
        "train_table": DAPHNIA,
        "query_table": ADORE,
        "split_root": DAPHNIA_SPLITS,
        "train_smiles_col": "canonical_SMILES_RDKit",
        "query_smiles_col": "smiles_connectivity",
        "train_y_col": "Tox_original",
        "query_y_col": "target_y",
        "query_partition": "external",
        "metric_prefix": "external",
        "expected_train_n": 316,
        "expected_query_n": 274,
    },
    "Fathead_TEST": {
        "train_table": FATHEAD,
        "query_table": FATHEAD,
        "split_root": FATHEAD_SPLITS,
        "train_smiles_col": "canonical_SMILES_RDKit",
        "query_smiles_col": "canonical_SMILES_RDKit",
        "train_y_col": "Tox_original",
        "query_y_col": "Tox_original",
        "query_partition": "test",
        "metric_prefix": "test",
        "expected_train_n": 821,
        "expected_query_n": 821,
    },
    "WebTEST_surrogate": {
        "train_table": WEBTEST,
        "query_table": WEBTEST,
        "split_root": WEBTEST_SPLITS,
        "train_smiles_col": "SMILES_RDKit",
        "query_smiles_col": "SMILES_RDKit",
        "train_y_col": "target_neglog10_molar",
        "query_y_col": "target_neglog10_molar",
        "query_partition": "test",
        "metric_prefix": "test",
        "expected_train_n": 119276,
        "expected_query_n": 119276,
    },
}


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
    atomic_write_text(
        path,
        json.dumps(obj, indent=2, ensure_ascii=False),
    )


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
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "wb") as f:
            np.savez_compressed(f, **arrays)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)


def load_split_indices(
    split_root: Path,
    seed: int,
    n_rows: int,
) -> Dict[str, np.ndarray]:
    path = split_root / f"split_seed{seed}" / "split_membership.csv"
    if not path.exists():
        raise FileNotFoundError(path)

    m = pd.read_csv(path, encoding="utf-8-sig")
    if "split" not in m.columns:
        raise KeyError(f"Missing split column: {path}")

    idx_col = None
    for c in ["original_row_index", "row_idx", "index"]:
        if c in m.columns:
            idx_col = c
            break
    if idx_col is None:
        raise KeyError(
            f"No row-index column in {path}; "
            "expected original_row_index / row_idx / index"
        )

    out = {}
    for name in ["train", "val", "test"]:
        arr = (
            pd.to_numeric(
                m.loc[
                    m["split"].astype(str).str.lower().eq(name),
                    idx_col,
                ],
                errors="raise",
            )
            .astype(int)
            .to_numpy()
        )
        out[name] = np.sort(arr)

    all_idx = np.concatenate(
        [out["train"], out["val"], out["test"]]
    )
    if len(all_idx) != n_rows:
        raise RuntimeError(
            f"split={seed}: assigned={len(all_idx)}, expected={n_rows}"
        )
    if len(np.unique(all_idx)) != n_rows:
        raise RuntimeError(f"split={seed}: duplicate row assignment")
    if all_idx.min() < 0 or all_idx.max() >= n_rows:
        raise RuntimeError(f"split={seed}: row index out of range")

    return out


def build_compact6(smiles: List[str]) -> np.ndarray:
    X = np.zeros((len(smiles), 6), dtype=np.float32)

    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            raise ValueError(f"Invalid SMILES row={i}: {smi}")

        vals = np.asarray(
            [
                Crippen.MolLogP(mol),
                Descriptors.MolWt(mol),
                rdMolDescriptors.CalcTPSA(mol),
                Lipinski.NumHAcceptors(mol),
                Lipinski.NumHDonors(mol),
                Lipinski.NumRotatableBonds(mol),
            ],
            dtype=np.float64,
        )
        if not np.isfinite(vals).all():
            raise ValueError(
                f"Non-finite six-descriptor set at row={i}: {smi}"
            )
        X[i] = vals.astype(np.float32)

    return X


def metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    rmse = float(
        np.sqrt(mean_squared_error(y_true, y_pred))
    )
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    pearson = float(
        pd.Series(y_true).corr(
            pd.Series(y_pred), method="pearson"
        )
    )
    spearman = float(
        pd.Series(y_true).corr(
            pd.Series(y_pred), method="spearman"
        )
    )

    bias = float(np.mean(y_pred - y_true))

    true_std = (
        float(np.std(y_true, ddof=1))
        if len(y_true) > 1 else np.nan
    )
    pred_std = (
        float(np.std(y_pred, ddof=1))
        if len(y_pred) > 1 else np.nan
    )
    sd_ratio = (
        float(pred_std / true_std)
        if np.isfinite(true_std)
        and true_std > 0
        and np.isfinite(pred_std)
        else np.nan
    )

    if (
        len(y_true) >= 2
        and np.isfinite(pred_std)
        and pred_std > 0
    ):
        slope, intercept = np.polyfit(
            y_pred, y_true, deg=1
        )
        slope = float(slope)
        intercept = float(intercept)
    else:
        slope = np.nan
        intercept = np.nan

    return {
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "Pearson": pearson,
        "Spearman": spearman,
        "bias_pred_minus_true": bias,
        "pred_to_true_sd_ratio": sd_ratio,
        "calibration_slope_ytrue_on_ypred": slope,
        "calibration_intercept_ytrue_on_ypred": intercept,
    }


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


def task_stem(
    split_seed: int,
    model_seed: int,
) -> str:
    return f"split{split_seed}_model{model_seed}"


def task_done(
    checkpoint_dir: Path,
    stem: str,
) -> bool:
    p = checkpoint_dir / f"{stem}.json"
    if not p.exists():
        return False
    try:
        obj = json.loads(
            p.read_text(encoding="utf-8")
        )
        return obj.get("status") == "DONE"
    except Exception:
        return False


def load_checkpoints(
    checkpoint_dir: Path,
    metric_prefix: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    run_rows = []
    pred_rows = []

    for jp in sorted(
        checkpoint_dir.glob("split*_model*.json")
    ):
        obj = json.loads(
            jp.read_text(encoding="utf-8")
        )
        if obj.get("status") != "DONE":
            continue

        run_rows.append(obj["result"])

        npz = jp.with_suffix(".npz")
        if not npz.exists():
            raise FileNotFoundError(
                f"Prediction checkpoint missing: {npz}"
            )

        a = np.load(npz)
        split_seed = int(obj["split_seed"])
        model_seed = int(obj["model_seed"])

        for i in range(len(a["y_true"])):
            pred_rows.append(
                {
                    "split_seed": split_seed,
                    "model_seed": model_seed,
                    "model": "six-descriptor RF",
                    "query_row_index": int(
                        a["query_row_index"][i]
                    ),
                    "y_true": float(a["y_true"][i]),
                    "y_pred": float(a["y_pred"][i]),
                    "stage": metric_prefix,
                }
            )

    return pd.DataFrame(run_rows), pd.DataFrame(pred_rows)


def hierarchical_summary(
    run_df: pd.DataFrame,
    metric_prefix: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    metric_names = [
        "RMSE",
        "MAE",
        "R2",
        "Pearson",
        "Spearman",
        "bias_pred_minus_true",
        "pred_to_true_sd_ratio",
        "calibration_slope_ytrue_on_ypred",
    ]
    cols = [
        f"{metric_prefix}_{m}"
        for m in metric_names
    ]

    splitmeans = (
        run_df.groupby(
            ["model", "split_seed"],
            as_index=False,
        )[cols]
        .mean(numeric_only=True)
        .sort_values(["model", "split_seed"])
        .reset_index(drop=True)
    )

    rows = []
    for model, g in splitmeans.groupby("model"):
        row = {
            "model": model,
            "n_structural_splits": int(
                g["split_seed"].nunique()
            ),
            "model_seeds_per_split": int(
                run_df.loc[
                    run_df["model"].eq(model),
                    "model_seed",
                ].nunique()
            ),
        }

        for c in cols:
            vals = g[c].to_numpy(dtype=float)
            row[f"{c}_splitmean_mean"] = float(
                np.mean(vals)
            )
            row[f"{c}_splitmean_std"] = float(
                np.std(vals, ddof=1)
            )
            row[f"{c}_splitmean_median"] = float(
                np.median(vals)
            )

        rows.append(row)

    return pd.DataFrame(rows), splitmeans


def audit_external_overlap(
    train_df: pd.DataFrame,
    query_df: pd.DataFrame,
) -> None:
    candidate_cols = [
        "smiles_connectivity",
        "SMILES_connectivity",
    ]
    tc = next(
        (c for c in candidate_cols if c in train_df.columns),
        None,
    )
    qc = next(
        (c for c in candidate_cols if c in query_df.columns),
        None,
    )
    if tc is None or qc is None:
        print(
            "[ADORE overlap QC] connectivity columns not both present; "
            "curated input provenance is relied upon.",
            flush=True,
        )
        return

    a = set(
        train_df[tc]
        .dropna()
        .astype(str)
        .str.strip()
    )
    b = set(
        query_df[qc]
        .dropna()
        .astype(str)
        .str.strip()
    )
    overlap = a & b
    if overlap:
        raise RuntimeError(
            f"ADORE connectivity overlap unexpectedly present: {len(overlap)}"
        )
    print(
        "[ADORE overlap QC PASS] train/external connectivity overlap = 0",
        flush=True,
    )


def run_context(
    context: str,
    spec: dict,
) -> None:
    out_dir = OUT_ROOT / context
    checkpoint_dir = out_dir / "_checkpoints"
    checkpoint_dir.mkdir(
        parents=True, exist_ok=True
    )

    train_df = pd.read_csv(
        spec["train_table"],
        encoding="utf-8-sig",
        low_memory=False,
    ).reset_index(drop=True)

    if (
        spec["query_table"].resolve()
        == spec["train_table"].resolve()
    ):
        query_df = train_df
    else:
        query_df = pd.read_csv(
            spec["query_table"],
            encoding="utf-8-sig",
            low_memory=False,
        ).reset_index(drop=True)

    if len(train_df) != spec["expected_train_n"]:
        raise RuntimeError(
            f"{context}: train N={len(train_df)}, "
            f"expected={spec['expected_train_n']}"
        )
    if len(query_df) != spec["expected_query_n"]:
        raise RuntimeError(
            f"{context}: query N={len(query_df)}, "
            f"expected={spec['expected_query_n']}"
        )

    for df, c in [
        (train_df, spec["train_smiles_col"]),
        (train_df, spec["train_y_col"]),
        (query_df, spec["query_smiles_col"]),
        (query_df, spec["query_y_col"]),
    ]:
        if c not in df.columns:
            raise KeyError(
                f"{context}: missing column {c}"
            )

    if context == "ADORE_external":
        audit_external_overlap(
            train_df, query_df
        )

    print(
        f"\n[{context}] Build six-descriptor set TRAIN",
        flush=True,
    )
    X_train_all = build_compact6(
        train_df[
            spec["train_smiles_col"]
        ].astype(str).tolist()
    )
    y_train_all = pd.to_numeric(
        train_df[spec["train_y_col"]],
        errors="raise",
    ).to_numpy(dtype=np.float64)

    if not np.isfinite(y_train_all).all():
        raise ValueError(
            f"{context}: non-finite train targets"
        )

    if query_df is train_df:
        X_query_all = X_train_all
        y_query_all = y_train_all
    else:
        print(
            f"[{context}] Build six-descriptor set QUERY",
            flush=True,
        )
        X_query_all = build_compact6(
            query_df[
                spec["query_smiles_col"]
            ].astype(str).tolist()
        )
        y_query_all = pd.to_numeric(
            query_df[spec["query_y_col"]],
            errors="raise",
        ).to_numpy(dtype=np.float64)
        if not np.isfinite(y_query_all).all():
            raise ValueError(
                f"{context}: non-finite query targets"
            )

    meta = {
        "module": "evaluate_six_descriptor_rf",
        "context": context,
        "status": (
            "ADDED_MATCHED_CONVENTIONAL_CONTROL"
        ),
        "purpose": (
            "Complete the same-feature comparison for the "
            "frozen six-descriptor set relational experiment."
        ),
        "train_table": str(spec["train_table"]),
        "query_table": str(spec["query_table"]),
        "split_root": str(spec["split_root"]),
        "n_train_dataset": int(len(train_df)),
        "n_query_dataset": int(len(query_df)),
        "query_partition": spec["query_partition"],
        "features": COMPACT6,
        "feature_selection_in_this_script": False,
        "toxicity_used_to_choose_features_in_this_script": False,
        "rf": {
            "n_estimators": N_ESTIMATORS,
            "max_depth": MAX_DEPTH,
            "min_samples_leaf": MIN_SAMPLES_LEAF,
            "max_features": MAX_FEATURES,
            "bootstrap": True,
            "n_jobs": N_JOBS,
            "hyperparameter_tuning_in_this_script": False,
        },
        "training_policy": (
            "For each frozen structural split, fit RF on TRAIN only. "
            "Validation is not merged into training."
        ),
        "evaluation_policy": (
            "Daphnia/Fathead/WebTEST: frozen TEST partition. "
            "ADORE: all 274 independent external chemicals."
        ),
        "posthoc_status": (
            "Baseline added after prior main TEST results were observed; "
            "model/feature/RF specifications are fixed and not selected "
            "using the new six-descriptor RF evaluation outcomes."
        ),
        "split_seeds": SPLIT_SEEDS,
        "model_seeds": MODEL_SEEDS,
        "rdkit_version": rdBase.rdkitVersion,
        "checkpoint_resume": True,
    }
    atomic_write_json(
        out_dir / "00_meta.json",
        meta,
    )

    total_tasks = (
        len(SPLIT_SEEDS)
        * len(MODEL_SEEDS)
    )

    completed = sum(
        task_done(
            checkpoint_dir,
            task_stem(s, m),
        )
        for s in SPLIT_SEEDS
        for m in MODEL_SEEDS
    )

    print(
        f"[{context}] {completed}/{total_tasks} "
        "tasks already complete",
        flush=True,
    )

    task_i = 0

    for split_seed in SPLIT_SEEDS:
        idx = load_split_indices(
            spec["split_root"],
            split_seed,
            len(train_df),
        )
        tr = idx["train"]

        if spec["query_partition"] == "test":
            q_idx = idx["test"]
        else:
            q_idx = np.arange(
                len(query_df),
                dtype=int,
            )

        Xtr = X_train_all[tr]
        ytr = y_train_all[tr]

        Xq = X_query_all[q_idx]
        yq = y_query_all[q_idx]

        for model_seed in MODEL_SEEDS:
            task_i += 1
            stem = task_stem(
                split_seed,
                model_seed,
            )

            if task_done(
                checkpoint_dir, stem
            ):
                print(
                    f"[SKIP {task_i}/{total_tasks}] "
                    f"{context} split={split_seed} "
                    f"model_seed={model_seed}",
                    flush=True,
                )
                continue

            print(
                f"[RUN  {task_i}/{total_tasks}] "
                f"{context} split={split_seed} "
                f"model_seed={model_seed}",
                flush=True,
            )

            rf = make_rf(model_seed)
            rf.fit(Xtr, ytr)
            pred = rf.predict(Xq)

            mm = metrics(yq, pred)
            prefix = spec["metric_prefix"]

            result = {
                "context": context,
                "split_seed": int(split_seed),
                "model_seed": int(model_seed),
                "model": "six-descriptor RF",
                "n_train_dataset": int(
                    len(train_df)
                ),
                "n_query_dataset": int(
                    len(query_df)
                ),
                "n_train": int(len(tr)),
                "n_query": int(len(q_idx)),
                "n_features": 6,
                **{
                    f"{prefix}_{k}": v
                    for k, v in mm.items()
                },
            }

            atomic_save_npz(
                checkpoint_dir
                / f"{stem}.npz",
                query_row_index=q_idx.astype(
                    np.int64
                ),
                y_true=yq.astype(
                    np.float64
                ),
                y_pred=np.asarray(
                    pred, dtype=np.float64
                ),
            )

            atomic_write_json(
                checkpoint_dir
                / f"{stem}.json",
                {
                    "status": "DONE",
                    "split_seed": int(
                        split_seed
                    ),
                    "model_seed": int(
                        model_seed
                    ),
                    "result": result,
                },
            )

            run_df, pred_df = load_checkpoints(
                checkpoint_dir,
                spec["metric_prefix"],
            )
            atomic_write_csv(
                out_dir
                / "01_completed_run_metrics.csv",
                run_df.sort_values(
                    ["split_seed", "model_seed"]
                ).reset_index(drop=True),
            )
            atomic_write_csv(
                out_dir
                / "02_predictions.csv",
                pred_df.sort_values(
                    [
                        "split_seed",
                        "model_seed",
                        "query_row_index",
                    ]
                ).reset_index(drop=True),
            )

    run_df, pred_df = load_checkpoints(
        checkpoint_dir,
        spec["metric_prefix"],
    )
    if len(run_df) != total_tasks:
        raise RuntimeError(
            f"{context}: only "
            f"{len(run_df)}/{total_tasks} "
            "tasks complete. Re-run to resume."
        )

    summary, splitmeans = (
        hierarchical_summary(
            run_df,
            spec["metric_prefix"],
        )
    )

    atomic_write_csv(
        out_dir
        / "01_completed_run_metrics.csv",
        run_df.sort_values(
            ["split_seed", "model_seed"]
        ).reset_index(drop=True),
    )
    atomic_write_csv(
        out_dir / "02_predictions.csv",
        pred_df.sort_values(
            [
                "split_seed",
                "model_seed",
                "query_row_index",
            ]
        ).reset_index(drop=True),
    )
    atomic_write_csv(
        out_dir / "03_splitmeans.csv",
        splitmeans,
    )
    atomic_write_csv(
        out_dir
        / "04_hierarchical_summary.csv",
        summary,
    )

    lines = [
        f"[Six-descriptor RF | {context}]",
        "",
        f"Features      : {', '.join(COMPACT6)}",
        (
            "RF            : "
            f"{N_ESTIMATORS} trees, "
            f"max_depth={MAX_DEPTH}, "
            f"min_leaf={MIN_SAMPLES_LEAF}, "
            f"max_features={MAX_FEATURES}"
        ),
        "RF tuning     : NO",
        "Feature change: NO",
        "Train policy  : structural-split TRAIN only",
        f"Query stage   : {spec['query_partition']}",
        "Checkpoint    : YES",
        "",
        "[Hierarchical summary]",
        summary.to_string(index=False),
        "",
        "[Interpretation boundary]",
        (
            "- This is a same-feature conventional control "
            "for the already-frozen six-descriptor set relational experiment."
        ),
        (
            "- Keep the result regardless of whether RF, the "
            "non-relational neural model, or G2 performs best."
        ),
        (
            "- Do not use this result to retune six-descriptor set, RF, "
            "G2, K, fingerprint definition, or graph policy."
        ),
        (
            "- Full Descriptor-RF and Morgan-RF remain broader "
            "representation benchmarks, not same-input architecture controls."
        ),
    ]
    atomic_write_text(
        out_dir / "six_descriptor_rf_summary.txt",
        "\n".join(lines),
    )

    print(
        f"[DONE] {context}",
        flush=True,
    )
    print(
        summary.to_string(index=False),
        flush=True,
    )


def main() -> None:
    RDLogger.DisableLog("rdApp.*")

    for context, spec in CONTEXTS.items():
        for p in [
            spec["train_table"],
            spec["query_table"],
            spec["split_root"],
        ]:
            if not p.exists():
                raise FileNotFoundError(p)

    OUT_ROOT.mkdir(
        parents=True, exist_ok=True
    )

    top_meta = {
        "module": "evaluate_six_descriptor_rf",
        "contexts": list(CONTEXTS),
        "compact6": COMPACT6,
        "rf_fixed_spec": {
            "n_estimators": N_ESTIMATORS,
            "max_depth": MAX_DEPTH,
            "min_samples_leaf": MIN_SAMPLES_LEAF,
            "max_features": MAX_FEATURES,
            "bootstrap": True,
        },
        "scientific_note": (
            "This adds a conventional model with the same six "
            "molecule-level variables used by the frozen G2 / "
            "non-relational neural comparison. It does not replace "
            "the existing full-descriptor or Morgan benchmarks."
        ),
    }
    atomic_write_json(
        OUT_ROOT / "00_overall_meta.json",
        top_meta,
    )

    print(
        "[Six-descriptor RF matched comparison]",
        flush=True,
    )
    print(
        f"[OUT] {OUT_ROOT}",
        flush=True,
    )

    for context, spec in CONTEXTS.items():
        run_context(context, spec)

    combined = []
    for context, spec in CONTEXTS.items():
        p = (
            OUT_ROOT / context
            / "04_hierarchical_summary.csv"
        )
        d = pd.read_csv(
            p,
            encoding="utf-8-sig",
        )
        d.insert(0, "context", context)
        combined.append(d)

    combined_df = pd.concat(
        combined,
        ignore_index=True,
    )
    atomic_write_csv(
        OUT_ROOT
        / "all_contexts_hierarchical_summary.csv",
        combined_df,
    )

    print("\n[SIX-DESCRIPTOR RF COMPLETE]")
    print(
        combined_df.to_string(index=False),
        flush=True,
    )
    print(
        f"[DONE] {OUT_ROOT}",
        flush=True,
    )


if __name__ == "__main__":
    main()
