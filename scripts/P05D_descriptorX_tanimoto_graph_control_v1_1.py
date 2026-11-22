#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare descriptor-only and Tanimoto-neighborhood GraphSAGE models on
TEST Daphnia structural validation splits.

Conditions: descriptor_self, descriptor_tanimoto_K5/K10,
and descriptor_shuffled_K5/K10.
Node features are RDKit descriptors, preprocessed using TRAIN only.
Target values are standardized on TRAIN and inverse-transformed for metrics.
Count Morgan (radius 4, 2048 dimensions, no chirality) defines the exact
Tanimoto graph; TRAIN uses an undirected Top-K union and VAL receives
TRAIN-to-VAL messages only. Message passing is unweighted.
The shuffled control permutes TRAIN identities while preserving the graph's
degree/component structure; query TRAIN sources use the same permutation.
Five structural splits and five model initialization seeds are summarized
hierarchically. Held-out TEST is not evaluated."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from rdkit import Chem, RDLogger, rdBase
from rdkit.Chem import Descriptors

try:
    import P05A_audit_G2_tanimoto_graphsage_v2 as base
except Exception as e:
    raise RuntimeError(
        "Could not import P05A_audit_G2_tanimoto_graphsage_v2.py. "
        "Place P05D in the same scripts folder as P05A v2."
    ) from e


DEFAULT_SPLIT_SEEDS = [42, 123, 777, 2025, 3407]
DEFAULT_MODEL_SEEDS = [42, 123, 777, 2025, 3407]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Descriptor-X + real/shuffled Tanimoto GraphSAGE control; validation only"
    )
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
        default="../07_results/Daphnia/P05D_descriptorX_tanimoto_graph_control",
    )
    p.add_argument(
        "--p05c_summary",
        default="../07_results/Daphnia/P05C_performance_sanity_RF_baselines/02_rf_hierarchical_summary.csv",
        help="Read-only optional comparison source.",
    )
    p.add_argument(
        "--p05b_summary",
        default="../07_results/Daphnia/P05B_G2_Morgan2048_multiseed_confirmation/02_hierarchical_validation_summary.csv",
        help="Read-only optional comparison source.",
    )
    p.add_argument("--smiles_col", default="canonical_SMILES_RDKit")
    p.add_argument("--y_col", default="Tox_original")
    p.add_argument("--split_seeds", nargs="*", type=int, default=DEFAULT_SPLIT_SEEDS)
    p.add_argument("--model_seeds", nargs="*", type=int, default=DEFAULT_MODEL_SEEDS)
    p.add_argument("--topk", nargs="*", type=int, default=[5, 10])

    p.add_argument("--min_similarity", type=float, default=0.0)
    p.add_argument("--radius", type=int, default=4)
    p.add_argument("--n_bits", type=int, default=2048)

    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--patience", type=int, default=50)
    p.add_argument("--device", default="cuda")
    p.add_argument("--disable_rdkit_warnings", action="store_true")
    return p.parse_args()


def resolve_path(path_str: str) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p
    return (Path(__file__).resolve().parent / p).resolve()


def build_rdkit_descriptor_raw(
    smiles: Sequence[str],
) -> Tuple[np.ndarray, List[str], pd.DataFrame]:
    rows: List[Dict[str, float]] = []

    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            raise ValueError(f"Invalid SMILES at row {i}: {smi}")

        vals = Descriptors.CalcMolDescriptors(mol, missingVal=np.nan, silent=True)
        clean = {}
        for k, v in vals.items():
            try:
                fv = float(v)
            except Exception:
                fv = np.nan
            if not np.isfinite(fv):
                fv = np.nan
            clean[str(k)] = fv
        rows.append(clean)

    ddf = pd.DataFrame(rows, dtype=float)
    names = ddf.columns.astype(str).tolist()
    X = ddf.to_numpy(dtype=np.float64)

    qc = pd.DataFrame({
        "descriptor": names,
        "n_nonfinite_all316": np.sum(~np.isfinite(X), axis=0).astype(int),
        "finite_fraction_all316": np.mean(np.isfinite(X), axis=0).astype(float),
    })
    return X, names, qc


def fit_descriptor_preprocessor_train_only(
    X_raw: np.ndarray,
    train_idx: np.ndarray,
    descriptor_names: Sequence[str],
) -> Dict:
    Xtr = np.asarray(X_raw[train_idx], dtype=np.float64)

    finite_any = np.isfinite(Xtr).any(axis=0)
    cols1 = np.where(finite_any)[0]
    if len(cols1) == 0:
        raise RuntimeError("All descriptors are TRAIN-all-missing.")

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

    # Defensive second variance filter after imputation.
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
        }
    }


def transform_descriptor_rows(
    X_raw: np.ndarray,
    row_idx: np.ndarray,
    pp: Dict,
) -> np.ndarray:
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


def remap_train_labels_in_edges(
    edge_index: torch.Tensor,
    n_train: int,
    permutation_old_to_new: np.ndarray,
) -> torch.Tensor:
    """
    Remap only TRAIN-local node ids (< n_train).
    Query node ids (>= n_train) stay fixed.
    """
    out = edge_index.clone()
    if out.numel() == 0:
        return out

    perm = torch.tensor(permutation_old_to_new, dtype=torch.long)
    for axis in [0, 1]:
        vals = out[axis]
        mask = vals < n_train
        if mask.any():
            out[axis, mask] = perm[vals[mask]]
    return out


def make_shuffled_edge_pair(
    train_edge_index: torch.Tensor,
    train_edge_weight: torch.Tensor,
    val_edge_index: torch.Tensor,
    val_edge_weight: torch.Tensor,
    n_train: int,
    shuffle_seed: int,
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor], np.ndarray]:
    rng = np.random.default_rng(int(shuffle_seed))
    perm = rng.permutation(n_train).astype(int)

    tr_ei = remap_train_labels_in_edges(train_edge_index, n_train, perm)
    va_ei = remap_train_labels_in_edges(val_edge_index, n_train, perm)

    # Unweighted GraphSAGE is used, but keep original weights for provenance.
    return (
        (tr_ei, train_edge_weight.clone()),
        (va_ei, val_edge_weight.clone()),
        perm,
    )


def seed_everything(seed: int) -> None:
    base.seed_all(int(seed))


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return base.regression_metrics(y_true, y_pred)


def fit_descriptor_graphsage(
    X_train_np: np.ndarray,
    X_val_np: np.ndarray,
    y_train_orig: np.ndarray,
    y_val_orig: np.ndarray,
    train_edges: Tuple[torch.Tensor, torch.Tensor],
    val_edges: Tuple[torch.Tensor, torch.Tensor],
    mode: str,
    model_seed: int,
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[Dict, np.ndarray]:
    """
    Train on TRAIN only; use VAL only for early stopping/model selection.
    y is z-scored using TRAIN only, then predictions are inverse-transformed.
    """
    seed_everything(model_seed)

    y_mean = float(np.mean(y_train_orig))
    y_std = float(np.std(y_train_orig, ddof=0))
    if not np.isfinite(y_std) or y_std <= 0:
        raise RuntimeError("Invalid TRAIN target std.")

    y_train_z = ((y_train_orig - y_mean) / y_std).astype(np.float32)

    X_train = torch.tensor(X_train_np, dtype=torch.float32, device=device)
    X_valgraph_np = np.concatenate([X_train_np, X_val_np], axis=0)
    X_valgraph = torch.tensor(X_valgraph_np, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train_z, dtype=torch.float32, device=device)

    tr_ei, tr_ew = train_edges
    va_ei, va_ew = val_edges
    tr_ei, tr_ew = tr_ei.to(device), tr_ew.to(device)
    va_ei, va_ew = va_ei.to(device), va_ew.to(device)

    model = base.ControlledGraphSAGE(
        input_dim=int(X_train_np.shape[1]),
        hidden_dim=int(args.hidden_dim),
        num_layers=int(args.num_layers),
        dropout=float(args.dropout),
        mode=mode,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )
    criterion = nn.MSELoss()

    best_state = None
    best_epoch = -1
    best_val_rmse = np.inf
    wait = 0

    n_train = len(X_train_np)
    val_local = np.arange(n_train, n_train + len(X_val_np), dtype=int)

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        optimizer.zero_grad()

        pred_train_z = model(X_train, tr_ei, tr_ew)
        loss = criterion(pred_train_z, y_train_t)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            pred_all_z = model(X_valgraph, va_ei, va_ew).detach().cpu().numpy()
            pred_val_orig = pred_all_z[val_local] * y_std + y_mean

        val_rmse = float(np.sqrt(np.mean((y_val_orig - pred_val_orig) ** 2)))
        if val_rmse < best_val_rmse - 1e-12:
            best_val_rmse = val_rmse
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= int(args.patience):
                break

    if best_state is None:
        raise RuntimeError("No best state captured.")

    model.load_state_dict(best_state)
    model.to(device)
    model.eval()

    with torch.no_grad():
        pred_train_z = model(X_train, tr_ei, tr_ew).detach().cpu().numpy()
        pred_all_z = model(X_valgraph, va_ei, va_ew).detach().cpu().numpy()

    pred_train = pred_train_z * y_std + y_mean
    pred_val = pred_all_z[val_local] * y_std + y_mean

    result = {
        "best_epoch": int(best_epoch),
        "train_y_mean_for_z": y_mean,
        "train_y_std_for_z": y_std,
        **{f"train_{k}": v for k, v in regression_metrics(y_train_orig, pred_train).items()},
        **{f"val_{k}": v for k, v in regression_metrics(y_val_orig, pred_val).items()},
    }
    return result, pred_val.astype(float)


def hierarchical_summary(run_df: pd.DataFrame) -> pd.DataFrame:
    metrics = ["val_RMSE", "val_MAE", "val_R2", "val_Pearson", "val_Spearman"]

    split_means = (
        run_df.groupby(
            ["split_seed", "condition", "relation_mode", "topk"],
            dropna=False,
            as_index=False,
        )[metrics]
        .mean()
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
        for m in metrics:
            row[f"{m}_splitmean_mean"] = float(g[m].mean())
            row[f"{m}_splitmean_std"] = float(g[m].std(ddof=1)) if len(g) > 1 else np.nan
            row[f"{m}_splitmean_median"] = float(g[m].median())
        rows.append(row)

    return pd.DataFrame(rows).sort_values(
        ["val_RMSE_splitmean_mean", "condition"],
        ascending=[True, True],
        na_position="last",
    ).reset_index(drop=True)


def paired_delta_summary(run_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Paired within same structural split and model seed.

    Negative delta means left named condition is better:
      real_minus_self = real - self
      shuffled_minus_self = shuffled - self
      real_minus_shuffled = real - shuffled
      real_k10_minus_k5 = real(K10) - real(K5)
    """
    raw_rows = []

    for (split_seed, model_seed), g in run_df.groupby(["split_seed", "model_seed"]):
        vals = {}
        for r in g.itertuples():
            key = (str(r.condition), None if pd.isna(r.topk) else int(r.topk))
            vals[key] = float(r.val_RMSE)

        self_rmse = vals[("descriptor_self", None)]

        for k in sorted(
            int(x) for x in g["topk"].dropna().unique()
        ):
            real = vals[(f"descriptor_tanimoto_K{k}", k)]
            shuf = vals[(f"descriptor_shuffled_K{k}", k)]

            raw_rows.extend([
                {
                    "split_seed": int(split_seed),
                    "model_seed": int(model_seed),
                    "topk": int(k),
                    "comparison": "real_minus_self",
                    "delta_val_RMSE": real - self_rmse,
                },
                {
                    "split_seed": int(split_seed),
                    "model_seed": int(model_seed),
                    "topk": int(k),
                    "comparison": "shuffled_minus_self",
                    "delta_val_RMSE": shuf - self_rmse,
                },
                {
                    "split_seed": int(split_seed),
                    "model_seed": int(model_seed),
                    "topk": int(k),
                    "comparison": "real_minus_shuffled",
                    "delta_val_RMSE": real - shuf,
                },
            ])

        if ("descriptor_tanimoto_K5", 5) in vals and ("descriptor_tanimoto_K10", 10) in vals:
            raw_rows.append({
                "split_seed": int(split_seed),
                "model_seed": int(model_seed),
                "topk": 10,
                "comparison": "real_k10_minus_k5",
                "delta_val_RMSE":
                    vals[("descriptor_tanimoto_K10", 10)]
                    - vals[("descriptor_tanimoto_K5", 5)],
            })

    raw = pd.DataFrame(raw_rows)

    split_delta = (
        raw.groupby(["split_seed", "topk", "comparison"], as_index=False)["delta_val_RMSE"]
        .mean()
        .rename(columns={"delta_val_RMSE": "split_mean_delta_val_RMSE"})
    )

    rows = []
    for keys, g in split_delta.groupby(["topk", "comparison"], dropna=False):
        topk, comp = keys
        arr = g["split_mean_delta_val_RMSE"].to_numpy(dtype=float)
        rows.append({
            "topk": topk,
            "comparison": comp,
            "n_structural_splits": int(len(arr)),
            "mean_split_delta_val_RMSE": float(np.mean(arr)),
            "std_split_delta_val_RMSE": float(np.std(arr, ddof=1)) if len(arr) > 1 else np.nan,
            "median_split_delta_val_RMSE": float(np.median(arr)),
            "n_splits_negative": int(np.sum(arr < 0)),
            "n_splits_positive": int(np.sum(arr > 0)),
            "min_split_delta_val_RMSE": float(np.min(arr)),
            "max_split_delta_val_RMSE": float(np.max(arr)),
        })

    summary = pd.DataFrame(rows).sort_values(["comparison", "topk"]).reset_index(drop=True)
    return raw, summary


def append_reference_comparison(
    p05d_summary: pd.DataFrame,
    p05c_path: Path,
    p05b_path: Path,
) -> pd.DataFrame:
    rows = []

    for r in p05d_summary.itertuples():
        rows.append({
            "source": "P05D",
            "model": str(r.condition),
            "relation_mode": str(r.relation_mode),
            "topk": r.topk,
            "val_RMSE_mean": float(r.val_RMSE_splitmean_mean),
            "val_MAE_mean": float(r.val_MAE_splitmean_mean),
            "val_R2_mean": float(r.val_R2_splitmean_mean),
            "val_Pearson_mean": float(r.val_Pearson_splitmean_mean),
            "val_Spearman_mean": float(r.val_Spearman_splitmean_mean),
        })

    if p05c_path.exists():
        c = pd.read_csv(p05c_path, encoding="utf-8-sig")
        for r in c.itertuples():
            rows.append({
                "source": "P05C",
                "model": str(r.model),
                "relation_mode": np.nan,
                "topk": np.nan,
                "val_RMSE_mean": float(r.val_RMSE_splitmean_mean),
                "val_MAE_mean": float(r.val_MAE_splitmean_mean),
                "val_R2_mean": float(r.val_R2_splitmean_mean),
                "val_Pearson_mean": float(r.val_Pearson_splitmean_mean),
                "val_Spearman_mean": float(r.val_Spearman_splitmean_mean),
            })

    if p05b_path.exists():
        b = pd.read_csv(p05b_path, encoding="utf-8-sig")
        for r in b.itertuples():
            rows.append({
                "source": "P05B",
                "model": "G2_Morgan2048_GraphSAGE",
                "relation_mode": str(r.relation_mode),
                "topk": r.topk,
                "val_RMSE_mean": float(r.val_RMSE_splitmean_mean),
                "val_MAE_mean": float(r.val_MAE_splitmean_mean),
                "val_R2_mean": float(r.val_R2_splitmean_mean),
                "val_Pearson_mean": float(r.val_Pearson_splitmean_mean),
                "val_Spearman_mean": float(r.val_Spearman_splitmean_mean),
            })

    return pd.DataFrame(rows).sort_values(
        "val_RMSE_mean", ascending=True
    ).reset_index(drop=True)


def main() -> None:
    args = parse_args()

    if args.disable_rdkit_warnings:
        RDLogger.DisableLog("rdApp.*")

    input_table = resolve_path(args.input_table)
    split_root = resolve_path(args.split_root)
    out_dir = resolve_path(args.out_dir)
    p05c_path = resolve_path(args.p05c_summary)
    p05b_path = resolve_path(args.p05b_summary)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_table, encoding="utf-8-sig", low_memory=False).reset_index(drop=True)
    for c in [args.smiles_col, args.y_col]:
        if c not in df.columns:
            raise KeyError(f"Missing required column: {c}")

    smiles = df[args.smiles_col].astype(str).tolist()
    y_all = pd.to_numeric(df[args.y_col], errors="raise").to_numpy(dtype=np.float64)
    if not np.isfinite(y_all).all():
        raise ValueError("Non-finite target values found.")

    print("[1/4] Building molecular assets: descriptors + Morgan fingerprints ...", flush=True)
    X_desc_raw, desc_names, desc_global_qc = build_rdkit_descriptor_raw(smiles)
    _, fps = base.build_count_morgan(
        smiles=smiles,
        radius=int(args.radius),
        n_bits=int(args.n_bits),
    )
    desc_global_qc.to_csv(
        out_dir / "00_descriptor_global_raw_qc.csv",
        index=False,
        encoding="utf-8-sig",
    )

    split_seeds = [int(x) for x in args.split_seeds]
    model_seeds = [int(x) for x in args.model_seeds]
    topks = sorted(set(int(x) for x in args.topk))
    if topks != [5, 10]:
        print(f"[WARN] audit scope was designed for K=[5,10]; received {topks}", flush=True)

    device = torch.device(
        args.device if str(args.device).startswith("cuda") and torch.cuda.is_available() else "cpu"
    )
    print(f"[2/4] Device: {device}", flush=True)

    run_rows: List[Dict] = []
    pred_rows: List[Dict] = []
    pp_rows: List[Dict] = []
    graph_qc_rows: List[Dict] = []
    shuffle_rows: List[Dict] = []

    # Per split/model seed:
    # self + real K5/K10 + shuffled K5/K10 = 5 conditions
    total_runs = len(split_seeds) * len(model_seeds) * (1 + 2 * len(topks))
    done_runs = 0

    print("[3/4] Running validation-only descriptor-X graph controls:", flush=True)

    for split_seed in split_seeds:
        idx = base.load_split_indices(split_root, split_seed, len(df))
        train_idx = idx["train"]
        val_idx = idx["val"]

        pp = fit_descriptor_preprocessor_train_only(
            X_raw=X_desc_raw,
            train_idx=train_idx,
            descriptor_names=desc_names,
        )
        X_train = transform_descriptor_rows(X_desc_raw, train_idx, pp)
        X_val = transform_descriptor_rows(X_desc_raw, val_idx, pp)

        pp_rows.append({
            "split_seed": int(split_seed),
            **pp["qc"],
            "kept_descriptor_names": "|".join(pp["names"]),
        })

        # SELF uses empty edge sets.
        empty_train = (
            torch.empty((2, 0), dtype=torch.long),
            torch.empty((0,), dtype=torch.float32),
        )
        empty_val = (
            torch.empty((2, 0), dtype=torch.long),
            torch.empty((0,), dtype=torch.float32),
        )

        graph_assets = {}
        for k in topks:
            tr_ei, tr_ew, tr_qc, _ = base.build_train_undirected_graph(
                train_global_idx=train_idx,
                fps=fps,
                topk=int(k),
                min_similarity=float(args.min_similarity),
            )
            va_ei, va_ew, va_qc = base.build_inductive_edges(
                train_global_idx=train_idx,
                query_global_idx=val_idx,
                fps=fps,
                topk=int(k),
                min_similarity=float(args.min_similarity),
            )

            graph_qc_rows.append({
                "split_seed": int(split_seed),
                "topk": int(k),
                **{f"train_{kk}": vv for kk, vv in tr_qc.items()},
                **{f"valgraph_{kk}": vv for kk, vv in va_qc.items()},
            })

            shuffle_seed = int(20_260_813 + split_seed * 100 + k)
            (sh_tr, sh_va, perm) = make_shuffled_edge_pair(
                train_edge_index=tr_ei,
                train_edge_weight=tr_ew,
                val_edge_index=va_ei,
                val_edge_weight=va_ew,
                n_train=len(train_idx),
                shuffle_seed=shuffle_seed,
            )

            shuffle_rows.append({
                "split_seed": int(split_seed),
                "topk": int(k),
                "shuffle_seed": int(shuffle_seed),
                "n_train": int(len(train_idx)),
                "permutation_is_identity": bool(np.array_equal(perm, np.arange(len(train_idx)))),
                "permutation_sha_like": "-".join(map(str, perm[:20].tolist())),
            })

            graph_assets[int(k)] = {
                "real_train": (tr_ei, tr_ew),
                "real_val": (va_ei, va_ew),
                "shuffled_train": sh_tr,
                "shuffled_val": sh_va,
            }

        y_train = y_all[train_idx]
        y_val = y_all[val_idx]

        for model_seed in model_seeds:
            conditions = [("descriptor_self", "self", np.nan, empty_train, empty_val)]

            for k in topks:
                ga = graph_assets[k]
                conditions.extend([
                    (
                        f"descriptor_tanimoto_K{k}",
                        "unweighted",
                        int(k),
                        ga["real_train"],
                        ga["real_val"],
                    ),
                    (
                        f"descriptor_shuffled_K{k}",
                        "unweighted",
                        int(k),
                        ga["shuffled_train"],
                        ga["shuffled_val"],
                    ),
                ])

            for condition, mode, topk_value, tr_edges, va_edges in conditions:
                result, pred_val = fit_descriptor_graphsage(
                    X_train_np=X_train,
                    X_val_np=X_val,
                    y_train_orig=y_train,
                    y_val_orig=y_val,
                    train_edges=tr_edges,
                    val_edges=va_edges,
                    mode=mode,
                    model_seed=int(model_seed),
                    args=args,
                    device=device,
                )

                run_rows.append({
                    "split_seed": int(split_seed),
                    "model_seed": int(model_seed),
                    "condition": condition,
                    "node_repr": "RDKit_2D_descriptors_train_z",
                    "edge_origin": (
                        "none" if mode == "self"
                        else "Morgan_count_r4_2048_Tanimoto"
                    ),
                    "relation_mode": mode,
                    "topk": topk_value,
                    "n_train": int(len(train_idx)),
                    "n_val": int(len(val_idx)),
                    "n_features": int(X_train.shape[1]),
                    "test_evaluated": False,
                    **result,
                })

                for j, global_idx in enumerate(val_idx):
                    pred_rows.append({
                        "split_seed": int(split_seed),
                        "model_seed": int(model_seed),
                        "condition": condition,
                        "topk": topk_value,
                        "global_row_index": int(global_idx),
                        "y_true": float(y_val[j]),
                        "y_pred": float(pred_val[j]),
                    })

                done_runs += 1
                pct = int(round(100 * done_runs / max(total_runs, 1)))
                print(
                    f"{pct}%",
                    end=" " if done_runs < total_runs else "\n",
                    flush=True,
                )

    run_df = pd.DataFrame(run_rows)
    pred_df = pd.DataFrame(pred_rows)
    pp_df = pd.DataFrame(pp_rows)
    graph_qc_df = pd.DataFrame(graph_qc_rows)
    shuffle_df = pd.DataFrame(shuffle_rows)

    summary_df = hierarchical_summary(run_df)
    delta_raw_df, delta_summary_df = paired_delta_summary(run_df)
    combined_df = append_reference_comparison(
        p05d_summary=summary_df,
        p05c_path=p05c_path,
        p05b_path=p05b_path,
    )

    run_df.to_csv(out_dir / "01_all_runs.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(out_dir / "02_hierarchical_validation_summary.csv", index=False, encoding="utf-8-sig")
    delta_raw_df.to_csv(out_dir / "03_paired_deltas_by_split_modelseed.csv", index=False, encoding="utf-8-sig")
    delta_summary_df.to_csv(out_dir / "04_hierarchical_delta_summary.csv", index=False, encoding="utf-8-sig")
    pred_df.to_csv(out_dir / "05_validation_predictions.csv", index=False, encoding="utf-8-sig")
    graph_qc_df.to_csv(out_dir / "06_graph_qc.csv", index=False, encoding="utf-8-sig")
    pp_df.to_csv(out_dir / "07_descriptor_trainonly_preprocess_qc.csv", index=False, encoding="utf-8-sig")
    shuffle_df.to_csv(out_dir / "08_shuffle_provenance.csv", index=False, encoding="utf-8-sig")
    combined_df.to_csv(out_dir / "09_comparison_with_RF_and_MorganG2.csv", index=False, encoding="utf-8-sig")

    meta = {
        "module": "P05D_descriptorX_tanimoto_graph_control",
        "input_table": str(input_table),
        "split_root": str(split_root),
        "n_chemicals": int(len(df)),
        "split_seeds": split_seeds,
        "model_seeds": model_seeds,
        "topk": topks,
        "test_evaluated": False,
        "node_X": {
            "representation": "RDKit 2D molecular descriptors",
            "raw_api": "Descriptors.CalcMolDescriptors",
            "data_dependent_preprocessing": "TRAIN only",
            "imputation": "TRAIN median",
            "zero_variance_filter": "TRAIN only",
            "scaling": "TRAIN mean/std z-score",
        },
        "target": {
            "column": args.y_col,
            "training_scaling": "TRAIN mean/std z-score",
            "metrics": "inverse-transformed original Tox scale",
        },
        "edge": {
            "fingerprint": f"Count Morgan radius={args.radius}, fpSize={args.n_bits}, chirality=False",
            "similarity": "Tanimoto",
            "train_graph": "undirected union of directed Top-K selections",
            "validation_graph": "TRAIN -> VAL only",
            "message_weighting": "unweighted",
        },
        "shuffled_control": {
            "type": "deterministic TRAIN node-label permutation of real graph",
            "preserves": "abstract graph topology, edge count, degree/component structure, query degree",
            "breaks": "chemical identity/features-to-Tanimoto-topology alignment",
            "held_fixed_across_model_seeds": True,
        },
        "gnn": {
            "family": "Controlled GraphSAGE-style",
            "hidden_dim": int(args.hidden_dim),
            "num_layers": int(args.num_layers),
            "dropout": float(args.dropout),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "epochs": int(args.epochs),
            "patience": int(args.patience),
        },
        "summary_policy": "model-seed mean within structural split, then summarize across structural splits",
        "rdkit_version": rdBase.rdkitVersion,
        "device_used": str(device),
    }
    (out_dir / "P05D_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    lines = [
        "[P05D DESCRIPTOR-X + TANIMOTO GRAPH CONTROL]",
        "",
        f"Input chemicals        : {len(df)}",
        f"Structural split seeds : {split_seeds}",
        f"Model seeds            : {model_seeds}",
        f"K values               : {topks}",
        "Frozen TEST scored     : NO",
        "Node X                 : RDKit 2D descriptors, TRAIN-only impute/filter/z-score",
        "Target training        : TRAIN-only z-score; metrics on original Tox scale",
        "Real edges             : Count Morgan r4/2048 Tanimoto Top-K",
        "GraphSAGE              : standard unweighted aggregation only",
        "Shuffled control       : degree/topology-preserving node-label permutation",
        "",
        "[Hierarchical validation summary]",
        summary_df.to_string(index=False),
        "",
        "[Paired delta summary]",
        delta_summary_df.to_string(index=False),
        "",
        "[Comparison with prior RF/G2 if found]",
        combined_df.to_string(index=False),
        "",
        "[Pre-specified interpretation]",
        "- real_minus_self < 0 across most structural splits:",
        "    Tanimoto-derived relational aggregation adds value beyond descriptor X alone.",
        "- real_minus_shuffled < 0 across most structural splits:",
        "    Improvement is specific to chemically meaningful Tanimoto topology, not arbitrary aggregation.",
        "- real ~= shuffled:",
        "    Any graph benefit may be generic aggregation/regularization rather than Tanimoto-specific topology.",
        "- Descriptor-G2 approaching or exceeding Descriptor-RF:",
        "    G2 performance bottleneck was largely the raw Morgan2048 neural node representation.",
        "- Descriptor-G2 still clearly worse than Descriptor-RF:",
        "    Diagnose GNN optimization/architecture only after this audit; do not touch frozen splits or TEST.",
    ]
    (out_dir / "P05D_summary.txt").write_text("\n".join(lines), encoding="utf-8")

    print("[4/4] Saved summaries and controls.", flush=True)
    print(f"[DONE] {out_dir}", flush=True)


if __name__ == "__main__":
    main()
