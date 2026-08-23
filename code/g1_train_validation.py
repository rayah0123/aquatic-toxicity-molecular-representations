#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
g1_train_validation.py

Validation-only G1 GraphSAGE trainer.

Protections
-----------
- graph asset and y are separate
- graph schema hash is verified
- graph/y SMILES row order is verified
- TRAIN-only y standardization
- early stopping uses validation RMSE on original target scale
- TEST metrics are NEVER computed here
- self-contained model bundle is saved for later frozen TEST/external inference

v1.1 hotfix
------------
PyG Data may expose y=None even for an unlabeled graph. attach_y() therefore
checks whether y is non-None before deleting it.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch_geometric.loader import DataLoader

import g1_core as core


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--graph_asset", required=True)
    p.add_argument("--y_csv", required=True)
    p.add_argument("--split_membership", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--run_name", required=True)
    p.add_argument("--split_seed", type=int, required=True)
    p.add_argument("--model_seed", type=int, required=True)

    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--patience", type=int, default=50)

    p.add_argument("--train_batch_size", type=int, default=1024)
    p.add_argument("--eval_batch_size", type=int, default=2048)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def load_split(path: Path, n_rows: int):
    m = pd.read_csv(path, encoding="utf-8-sig")
    if "split" not in m.columns:
        raise KeyError(f"Missing split column: {path}")

    idx_col = next(
        (
            c
            for c in [
                "original_row_index",
                "row_idx",
                "index",
                "global_row_index",
            ]
            if c in m.columns
        ),
        None,
    )
    if idx_col is None:
        raise KeyError(f"No row index column found in {path}")

    out = {}
    for part in ["train", "val", "test"]:
        out[part] = np.sort(
            pd.to_numeric(
                m.loc[m["split"].astype(str).str.lower().eq(part), idx_col],
                errors="raise",
            )
            .astype(int)
            .to_numpy()
        )

    all_idx = np.concatenate([out["train"], out["val"], out["test"]])
    if len(all_idx) != n_rows or len(np.unique(all_idx)) != n_rows:
        raise RuntimeError("Split membership does not cover each row exactly once.")

    if all_idx.min() < 0 or all_idx.max() >= n_rows:
        raise RuntimeError("Split membership contains out-of-range row indices.")

    return out


def attach_y(data_list, indices, y_z):
    """
    Clone unlabeled graph assets and attach standardized y only to the
    train/validation working copies. The cached graph assets remain unlabeled.
    """
    out = []
    for local_j, global_i in enumerate(indices):
        d = data_list[int(global_i)].clone()

        # PyG may expose y=None even if no actual target was stored.
        if getattr(d, "y", None) is not None:
            del d.y

        d.y = torch.tensor([float(y_z[local_j])], dtype=torch.float32)
        out.append(d)
    return out


def regression_metrics(y_true, y_pred):
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
        "Pearson": float(
            pd.Series(y_true).corr(pd.Series(y_pred), method="pearson")
        ),
        "Spearman": float(
            pd.Series(y_true).corr(pd.Series(y_pred), method="spearman")
        ),
    }


@torch.no_grad()
def predict_z(model, loader, device):
    model.eval()
    rows, ys, preds = [], [], []

    for batch in loader:
        batch = batch.to(device)
        pred = model(batch)

        rows.append(batch.row_idx.view(-1).cpu().numpy())
        ys.append(batch.y.view(-1).cpu().numpy())
        preds.append(pred.cpu().numpy())

    return (
        np.concatenate(rows),
        np.concatenate(ys),
        np.concatenate(preds),
    )


def main():
    args = parse_args()
    set_seed(args.model_seed)

    graph_asset = Path(args.graph_asset).resolve()
    y_csv = Path(args.y_csv).resolve()
    split_membership = Path(args.split_membership).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    obj = torch.load(graph_asset, map_location="cpu", weights_only=False)
    if not isinstance(obj, dict) or "data_list" not in obj or "meta" not in obj:
        raise RuntimeError("Unexpected graph asset format.")

    data_list = obj["data_list"]
    graph_meta = obj["meta"]

    if graph_meta.get("contains_y") is not False:
        raise RuntimeError("Graph asset is not explicitly unlabeled.")

    if graph_meta.get("graph_schema_hash") != core.graph_schema_hash():
        raise RuntimeError(
            "Graph asset schema hash does not match current g1_core.py."
        )

    # Verify every cached graph is still unlabeled and schema-compatible.
    for i, d in enumerate(data_list):
        core.validate_single_graph(
            d, row_idx=i, smiles=getattr(d, "smiles", None)
        )

    ydf = pd.read_csv(y_csv, encoding="utf-8-sig")
    required = {"global_row_index", "SMILES", "y"}
    if not required.issubset(ydf.columns):
        raise KeyError(f"y_csv missing required columns: {sorted(required)}")

    if len(ydf) != len(data_list):
        raise RuntimeError(
            f"Graph/y row count mismatch: {len(data_list)} vs {len(ydf)}"
        )

    expected_idx = np.arange(len(ydf), dtype=int)
    y_idx = pd.to_numeric(
        ydf["global_row_index"], errors="raise"
    ).astype(int).to_numpy()
    if not np.array_equal(y_idx, expected_idx):
        raise RuntimeError("y_csv global_row_index is not exactly 0..N-1.")

    y = pd.to_numeric(ydf["y"], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(y).all():
        raise ValueError("Non-finite y values found.")

    graph_smiles = [str(d.smiles) for d in data_list]
    y_smiles = ydf["SMILES"].astype(str).tolist()
    graph_smiles_hash = core.smiles_sequence_hash(graph_smiles)
    y_smiles_hash = core.smiles_sequence_hash(y_smiles)

    if graph_smiles_hash != y_smiles_hash:
        raise RuntimeError(
            "Graph/y SMILES order mismatch. "
            f"graph={graph_smiles_hash}, y={y_smiles_hash}"
        )

    split = load_split(split_membership, len(data_list))
    tr = split["train"]
    va = split["val"]

    # TEST membership is loaded above only to verify partition completeness.
    # No TEST y values, predictions, or metrics are accessed below.

    y_train = y[tr]
    y_val = y[va]

    y_mean = float(np.mean(y_train))
    y_std = float(np.std(y_train, ddof=0))
    if not np.isfinite(y_std) or y_std <= 0:
        raise RuntimeError("Invalid TRAIN y standard deviation.")

    y_train_z = (y_train - y_mean) / y_std
    y_val_z = (y_val - y_mean) / y_std

    train_graphs = attach_y(data_list, tr, y_train_z)
    val_graphs = attach_y(data_list, va, y_val_z)

    train_loader = DataLoader(
        train_graphs,
        batch_size=args.train_batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_graphs,
        batch_size=args.eval_batch_size,
        shuffle=False,
    )

    device = torch.device(
        args.device
        if str(args.device).startswith("cuda") and torch.cuda.is_available()
        else "cpu"
    )

    model = core.G1GraphSAGE(
        in_dim=core.ATOM_FEATURE_DIM,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    criterion = nn.MSELoss()

    history = []
    best_state = None
    best_epoch = -1
    best_val_rmse = np.inf
    wait = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            pred = model(batch)
            loss = criterion(pred, batch.y.view(-1).float())

            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))

        _, _, val_pred_z = predict_z(model, val_loader, device)
        val_pred = val_pred_z * y_std + y_mean
        val_rmse = float(np.sqrt(mean_squared_error(y_val, val_pred)))

        history.append(
            {
                "epoch": int(epoch),
                "train_loss_z": (
                    float(np.mean(losses)) if losses else np.nan
                ),
                "val_RMSE": val_rmse,
            }
        )

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= args.patience:
                break

    if best_state is None:
        raise RuntimeError("No best model state was saved.")

    model.load_state_dict(best_state)

    train_rows, train_true_z, train_pred_z = predict_z(
        model, train_loader, device
    )
    val_rows, val_true_z, val_pred_z = predict_z(
        model, val_loader, device
    )

    train_true = train_true_z * y_std + y_mean
    train_pred = train_pred_z * y_std + y_mean
    val_true = val_true_z * y_std + y_mean
    val_pred = val_pred_z * y_std + y_mean

    train_metrics = regression_metrics(train_true, train_pred)
    val_metrics = regression_metrics(val_true, val_pred)

    result = {
        "run_name": args.run_name,
        "split_seed": int(args.split_seed),
        "model_seed": int(args.model_seed),
        "n_train": int(len(tr)),
        "n_val": int(len(va)),
        "test_evaluated": False,
        "best_epoch": int(best_epoch),
        "device": str(device),
        **{f"train_{k}": v for k, v in train_metrics.items()},
        **{f"val_{k}": v for k, v in val_metrics.items()},
    }

    pd.DataFrame(history).to_csv(
        out_dir / "history.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Restore validation predictions to global row order explicitly.
    order = np.argsort(val_rows)
    pd.DataFrame(
        {
            "global_row_index": val_rows[order].astype(int),
            "y_true": val_true[order],
            "y_pred": val_pred[order],
        }
    ).to_csv(
        out_dir / "validation_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    bundle = {
        "bundle_version": "G1_MODEL_BUNDLE_V1_1",
        "run_name": args.run_name,
        "model_state_dict": {
            k: v.detach().cpu().clone()
            for k, v in model.state_dict().items()
        },
        "model_spec": core.model_spec(
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
        ),
        "graph_schema_hash": core.graph_schema_hash(),
        "graph_schema_spec": core.graph_schema_spec(),
        "graph_asset_smiles_sha256": graph_meta[
            "smiles_sequence_sha256"
        ],
        "target_transform": {
            "kind": "train_zscore",
            "train_mean": y_mean,
            "train_std": y_std,
        },
        "training_spec": {
            "split_seed": int(args.split_seed),
            "model_seed": int(args.model_seed),
            "split_membership": str(split_membership),
            "n_train": int(len(tr)),
            "n_val": int(len(va)),
            "optimizer": "Adam",
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "max_epochs": int(args.epochs),
            "patience": int(args.patience),
            "best_epoch": int(best_epoch),
            "train_batch_size": int(args.train_batch_size),
            "eval_batch_size": int(args.eval_batch_size),
            "selection_metric": "validation_RMSE_original_scale",
            "test_evaluated": False,
        },
    }

    bundle_path = out_dir / "model_bundle.pt"
    torch.save(bundle, bundle_path)

    result["bundle_path"] = str(bundle_path)
    result["bundle_sha256"] = core.file_sha256(bundle_path)
    result["graph_schema_hash"] = core.graph_schema_hash()
    result["graph_y_smiles_hash"] = graph_smiles_hash

    pd.DataFrame([result]).to_csv(
        out_dir / "result.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (out_dir / "result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(
        f"[DONE] split={args.split_seed} seed={args.model_seed} "
        f"val_RMSE={val_metrics['RMSE']:.6f} "
        f"val_R2={val_metrics['R2']:.6f} "
        f"best_epoch={best_epoch}"
    )
    print(f"[graph/y identity] PASS {graph_smiles_hash}")
    print("[TEST evaluated] NO")


if __name__ == "__main__":
    main()
