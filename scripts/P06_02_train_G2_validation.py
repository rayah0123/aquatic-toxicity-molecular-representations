#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train one frozen G2 or Compact6-self condition using TRAIN + VAL only.
TEST is never predicted here.
"""

from __future__ import annotations

import argparse, copy, json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import P06_00_g2_core as core


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--x_raw", required=True)
    p.add_argument("--rowmap_csv", required=True)
    p.add_argument("--y_csv", required=True)
    p.add_argument("--split_membership", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--run_name", required=True)
    p.add_argument("--split_seed", type=int, required=True)
    p.add_argument("--model_seed", type=int, required=True)
    p.add_argument("--condition", choices=["g2", "self"], default="g2")
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--patience", type=int, default=50)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def load_split(path: Path, n_rows: int):
    m = pd.read_csv(path, encoding="utf-8-sig")
    if "split" not in m.columns:
        raise KeyError("split")
    idx_col = next(
        (c for c in ["original_row_index","row_idx","index","global_row_index"] if c in m.columns),
        None,
    )
    if idx_col is None:
        raise KeyError("No row index column in split membership.")

    out = {}
    for part in ["train","val","test"]:
        out[part] = np.sort(
            pd.to_numeric(
                m.loc[m["split"].astype(str).str.lower().eq(part), idx_col],
                errors="raise",
            ).astype(int).to_numpy()
        )
    all_idx = np.concatenate([out["train"], out["val"], out["test"]])
    if len(all_idx) != n_rows or len(np.unique(all_idx)) != n_rows:
        raise RuntimeError("Split membership does not cover every row exactly once.")
    return out


def load_y(y_csv: Path, rowmap: pd.DataFrame):
    ydf = pd.read_csv(y_csv, encoding="utf-8-sig")
    req = {"global_row_index","SMILES","y"}
    if not req.issubset(ydf.columns):
        raise KeyError(f"y_csv missing {req}")
    if len(ydf) != len(rowmap):
        raise RuntimeError("rowmap/y row count mismatch.")
    if not np.array_equal(
        pd.to_numeric(ydf["global_row_index"], errors="raise").astype(int).to_numpy(),
        np.arange(len(ydf)),
    ):
        raise RuntimeError("y global_row_index is not 0..N-1.")
    if core.smiles_hash(ydf["SMILES"].astype(str).tolist()) != core.smiles_hash(rowmap["SMILES"].astype(str).tolist()):
        raise RuntimeError("rowmap/y SMILES order mismatch.")
    y = pd.to_numeric(ydf["y"], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(y).all():
        raise RuntimeError("Non-finite y.")
    return y


def main():
    args = parse_args()
    core.seed_all(args.model_seed)

    x_path = Path(args.x_raw).resolve()
    rowmap_path = Path(args.rowmap_csv).resolve()
    y_path = Path(args.y_csv).resolve()
    split_path = Path(args.split_membership).resolve()
    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    X_raw = np.load(x_path)
    rowmap = pd.read_csv(rowmap_path, encoding="utf-8-sig")
    smiles = rowmap["SMILES"].astype(str).tolist()

    if X_raw.shape != (len(rowmap), 6):
        raise RuntimeError(f"Unexpected X shape: {X_raw.shape}")
    y = load_y(y_path, rowmap)
    split = load_split(split_path, len(rowmap))
    tr, va = split["train"], split["val"]

    pp = core.fit_preprocessor_train_only(X_raw, tr)
    Xtr = core.transform_rows(X_raw, tr, pp)
    Xva = core.transform_rows(X_raw, va, pp)

    fps = core.build_count_morgan(smiles)

    tr_ei, tr_ew, tr_qc, _ = core.build_train_undirected_graph(tr, fps)
    va_ei, va_ew, va_qc = core.build_inductive_edges(tr, va, fps)

    if args.condition == "self":
        mode = "self"
        tr_ei = torch.empty((2,0), dtype=torch.long)
        tr_ew = torch.empty((0,), dtype=torch.float32)
        va_ei = torch.empty((2,0), dtype=torch.long)
        va_ew = torch.empty((0,), dtype=torch.float32)
    else:
        mode = "unweighted"

    ytr = y[tr]
    yva = y[va]
    y_mean = float(np.mean(ytr))
    y_std = float(np.std(ytr, ddof=0))
    if not np.isfinite(y_std) or y_std <= 0:
        raise RuntimeError("Invalid TRAIN y std.")

    ytr_z = ((ytr-y_mean)/y_std).astype(np.float32)

    device = torch.device(
        args.device if str(args.device).startswith("cuda") and torch.cuda.is_available()
        else "cpu"
    )

    Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=device)
    Xvg_t = torch.tensor(np.concatenate([Xtr, Xva], axis=0), dtype=torch.float32, device=device)
    ytr_t = torch.tensor(ytr_z, dtype=torch.float32, device=device)
    tr_ei, tr_ew = tr_ei.to(device), tr_ew.to(device)
    va_ei, va_ew = va_ei.to(device), va_ew.to(device)

    model = core.ControlledGraphSAGE(
        input_dim=6,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        mode=mode,
    ).to(device)

    opt = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    criterion = nn.MSELoss()

    best_state = None
    best_epoch = -1
    best_rmse = np.inf
    wait = 0
    hist = []
    va_local = np.arange(len(tr), len(tr)+len(va), dtype=int)

    for epoch in range(1, args.epochs+1):
        model.train()
        opt.zero_grad()
        pred_z = model(Xtr_t, tr_ei, tr_ew)
        loss = criterion(pred_z, ytr_t)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            pall_z = model(Xvg_t, va_ei, va_ew).detach().cpu().numpy()
        pva = pall_z[va_local]*y_std+y_mean
        rmse = float(np.sqrt(np.mean((yva-pva)**2)))
        hist.append({"epoch": epoch, "train_loss_z": float(loss.item()), "val_RMSE": rmse})

        if rmse < best_rmse - 1e-12:
            best_rmse = rmse
            best_epoch = epoch
            best_state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= args.patience:
                break

    if best_state is None:
        raise RuntimeError("No best state.")

    model.load_state_dict(best_state)
    model.to(device).eval()
    with torch.no_grad():
        ptr_z = model(Xtr_t, tr_ei, tr_ew).detach().cpu().numpy()
        pall_z = model(Xvg_t, va_ei, va_ew).detach().cpu().numpy()
    ptr = ptr_z*y_std+y_mean
    pva = pall_z[va_local]*y_std+y_mean

    mtr = core.regression_metrics(ytr, ptr)
    mva = core.regression_metrics(yva, pva)

    bundle = {
        "bundle_version": "G2_MODEL_BUNDLE_V1_0",
        "condition": args.condition,
        "mode": mode,
        "run_name": args.run_name,
        "model_state_dict": best_state,
        "schema_hash": core.schema_hash(),
        "spec": core.spec(),
        "preprocessor": {
            "names": pp["names"],
            "medians": np.asarray(pp["medians"]).tolist(),
            "mean": np.asarray(pp["mean"]).tolist(),
            "std": np.asarray(pp["std"]).tolist(),
        },
        "target_transform": {
            "kind": "train_zscore",
            "train_mean": y_mean,
            "train_std": y_std,
        },
        "training": {
            "split_seed": int(args.split_seed),
            "model_seed": int(args.model_seed),
            "train_indices": tr.astype(int).tolist(),
            "val_indices": va.astype(int).tolist(),
            "split_membership": str(split_path),
            "hidden_dim": int(args.hidden_dim),
            "num_layers": int(args.num_layers),
            "dropout": float(args.dropout),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "max_epochs": int(args.epochs),
            "patience": int(args.patience),
            "best_epoch": int(best_epoch),
            "test_evaluated": False,
        },
        "asset_identity": {
            "smiles_sha256": core.smiles_hash(smiles),
            "x_raw_sha256": core.file_sha256(x_path),
            "rowmap_sha256": core.file_sha256(rowmap_path),
        },
        "graph_qc": {
            "train": tr_qc,
            "validation": va_qc,
        },
    }

    bundle_path = out / "model_bundle.pt"
    torch.save(bundle, bundle_path)

    result = {
        "run_name": args.run_name,
        "condition": args.condition,
        "split_seed": int(args.split_seed),
        "model_seed": int(args.model_seed),
        "n_total": int(len(rowmap)),
        "n_train": int(len(tr)),
        "n_val": int(len(va)),
        "n_test": int(len(split["test"])),
        "best_epoch": int(best_epoch),
        "test_evaluated": False,
        **{f"train_{k}":v for k,v in mtr.items()},
        **{f"val_{k}":v for k,v in mva.items()},
        "bundle_path": str(bundle_path),
    }

    pd.DataFrame(hist).to_csv(out/"history.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({
        "global_row_index": va,
        "y_true": yva,
        "y_pred": pva,
    }).to_csv(out/"validation_predictions.csv", index=False, encoding="utf-8-sig")

    result["bundle_sha256"] = core.file_sha256(bundle_path)
    result["schema_hash"] = core.schema_hash()
    pd.DataFrame([result]).to_csv(out/"result.csv", index=False, encoding="utf-8-sig")
    (out/"result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(
        f"[DONE] {args.condition} split={args.split_seed} seed={args.model_seed} "
        f"n={len(rowmap)} train={len(tr)} val={len(va)} test={len(split['test'])} "
        f"val_RMSE={mva['RMSE']:.6f} best_epoch={best_epoch}"
    )
    print("[TEST evaluated] NO")


if __name__ == "__main__":
    main()
