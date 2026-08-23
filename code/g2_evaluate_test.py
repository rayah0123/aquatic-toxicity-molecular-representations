#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Frozen TEST evaluator for pre-frozen G2 and descriptor-only bundles.

No training, no hyperparameter selection, no representation changes.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import g2_core as core


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--x_raw", required=True)
    p.add_argument("--rowmap_csv", required=True)
    p.add_argument("--y_csv", required=True)
    p.add_argument("--split_root", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def load_test_idx(path: Path):
    m = pd.read_csv(path, encoding="utf-8-sig")
    idx_col = next(
        (c for c in ["original_row_index","row_idx","index","global_row_index"] if c in m.columns),
        None,
    )
    if idx_col is None:
        raise KeyError("No row index column.")
    return np.sort(
        pd.to_numeric(
            m.loc[m["split"].astype(str).str.lower().eq("test"), idx_col],
            errors="raise",
        ).astype(int).to_numpy()
    )


def main():
    args = parse_args()
    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(args.manifest, encoding="utf-8-sig")
    X_raw = np.load(args.x_raw)
    rowmap = pd.read_csv(args.rowmap_csv, encoding="utf-8-sig")
    ydf = pd.read_csv(args.y_csv, encoding="utf-8-sig")

    if X_raw.shape != (len(rowmap), 6) or len(ydf) != len(rowmap):
        raise RuntimeError("Asset row mismatch.")
    smiles = rowmap["SMILES"].astype(str).tolist()
    if core.smiles_hash(smiles) != core.smiles_hash(ydf["SMILES"].astype(str).tolist()):
        raise RuntimeError("rowmap/y SMILES order mismatch.")
    y = pd.to_numeric(ydf["y"], errors="raise").to_numpy(dtype=float)

    fps = core.build_count_morgan(smiles)
    device = torch.device(
        args.device if str(args.device).startswith("cuda") and torch.cuda.is_available()
        else "cpu"
    )

    run_rows = []
    pred_rows = []
    graph_qc_rows = []

    for r in manifest.itertuples(index=False):
        bundle_path = Path(r.bundle_path).resolve()
        if core.file_sha256(bundle_path) != str(r.bundle_sha256):
            raise RuntimeError(f"Bundle hash mismatch: {bundle_path}")

        bundle = torch.load(bundle_path, map_location="cpu", weights_only=False)
        if bundle.get("schema_hash") != core.schema_hash():
            raise RuntimeError(f"Schema mismatch: {bundle_path}")
        if bundle.get("condition") != r.condition:
            raise RuntimeError(f"Condition mismatch: {bundle_path}")

        tr = np.asarray(bundle["training"]["train_indices"], dtype=int)
        split_path = Path(args.split_root).resolve()/f"split_seed{int(r.split_seed)}"/"split_membership.csv"
        te = load_test_idx(split_path)

        pp = bundle["preprocessor"]
        Xtr = core.transform_rows(X_raw, tr, pp)
        Xte = core.transform_rows(X_raw, te, pp)

        if r.condition == "g2":
            ei, ew, qc = core.build_inductive_edges(tr, te, fps)
            mode = "unweighted"
        elif r.condition == "self":
            ei = torch.empty((2,0), dtype=torch.long)
            ew = torch.empty((0,), dtype=torch.float32)
            qc = {"n_train": int(len(tr)), "n_query": int(len(te)), "topk": np.nan}
            mode = "self"
        else:
            raise RuntimeError(f"Unexpected condition: {r.condition}")

        graph_qc_rows.append({
            "run_name": r.run_name,
            "condition": r.condition,
            "split_seed": int(r.split_seed),
            "model_seed": int(r.model_seed),
            **qc,
        })

        model = core.ControlledGraphSAGE(
            input_dim=6,
            hidden_dim=int(bundle["training"]["hidden_dim"]),
            num_layers=int(bundle["training"]["num_layers"]),
            dropout=float(bundle["training"]["dropout"]),
            mode=mode,
        ).to(device)
        model.load_state_dict(bundle["model_state_dict"])
        model.eval()

        Xall = torch.tensor(
            np.concatenate([Xtr, Xte], axis=0),
            dtype=torch.float32,
            device=device,
        )
        ei, ew = ei.to(device), ew.to(device)

        with torch.no_grad():
            pall_z = model(Xall, ei, ew).detach().cpu().numpy()

        local_te = np.arange(len(tr), len(tr)+len(te), dtype=int)
        tf = bundle["target_transform"]
        pred = pall_z[local_te]*float(tf["train_std"])+float(tf["train_mean"])

        met = core.regression_metrics(y[te], pred)
        run_rows.append({
            "run_name": r.run_name,
            "condition": r.condition,
            "split_seed": int(r.split_seed),
            "model_seed": int(r.model_seed),
            "n_total": int(len(rowmap)),
            "n_train": int(len(tr)),
            "n_val": int(r.n_val),
            "n_test": int(len(te)),
            **{f"test_{k}":v for k,v in met.items()},
        })
        for gi, yt, yp in zip(te, y[te], pred):
            pred_rows.append({
                "run_name": r.run_name,
                "condition": r.condition,
                "split_seed": int(r.split_seed),
                "model_seed": int(r.model_seed),
                "global_row_index": int(gi),
                "y_true": float(yt),
                "y_pred": float(yp),
            })

    rdf = pd.DataFrame(run_rows)
    rdf.to_csv(out/"g2_frozen_test_all_runs.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(pred_rows).to_csv(
        out/"g2_frozen_test_predictions.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(graph_qc_rows).to_csv(
        out/"g2_frozen_test_graph_qc.csv", index=False, encoding="utf-8-sig"
    )

    metric_cols = ["test_RMSE","test_MAE","test_R2","test_Pearson","test_Spearman"]
    split_means = rdf.groupby(["condition","split_seed"], as_index=False)[metric_cols].mean()
    counts = rdf.groupby(["condition","split_seed"], as_index=False)[["n_total","n_train","n_val","n_test"]].first()
    split_means = counts.merge(split_means, on=["condition","split_seed"], how="left")
    split_means.to_csv(
        out/"g2_frozen_test_splitmeans.csv", index=False, encoding="utf-8-sig"
    )

    summary = []
    for condition, g in split_means.groupby("condition"):
        row = {
            "condition": condition,
            "n_total": int(g["n_total"].iloc[0]),
            "n_structural_splits": int(len(g)),
            "train_n_min": int(g["n_train"].min()),
            "train_n_max": int(g["n_train"].max()),
            "val_n_min": int(g["n_val"].min()),
            "val_n_max": int(g["n_val"].max()),
            "test_n_min": int(g["n_test"].min()),
            "test_n_max": int(g["n_test"].max()),
        }
        for m in metric_cols:
            row[f"{m}_splitmean_mean"] = float(g[m].mean())
            row[f"{m}_splitmean_std"] = float(g[m].std(ddof=1))
            row[f"{m}_splitmean_median"] = float(g[m].median())
        summary.append(row)

    pd.DataFrame(summary).to_csv(
        out/"g2_frozen_test_hierarchical_summary.csv",
        index=False, encoding="utf-8-sig"
    )

    print(f"[DONE] {out}")
    print("[NOTE] Frozen TEST evaluated; no training occurred.")


if __name__ == "__main__":
    main()
