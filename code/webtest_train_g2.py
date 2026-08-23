#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
webtest_train_g2.py

Validation-only production training for revised WebTEST Daphnia large-N regime.

Scientific specification is inherited unchanged from the frozen G2 specification:
- six-descriptor set node X
- TRAIN-only preprocessing
- target = -log10(LC50 mol/L), TRAIN-only z-score for optimization
- Controlled GraphSAGE: hidden=128, 2 layers, dropout=.30
- Adam lr=1e-3, weight_decay=1e-4
- epochs max=500, patience=50
- conditions:
    descriptor_self
    descriptor_tanimoto_K10 (unweighted)
- 5 structural splits x 5 model initialization seeds
- validation only; TEST is NOT evaluated

Graph topology is read from exact precomputed Top-10 assets.
No Tanimoto graph is rebuilt here.

Saves best validation-selected model bundle for later frozen TEST evaluation.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from rdkit import Chem, RDLogger, rdBase
from rdkit.Chem import Descriptors

import g2_shared as shared
import g2_core as base

ROOT = Path(__file__).resolve().parents[1]

INPUT = (
    ROOT / "03_processed" / "WebTEST"
    / "WebTEST_Daphnia_consensus_unique_connectivity_frozen.csv"
)
SPLIT_ROOT = (
    ROOT / "04_splits" / "WebTEST_Daphnia"
    / "scaffold_split_murcko_topology_refined"
)
GRAPH_ROOT = (
    ROOT / "07_results" / "WebTEST_revised"
    / "top10_graph_assets"
)
OUT = (
    ROOT / "07_results" / "WebTEST_revised"
    / "g2_descriptor_validation"
)

EXPECTED_N = 119276
SMILES_COL = "SMILES_RDKit"
Y_COL = "target_neglog10_molar"

SPLIT_SEEDS = [42, 123, 777, 2025, 3407]
MODEL_SEEDS = [42, 123, 777, 2025, 3407]

COMPACT6 = [
    "MolLogP",
    "MolWt",
    "TPSA",
    "NumHAcceptors",
    "NumHDonors",
    "NumRotatableBonds",
]
FUNCS = {
    "MolLogP": Descriptors.MolLogP,
    "MolWt": Descriptors.MolWt,
    "TPSA": Descriptors.TPSA,
    "NumHAcceptors": Descriptors.NumHAcceptors,
    "NumHDonors": Descriptors.NumHDonors,
    "NumRotatableBonds": Descriptors.NumRotatableBonds,
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda")
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--patience", type=int, default=50)
    p.add_argument("--skip_existing", action="store_true")
    p.add_argument("--disable_rdkit_warnings", action="store_true")
    return p.parse_args()


def seed_everything(seed: int):
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def load_split(seed: int, n: int) -> Dict[str, np.ndarray]:
    p = SPLIT_ROOT / f"split_seed{seed}" / "split_membership.csv"
    if not p.exists():
        raise FileNotFoundError(p)
    m = pd.read_csv(p, encoding="utf-8-sig", low_memory=False)

    idx_col = next(
        (c for c in ["original_row_index", "row_idx", "index"] if c in m.columns),
        None,
    )
    if idx_col is None:
        raise KeyError(f"No row index column in {p}")

    out = {}
    for part in ["train", "val", "test"]:
        out[part] = np.sort(
            pd.to_numeric(
                m.loc[m["split"].astype(str).str.lower().eq(part), idx_col],
                errors="raise",
            ).astype(int).to_numpy()
        )

    all_idx = np.concatenate([out["train"], out["val"], out["test"]])
    if len(all_idx) != n or len(np.unique(all_idx)) != n:
        raise RuntimeError(f"Invalid split membership seed={seed}")
    return out


def build_compact6_raw(smiles: List[str]) -> np.ndarray:
    rows = []
    for i, smi in enumerate(smiles):
        if i % 10000 == 0:
            print(f"[six-descriptor set] {i}/{len(smiles)}", flush=True)
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            raise ValueError(f"Invalid SMILES row={i}: {smi}")
        vals = []
        for name in COMPACT6:
            v = float(FUNCS[name](mol))
            if not np.isfinite(v):
                raise ValueError(f"Non-finite {name} row={i}")
            vals.append(v)
        rows.append(vals)
    return np.asarray(rows, dtype=np.float64)


def load_precomputed_edges(
    seed: int,
    train_idx: np.ndarray,
    query_idx: np.ndarray,
    query_part: str,
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    """
    Returns:
      TRAIN edge pair localized to [0..n_train-1]
      TRAIN+QUERY edge pair localized to:
        train = [0..n_train-1]
        query = [n_train..n_train+n_query-1]

    Precomputed query assets store:
       source_global_idx = QUERY
       neighbor_train_global_idx = TRAIN
    Message direction required by frozen design is TRAIN -> QUERY.
    """
    seed_dir = GRAPH_ROOT / f"split_seed{seed}"

    trp = seed_dir / "train_undirected_union_exact_top10.npz"
    qp = seed_dir / f"{query_part}_source_to_train_exact_top10.npz"
    if not trp.exists():
        raise FileNotFoundError(trp)
    if not qp.exists():
        raise FileNotFoundError(qp)

    train_idx = np.asarray(train_idx, dtype=np.int64)
    query_idx = np.asarray(query_idx, dtype=np.int64)

    train_pos = {int(g): i for i, g in enumerate(train_idx)}
    query_pos = {int(g): i for i, g in enumerate(query_idx)}

    tr = np.load(trp)
    u = tr["u_global"].astype(np.int64)
    v = tr["v_global"].astype(np.int64)
    sim_uv = tr["similarity"].astype(np.float32)

    try:
        ul = np.asarray([train_pos[int(x)] for x in u], dtype=np.int64)
        vl = np.asarray([train_pos[int(x)] for x in v], dtype=np.int64)
    except KeyError as e:
        raise RuntimeError(f"TRAIN graph contains non-TRAIN node seed={seed}: {e}")

    # bidirectional message edges
    tr_src = np.concatenate([ul, vl])
    tr_dst = np.concatenate([vl, ul])
    tr_w = np.concatenate([sim_uv, sim_uv]).astype(np.float32)

    tr_ei = torch.tensor(
        np.vstack([tr_src, tr_dst]), dtype=torch.long
    )
    tr_ew = torch.tensor(tr_w, dtype=torch.float32)

    q = np.load(qp)
    q_global = q["source_global_idx"].astype(np.int64)
    nbr_global = q["neighbor_train_global_idx"].astype(np.int64)
    q_sim = q["similarity"].astype(np.float32)

    try:
        nbr_local = np.asarray(
            [train_pos[int(x)] for x in nbr_global], dtype=np.int64
        )
        q_local = np.asarray(
            [len(train_idx) + query_pos[int(x)] for x in q_global],
            dtype=np.int64,
        )
    except KeyError as e:
        raise RuntimeError(f"QUERY edge mapping failure seed={seed}/{query_part}: {e}")

    q_ei = torch.tensor(
        np.vstack([nbr_local, q_local]), dtype=torch.long
    )
    q_ew = torch.tensor(q_sim, dtype=torch.float32)

    # Validation/query graph includes the TRAIN graph plus incoming TRAIN->QUERY.
    val_ei = torch.cat([tr_ei, q_ei], dim=1)
    val_ew = torch.cat([tr_ew, q_ew], dim=0)

    # Strong direction checks.
    n_train = len(train_idx)
    if q_ei.numel():
        if not bool((q_ei[0] < n_train).all()):
            raise RuntimeError("Query graph has non-TRAIN message source.")
        if not bool((q_ei[1] >= n_train).all()):
            raise RuntimeError("Query graph destination is not QUERY.")
    return (tr_ei, tr_ew), (val_ei, val_ew)


def regression_metrics(y_true, y_pred):
    return shared.regression_metrics(
        np.asarray(y_true, dtype=float),
        np.asarray(y_pred, dtype=float),
    )


def train_one(
    X_train_np: np.ndarray,
    X_val_np: np.ndarray,
    y_train_orig: np.ndarray,
    y_val_orig: np.ndarray,
    train_edges: Tuple[torch.Tensor, torch.Tensor],
    val_edges: Tuple[torch.Tensor, torch.Tensor],
    mode: str,
    model_seed: int,
    args,
    device: torch.device,
):
    """
    Matched to the frozen six-descriptor G2 training logic; additionally returns best_state.
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
    tr_ei = tr_ei.to(device)
    tr_ew = tr_ew.to(device)
    va_ei = va_ei.to(device)
    va_ew = va_ew.to(device)

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
    val_local = np.arange(
        n_train, n_train + len(X_val_np), dtype=int
    )

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        optimizer.zero_grad()

        pred_train_z = model(X_train, tr_ei, tr_ew)
        loss = criterion(pred_train_z, y_train_t)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            pred_all_z = (
                model(X_valgraph, va_ei, va_ew)
                .detach().cpu().numpy()
            )
            pred_val_orig = pred_all_z[val_local] * y_std + y_mean

        val_rmse = float(
            np.sqrt(np.mean((y_val_orig - pred_val_orig) ** 2))
        )

        if val_rmse < best_val_rmse - 1e-12:
            best_val_rmse = val_rmse
            best_epoch = epoch
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
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
        pred_train_z = (
            model(X_train, tr_ei, tr_ew)
            .detach().cpu().numpy()
        )
        pred_all_z = (
            model(X_valgraph, va_ei, va_ew)
            .detach().cpu().numpy()
        )

    pred_train = pred_train_z * y_std + y_mean
    pred_val = pred_all_z[val_local] * y_std + y_mean

    result = {
        "best_epoch": int(best_epoch),
        "epochs_ran": int(epoch),
        "train_y_mean_for_z": y_mean,
        "train_y_std_for_z": y_std,
        **{
            f"train_{k}": v
            for k, v in regression_metrics(y_train_orig, pred_train).items()
        },
        **{
            f"val_{k}": v
            for k, v in regression_metrics(y_val_orig, pred_val).items()
        },
    }
    return result, pred_val.astype(float), best_state, y_mean, y_std


def main():
    args = parse_args()
    if args.disable_rdkit_warnings:
        RDLogger.DisableLog("rdApp.*")

    for p in [INPUT, SPLIT_ROOT, GRAPH_ROOT]:
        if not p.exists():
            raise FileNotFoundError(p)

    OUT.mkdir(parents=True, exist_ok=True)
    bundle_dir = OUT / "model_bundles"
    bundle_dir.mkdir(exist_ok=True)

    df = pd.read_csv(INPUT, encoding="utf-8-sig", low_memory=False).reset_index(drop=True)
    if len(df) != EXPECTED_N:
        raise RuntimeError(f"Expected N={EXPECTED_N}, found {len(df)}")
    for c in [SMILES_COL, Y_COL]:
        if c not in df.columns:
            raise KeyError(c)

    y_all = pd.to_numeric(df[Y_COL], errors="raise").to_numpy(dtype=np.float64)
    if not np.isfinite(y_all).all():
        raise RuntimeError("Non-finite y.")

    print("[1/4] Building frozen six-descriptor set raw X ...", flush=True)
    X_raw = build_compact6_raw(df[SMILES_COL].astype(str).tolist())

    device = torch.device(
        args.device
        if str(args.device).startswith("cuda") and torch.cuda.is_available()
        else "cpu"
    )
    print(f"[2/4] Device = {device}", flush=True)

    run_rows = []
    pred_rows = []
    pp_rows = []
    manifest_rows = []

    total_runs = len(SPLIT_SEEDS) * len(MODEL_SEEDS) * 2
    done = 0

    print("[3/4] Validation-only training ...", flush=True)

    for split_seed in SPLIT_SEEDS:
        idx = load_split(split_seed, len(df))
        train_idx = idx["train"]
        val_idx = idx["val"]

        # TEST index is deliberately loaded only as membership metadata;
        # no test y/prediction/edges are read or scored.
        n_test = len(idx["test"])

        pp = shared.fit_descriptor_preprocessor_train_only(
            X_raw=X_raw,
            train_idx=train_idx,
            descriptor_names=COMPACT6,
        )
        if pp["names"] != COMPACT6:
            raise RuntimeError(
                f"six-descriptor set changed after preprocessing split={split_seed}: {pp['names']}"
            )

        X_train = shared.transform_descriptor_rows(X_raw, train_idx, pp)
        X_val = shared.transform_descriptor_rows(X_raw, val_idx, pp)

        pp_rows.append({
            "split_seed": split_seed,
            **pp["qc"],
            "kept_descriptor_names": "|".join(pp["names"]),
        })

        real_train, real_val = load_precomputed_edges(
            seed=split_seed,
            train_idx=train_idx,
            query_idx=val_idx,
            query_part="val",
        )

        empty_train = (
            torch.empty((2, 0), dtype=torch.long),
            torch.empty((0,), dtype=torch.float32),
        )
        empty_val = (
            torch.empty((2, 0), dtype=torch.long),
            torch.empty((0,), dtype=torch.float32),
        )

        y_train = y_all[train_idx]
        y_val = y_all[val_idx]

        for model_seed in MODEL_SEEDS:
            conditions = [
                (
                    "descriptor_self",
                    "self",
                    np.nan,
                    empty_train,
                    empty_val,
                ),
                (
                    "descriptor_tanimoto_K10",
                    "unweighted",
                    10,
                    real_train,
                    real_val,
                ),
            ]

            for condition, mode, topk, tr_edges, va_edges in conditions:
                bundle_path = (
                    bundle_dir
                    / f"split{split_seed}__model{model_seed}__{condition}.pt"
                )

                if args.skip_existing and bundle_path.exists():
                    print(f"[SKIP existing] {bundle_path.name}", flush=True)
                    continue

                result, pred_val, state, y_mean, y_std = train_one(
                    X_train_np=X_train,
                    X_val_np=X_val,
                    y_train_orig=y_train,
                    y_val_orig=y_val,
                    train_edges=tr_edges,
                    val_edges=va_edges,
                    mode=mode,
                    model_seed=model_seed,
                    args=args,
                    device=device,
                )

                row = {
                    "split_seed": int(split_seed),
                    "model_seed": int(model_seed),
                    "condition": condition,
                    "relation_mode": mode,
                    "topk": topk,
                    "n_total": int(len(df)),
                    "n_train": int(len(train_idx)),
                    "n_val": int(len(val_idx)),
                    "n_test": int(n_test),
                    "n_features": 6,
                    "test_evaluated": False,
                    **result,
                }
                run_rows.append(row)

                for j, global_idx in enumerate(val_idx):
                    pred_rows.append({
                        "split_seed": int(split_seed),
                        "model_seed": int(model_seed),
                        "condition": condition,
                        "global_row_index": int(global_idx),
                        "y_true": float(y_val[j]),
                        "y_pred": float(pred_val[j]),
                    })

                bundle = {
                    "state_dict": state,
                    "split_seed": int(split_seed),
                    "model_seed": int(model_seed),
                    "condition": condition,
                    "relation_mode": mode,
                    "topk": None if pd.isna(topk) else int(topk),
                    "architecture": {
                        "input_dim": 6,
                        "hidden_dim": int(args.hidden_dim),
                        "num_layers": int(args.num_layers),
                        "dropout": float(args.dropout),
                    },
                    "optimizer_spec": {
                        "lr": float(args.lr),
                        "weight_decay": float(args.weight_decay),
                        "epochs_max": int(args.epochs),
                        "patience": int(args.patience),
                    },
                    "best_epoch": int(result["best_epoch"]),
                    "preprocessor": {
                        "kept_cols": np.asarray(pp["kept_cols"], dtype=int),
                        "medians": np.asarray(pp["medians"], dtype=float),
                        "mean": np.asarray(pp["mean"], dtype=float),
                        "std": np.asarray(pp["std"], dtype=float),
                        "names": list(pp["names"]),
                    },
                    "target_z": {
                        "mean": float(y_mean),
                        "std": float(y_std),
                    },
                    "n_total": int(len(df)),
                    "n_train": int(len(train_idx)),
                    "n_val": int(len(val_idx)),
                    "n_test": int(n_test),
                    "test_evaluated": False,
                    "scientific_spec": {
                        "node_X": "six-descriptor set",
                        "edge": (
                            "none"
                            if mode == "self"
                            else "Count Morgan r4/2048 exact Tanimoto Top10"
                        ),
                        "message_weighting": (
                            "none" if mode == "self" else "unweighted"
                        ),
                        "query_policy": (
                            "self"
                            if mode == "self"
                            else "TRAIN -> QUERY only"
                        ),
                    },
                }
                torch.save(bundle, bundle_path)

                manifest_rows.append({
                    "split_seed": int(split_seed),
                    "model_seed": int(model_seed),
                    "condition": condition,
                    "bundle_path": str(bundle_path),
                    "best_epoch": int(result["best_epoch"]),
                    "n_total": int(len(df)),
                    "n_train": int(len(train_idx)),
                    "n_val": int(len(val_idx)),
                    "n_test": int(n_test),
                    "test_evaluated": False,
                })

                done += 1
                print(
                    f"[{done}/{total_runs}] split={split_seed} model={model_seed} "
                    f"{condition} best_epoch={result['best_epoch']} "
                    f"val_RMSE={result['val_RMSE']:.6f}",
                    flush=True,
                )

    if not run_rows:
        if args.skip_existing:
            print("No new runs created because all requested bundles already exist.")
            print("Re-run without --skip_existing if summaries need rebuilding.")
            return
        raise RuntimeError("No runs completed.")

    run_df = pd.DataFrame(run_rows)
    pred_df = pd.DataFrame(pred_rows)
    pp_df = pd.DataFrame(pp_rows).drop_duplicates(subset=["split_seed"])
    manifest_df = pd.DataFrame(manifest_rows)

    summary_df = shared.hierarchical_summary(run_df)

    # Simple paired G2-self delta, same split/model seed.
    piv = run_df.pivot_table(
        index=["split_seed", "model_seed"],
        columns="condition",
        values="val_RMSE",
        aggfunc="first",
    ).reset_index()
    if {"descriptor_self", "descriptor_tanimoto_K10"}.issubset(piv.columns):
        piv["g2_minus_self_val_RMSE"] = (
            piv["descriptor_tanimoto_K10"] - piv["descriptor_self"]
        )
        split_delta = (
            piv.groupby("split_seed", as_index=False)["g2_minus_self_val_RMSE"]
            .mean()
        )
        delta_summary = pd.DataFrame([{
            "comparison": "G2_minus_six-descriptor set_self",
            "n_structural_splits": int(split_delta["split_seed"].nunique()),
            "model_seeds_per_split": int(len(MODEL_SEEDS)),
            "delta_splitmean_mean": float(split_delta["g2_minus_self_val_RMSE"].mean()),
            "delta_splitmean_std": float(split_delta["g2_minus_self_val_RMSE"].std(ddof=1)),
            "delta_splitmean_median": float(split_delta["g2_minus_self_val_RMSE"].median()),
            "n_splits_G2_better": int((split_delta["g2_minus_self_val_RMSE"] < 0).sum()),
            "n_splits_self_better": int((split_delta["g2_minus_self_val_RMSE"] > 0).sum()),
        }])
    else:
        piv = pd.DataFrame()
        split_delta = pd.DataFrame()
        delta_summary = pd.DataFrame()

    run_df.to_csv(OUT / "01_all_runs.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(
        OUT / "02_hierarchical_validation_summary.csv",
        index=False, encoding="utf-8-sig"
    )
    pred_df.to_csv(
        OUT / "03_validation_predictions.csv",
        index=False, encoding="utf-8-sig"
    )
    pp_df.to_csv(
        OUT / "04_compact6_trainonly_preprocess_qc.csv",
        index=False, encoding="utf-8-sig"
    )
    manifest_df.to_csv(
        OUT / "05_model_manifest.csv",
        index=False, encoding="utf-8-sig"
    )
    if len(piv):
        piv.to_csv(
            OUT / "06_paired_G2_self_by_split_modelseed.csv",
            index=False, encoding="utf-8-sig"
        )
        split_delta.to_csv(
            OUT / "07_paired_G2_self_splitmeans.csv",
            index=False, encoding="utf-8-sig"
        )
        delta_summary.to_csv(
            OUT / "08_paired_G2_self_delta_summary.csv",
            index=False, encoding="utf-8-sig"
        )

    meta = {
        "module": "webtest_train_g2",
        "n_total": len(df),
        "split_seeds": SPLIT_SEEDS,
        "model_seeds": MODEL_SEEDS,
        "test_evaluated": False,
        "target": "-log10(LC50 mol/L)",
        "node_X": COMPACT6,
        "preprocessing": "TRAIN-only, same the frozen six-descriptor G2 implementation",
        "graph_assets": str(GRAPH_ROOT),
        "graph_rebuilt_during_training": False,
        "G2": {
            "K": 10,
            "similarity": "exact Tanimoto",
            "fingerprint": "Count Morgan r4/2048 chirality=False",
            "message_weighting": "unweighted",
            "strict_inductive": "TRAIN -> QUERY only",
        },
        "model": {
            "family": "Controlled GraphSAGE",
            "hidden_dim": int(args.hidden_dim),
            "num_layers": int(args.num_layers),
            "dropout": float(args.dropout),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "epochs_max": int(args.epochs),
            "patience": int(args.patience),
        },
        "rdkit_version": rdBase.rdkitVersion,
        "torch_version": torch.__version__,
        "device": str(device),
    }
    (OUT / "g2_validation_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    lines = [
        "[WebTEST G2 / descriptor-only GraphSAGE validation]",
        f"N total              = {len(df)}",
        f"Structural splits    = {SPLIT_SEEDS}",
        f"Model seeds          = {MODEL_SEEDS}",
        "TEST evaluated       = NO",
        "Target               = -log10(LC50 mol/L)",
        "Node X               = six-descriptor set, TRAIN-only preprocessing",
        "G2 edge              = Count Morgan r4/2048 exact Tanimoto Top10",
        "G2 message passing   = unweighted",
        "Validation graph     = TRAIN -> VAL only",
        "",
        "[HIERARCHICAL VALIDATION SUMMARY]",
        summary_df.to_string(index=False),
        "",
        "[PAIRED G2 - SELF DELTA]",
        delta_summary.to_string(index=False) if len(delta_summary) else "not generated",
        "",
        "Interpretation rule fixed before TEST:",
        "- delta < 0 : G2 relational aggregation improves over descriptor-only",
        "- delta ~ 0 : no clear incremental relational gain",
        "- delta > 0 : relational aggregation is worse than descriptor-only",
        "- Do not change K, six-descriptor set, architecture, or graph policy after this result.",
        "",
        "STATUS: VALIDATION COMPLETE; frozen TEST remains unopened.",
    ]
    text = "\n".join(lines)
    (OUT / "g2_validation_summary.txt").write_text(text, encoding="utf-8")

    print("[4/4] DONE")
    print(text)


if __name__ == "__main__":
    main()
