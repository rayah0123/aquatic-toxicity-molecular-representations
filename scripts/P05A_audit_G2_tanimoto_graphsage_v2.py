#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare molecular-feature and similarity-graph representations for
the TEST Daphnia validation splits.

Node representations: Count Morgan fingerprints (radius 4, 2048 dimensions,
no chirality) and TRAIN-fitted SVD representations (128 dimensions).
Conditions: self-only, unweighted Top-K Tanimoto aggregation, and
normalized Tanimoto-weighted Top-K aggregation.
Graph construction is strictly inductive: training compounds only in the
training graph; incoming TRAIN-to-VAL messages only for validation nodes.
SVD and early-stopping criteria use training/validation information only.
The held-out TEST partition is not evaluated in this script.

Writes validation metrics, paired comparisons, predictions, and graph QC."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


DEFAULT_SPLIT_SEEDS = [42, 123, 777, 2025, 3407]


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validation-only G2 Tanimoto/GraphSAGE design audit")
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
        default="../07_results/Daphnia/P05A_G2_design_audit",
    )
    p.add_argument("--smiles_col", default="canonical_SMILES_RDKit")
    p.add_argument("--y_col", default="Tox_original")
    p.add_argument("--split_seeds", nargs="*", type=int, default=DEFAULT_SPLIT_SEEDS)
    p.add_argument("--topk", nargs="*", type=int, default=[5, 10, 20],
                   help="K values used for validation-only model audit (default: 5 10 20)")
    p.add_argument("--topology_k", nargs="*", type=int, default=[3, 5, 10, 20, 30, 50],
                   help="K values used for label-free topology audit only")
    p.add_argument("--min_similarity", type=float, default=0.0,
                   help="Edges require similarity > this value. Default excludes zero-similarity edges.")
    p.add_argument("--radius", type=int, default=4)
    p.add_argument("--n_bits", type=int, default=2048)
    p.add_argument("--svd_dim", type=int, default=128)
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--patience", type=int, default=50)
    p.add_argument("--model_seed", type=int, default=20260813)
    p.add_argument("--device", default="cuda")
    p.add_argument("--disable_rdkit_warnings", action="store_true")
    return p.parse_args()


def resolve_path(path_str: str) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p
    # Script is expected under <project>/scripts/
    script_dir = Path(__file__).resolve().parent
    return (script_dir / p).resolve()


def build_count_morgan(
    smiles: Sequence[str], radius: int, n_bits: int
) -> Tuple[np.ndarray, List]:
    gen = rdFingerprintGenerator.GetMorganGenerator(
        radius=radius,
        fpSize=n_bits,
        includeChirality=False,
    )

    X = np.zeros((len(smiles), n_bits), dtype=np.float32)
    fps = []

    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            raise ValueError(f"Invalid SMILES at row {i}: {smi}")

        fp = gen.GetCountFingerprint(mol)
        fps.append(fp)
        for bit_idx, count in fp.GetNonzeroElements().items():
            if 0 <= int(bit_idx) < n_bits:
                X[i, int(bit_idx)] = float(count)

    return X, fps


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
            "Expected one of original_row_index, row_idx, index."
        )

    out = {}
    for name in ["train", "val", "test"]:
        arr = (
            pd.to_numeric(m.loc[m["split"].astype(str).str.lower().eq(name), idx_col], errors="raise")
            .astype(int)
            .to_numpy()
        )
        out[name] = np.sort(arr)

    all_idx = np.concatenate([out["train"], out["val"], out["test"]])
    if len(all_idx) != n_rows:
        raise RuntimeError(f"Split seed {seed}: assigned={len(all_idx)} expected={n_rows}")
    if len(np.unique(all_idx)) != n_rows:
        raise RuntimeError(f"Split seed {seed}: duplicate row assignment detected")
    if all_idx.min() < 0 or all_idx.max() >= n_rows:
        raise RuntimeError(f"Split seed {seed}: row index out of range")

    return out


def select_topk_train_neighbors(
    target_global_idx: int,
    train_global_idx: np.ndarray,
    fps: Sequence,
    topk: int,
    min_similarity: float,
    exclude_self: bool,
) -> List[Tuple[int, float]]:
    target_fp = fps[int(target_global_idx)]
    train_fps = [fps[int(i)] for i in train_global_idx]
    sims = DataStructs.BulkTanimotoSimilarity(target_fp, train_fps)

    candidates = []
    for train_pos, (global_idx, sim) in enumerate(zip(train_global_idx, sims)):
        if exclude_self and int(global_idx) == int(target_global_idx):
            continue
        sim = float(sim)
        if not np.isfinite(sim):
            continue
        if sim <= min_similarity:
            continue
        candidates.append((train_pos, sim))

    candidates.sort(key=lambda x: (-x[1], x[0]))
    return candidates[:topk]


def _component_stats(n_nodes: int, undirected_pairs: List[Tuple[int, int, float]]) -> Dict:
    """Connected-component and degree diagnostics on the undirected training graph."""
    parent = list(range(n_nodes))
    rank = [0] * n_nodes

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if rank[ra] < rank[rb]:
            parent[ra] = rb
        elif rank[ra] > rank[rb]:
            parent[rb] = ra
        else:
            parent[rb] = ra
            rank[ra] += 1

    degree = np.zeros(n_nodes, dtype=int)
    for u, v, _ in undirected_pairs:
        if u == v:
            continue
        union(int(u), int(v))
        degree[int(u)] += 1
        degree[int(v)] += 1

    comp_sizes: Dict[int, int] = {}
    for i in range(n_nodes):
        r = find(i)
        comp_sizes[r] = comp_sizes.get(r, 0) + 1
    sizes = sorted(comp_sizes.values(), reverse=True)
    largest = sizes[0] if sizes else 0
    second = sizes[1] if len(sizes) > 1 else 0
    isolates = int((degree == 0).sum())

    return {
        "n_components": int(len(sizes)),
        "largest_component_size": int(largest),
        "second_largest_component_size": int(second),
        "LCC_ratio": float(largest / n_nodes) if n_nodes else np.nan,
        "isolates": isolates,
        "degree_min": int(degree.min()) if n_nodes else 0,
        "degree_q25": float(np.quantile(degree, 0.25)) if n_nodes else np.nan,
        "degree_median": float(np.median(degree)) if n_nodes else np.nan,
        "degree_mean": float(degree.mean()) if n_nodes else np.nan,
        "degree_q75": float(np.quantile(degree, 0.75)) if n_nodes else np.nan,
        "degree_max": int(degree.max()) if n_nodes else 0,
    }


def build_train_undirected_graph(
    train_global_idx: np.ndarray,
    fps: Sequence,
    topk: int,
    min_similarity: float,
) -> Tuple[torch.Tensor, torch.Tensor, Dict, List[Tuple[int, int, float]]]:
    """Training-graph construction:
1. Each TRAIN node selects its Top-K TRAIN neighbors by Tanimoto similarity.
2. Directed selections form an undirected union graph.
3. Duplicate pair similarities are combined by maximum.
4. Undirected pairs are expanded in both message-passing directions."""
    pair_score: Dict[Tuple[int, int], float] = {}
    directed_selected_sims: List[float] = []

    for target_local, target_global in enumerate(train_global_idx):
        nbrs = select_topk_train_neighbors(
            target_global_idx=int(target_global),
            train_global_idx=train_global_idx,
            fps=fps,
            topk=topk,
            min_similarity=min_similarity,
            exclude_self=True,
        )
        for nbr_local, sim in nbrs:
            u, v = sorted((int(target_local), int(nbr_local)))
            if u == v:
                continue
            pair_score[(u, v)] = max(float(sim), pair_score.get((u, v), -np.inf))
            directed_selected_sims.append(float(sim))

    undirected_pairs = [(u, v, float(score)) for (u, v), score in sorted(pair_score.items())]

    src: List[int] = []
    dst: List[int] = []
    weights: List[float] = []
    unique_pair_sims: List[float] = []
    for u, v, score in undirected_pairs:
        src.extend([u, v])
        dst.extend([v, u])
        weights.extend([score, score])
        unique_pair_sims.append(score)

    if src:
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_weight = torch.tensor(weights, dtype=torch.float32)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_weight = torch.empty((0,), dtype=torch.float32)

    def sim_stats(vals: List[float]) -> Dict:
        a = np.asarray(vals, dtype=float)
        if a.size == 0:
            return {
                "unique_edge_n": 0,
                "sim_min": np.nan,
                "sim_q05": np.nan,
                "sim_q25": np.nan,
                "sim_median": np.nan,
                "sim_q75": np.nan,
                "sim_max": np.nan,
                "sim_mean": np.nan,
            }
        return {
            "unique_edge_n": int(a.size),
            "sim_min": float(np.min(a)),
            "sim_q05": float(np.quantile(a, 0.05)),
            "sim_q25": float(np.quantile(a, 0.25)),
            "sim_median": float(np.median(a)),
            "sim_q75": float(np.quantile(a, 0.75)),
            "sim_max": float(np.max(a)),
            "sim_mean": float(np.mean(a)),
        }

    qc = {
        "n_train": int(len(train_global_idx)),
        "topk": int(topk),
        "min_similarity": float(min_similarity),
        "directed_topk_selection_n": int(len(directed_selected_sims)),
        "message_edge_n": int(edge_index.shape[1]),
        **sim_stats(unique_pair_sims),
        **_component_stats(len(train_global_idx), undirected_pairs),
    }
    return edge_index, edge_weight, qc, undirected_pairs


def build_query_edges(
    train_global_idx: np.ndarray,
    query_global_idx: np.ndarray,
    fps: Sequence,
    topk: int,
    min_similarity: float,
) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    """One-way TRAIN -> QUERY edges. Query nodes never message back to training nodes."""
    src: List[int] = []
    dst: List[int] = []
    weights: List[float] = []
    sims_all: List[float] = []
    degrees: List[int] = []
    n_train = len(train_global_idx)

    for q_pos, q_global in enumerate(query_global_idx):
        target_local = n_train + q_pos
        nbrs = select_topk_train_neighbors(
            target_global_idx=int(q_global),
            train_global_idx=train_global_idx,
            fps=fps,
            topk=topk,
            min_similarity=min_similarity,
            exclude_self=False,
        )
        degrees.append(len(nbrs))
        for nbr_local, sim in nbrs:
            src.append(int(nbr_local))
            dst.append(int(target_local))
            weights.append(float(sim))
            sims_all.append(float(sim))

    if src:
        ei = torch.tensor([src, dst], dtype=torch.long)
        ew = torch.tensor(weights, dtype=torch.float32)
    else:
        ei = torch.empty((2, 0), dtype=torch.long)
        ew = torch.empty((0,), dtype=torch.float32)

    a = np.asarray(sims_all, dtype=float)
    qc = {
        "n_query": int(len(query_global_idx)),
        "query_message_edge_n": int(len(src)),
        "query_neighbor_count_min": int(np.min(degrees)) if degrees else 0,
        "query_neighbor_count_mean": float(np.mean(degrees)) if degrees else 0.0,
        "query_neighbor_count_max": int(np.max(degrees)) if degrees else 0,
        "query_zero_neighbor_n": int(sum(d == 0 for d in degrees)),
        "query_sim_min": float(np.min(a)) if a.size else np.nan,
        "query_sim_q05": float(np.quantile(a, 0.05)) if a.size else np.nan,
        "query_sim_median": float(np.median(a)) if a.size else np.nan,
        "query_sim_mean": float(np.mean(a)) if a.size else np.nan,
        "query_sim_max": float(np.max(a)) if a.size else np.nan,
    }
    return ei, ew, qc


def build_inductive_edges(
    train_global_idx: np.ndarray,
    query_global_idx: np.ndarray,
    fps: Sequence,
    topk: int,
    min_similarity: float,
) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    """
    Local node order = [train nodes, query nodes].

    TRAIN graph: undirected Top-K union (expanded both directions).
    QUERY graph: only incoming TRAIN -> QUERY edges.
    No QUERY -> TRAIN and no QUERY -> QUERY edges.
    """
    tr_ei, tr_ew, tr_qc, _ = build_train_undirected_graph(
        train_global_idx=train_global_idx,
        fps=fps,
        topk=topk,
        min_similarity=min_similarity,
    )

    if len(query_global_idx) == 0:
        return tr_ei, tr_ew, {**tr_qc, "n_query": 0}

    q_ei, q_ew, q_qc = build_query_edges(
        train_global_idx=train_global_idx,
        query_global_idx=query_global_idx,
        fps=fps,
        topk=topk,
        min_similarity=min_similarity,
    )

    if tr_ei.numel() == 0:
        edge_index = q_ei
        edge_weight = q_ew
    elif q_ei.numel() == 0:
        edge_index = tr_ei
        edge_weight = tr_ew
    else:
        edge_index = torch.cat([tr_ei, q_ei], dim=1)
        edge_weight = torch.cat([tr_ew, q_ew], dim=0)

    return edge_index, edge_weight, {**tr_qc, **q_qc}

def normalized_neighbor_mean(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor | None,
    weighted: bool,
    eps: float = 1e-12,
) -> torch.Tensor:
    n = x.size(0)
    if edge_index.numel() == 0:
        return torch.zeros_like(x)

    src = edge_index[0]
    dst = edge_index[1]

    if weighted:
        if edge_weight is None:
            raise ValueError("weighted=True requires edge_weight")
        w = edge_weight.to(dtype=x.dtype, device=x.device)
    else:
        w = torch.ones(src.numel(), dtype=x.dtype, device=x.device)

    msg = x[src] * w.unsqueeze(1)
    agg = torch.zeros_like(x)
    agg.index_add_(0, dst, msg)

    denom = torch.zeros(n, dtype=x.dtype, device=x.device)
    denom.index_add_(0, dst, w)
    denom = denom.clamp_min(eps).unsqueeze(1)

    return agg / denom


class ControlledSAGELayer(nn.Module):
    """
    Controlled GraphSAGE-style layer for ablation.

    self_only=True:
        W_self x_i

    self_only=False, weighted=False:
        W_self x_i + W_neigh mean_j(x_j)

    self_only=False, weighted=True:
        W_self x_i + W_neigh [sum_j s_ji x_j / sum_j s_ji]
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.lin_self = nn.Linear(in_dim, out_dim, bias=True)
        self.lin_neigh = nn.Linear(in_dim, out_dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None,
        mode: str,
    ) -> torch.Tensor:
        out = self.lin_self(x)
        if mode == "self":
            return out

        if mode not in {"unweighted", "weighted"}:
            raise ValueError(f"Unsupported mode: {mode}")

        neigh = normalized_neighbor_mean(
            x=x,
            edge_index=edge_index,
            edge_weight=edge_weight,
            weighted=(mode == "weighted"),
        )
        return out + self.lin_neigh(neigh)


class ControlledGraphSAGE(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        mode: str,
    ):
        super().__init__()
        self.mode = mode
        self.dropout = float(dropout)
        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList()

        self.layers.append(ControlledSAGELayer(input_dim, hidden_dim))
        self.bns.append(nn.BatchNorm1d(hidden_dim))
        for _ in range(num_layers - 1):
            self.layers.append(ControlledSAGELayer(hidden_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        self.head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None,
    ) -> torch.Tensor:
        h = x
        for layer, bn in zip(self.layers, self.bns):
            h = layer(h, edge_index, edge_weight, self.mode)
            h = bn(h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        return self.head(h).view(-1)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
        "Pearson": float(pd.Series(y_true).corr(pd.Series(y_pred), method="pearson")),
        "Spearman": float(pd.Series(y_true).corr(pd.Series(y_pred), method="spearman")),
    }


def fit_one_run(
    X_all: np.ndarray,
    y_all: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    train_edges: Tuple[torch.Tensor, torch.Tensor],
    val_edges: Tuple[torch.Tensor, torch.Tensor],
    mode: str,
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[Dict, pd.DataFrame]:
    # Use the same model initialization seed for every condition.
    seed_all(args.model_seed)

    X_train = torch.tensor(X_all[train_idx], dtype=torch.float32, device=device)
    y_train = torch.tensor(y_all[train_idx], dtype=torch.float32, device=device)

    combined_idx = np.concatenate([train_idx, val_idx])
    X_valgraph = torch.tensor(X_all[combined_idx], dtype=torch.float32, device=device)

    tr_ei, tr_ew = train_edges
    va_ei, va_ew = val_edges
    tr_ei = tr_ei.to(device)
    tr_ew = tr_ew.to(device)
    va_ei = va_ei.to(device)
    va_ew = va_ew.to(device)

    model = ControlledGraphSAGE(
        input_dim=X_all.shape[1],
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        mode=mode,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    criterion = nn.MSELoss()

    best_state = None
    best_epoch = -1
    best_val_rmse = np.inf
    wait = 0

    val_local_start = len(train_idx)
    val_local = np.arange(val_local_start, val_local_start + len(val_idx), dtype=int)

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        pred_train = model(X_train, tr_ei, tr_ew)
        loss = criterion(pred_train, y_train)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            pred_all_valgraph = model(X_valgraph, va_ei, va_ew)
            pred_val = pred_all_valgraph[val_local].detach().cpu().numpy()

        val_rmse = float(np.sqrt(mean_squared_error(y_all[val_idx], pred_val)))
        if val_rmse < best_val_rmse - 1e-12:
            best_val_rmse = val_rmse
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= args.patience:
                break

    if best_state is None:
        raise RuntimeError("No best model state captured")

    model.load_state_dict(best_state)
    model.to(device)
    model.eval()

    with torch.no_grad():
        pred_train = model(X_train, tr_ei, tr_ew).detach().cpu().numpy()
        pred_all_valgraph = model(X_valgraph, va_ei, va_ew).detach().cpu().numpy()
        pred_val = pred_all_valgraph[val_local]

    train_metrics = regression_metrics(y_all[train_idx], pred_train)
    val_metrics = regression_metrics(y_all[val_idx], pred_val)

    result = {
        "best_epoch": int(best_epoch),
        **{f"train_{k}": v for k, v in train_metrics.items()},
        **{f"val_{k}": v for k, v in val_metrics.items()},
    }

    pred_df = pd.DataFrame({
        "global_row_index": val_idx.astype(int),
        "y_true": y_all[val_idx].astype(float),
        "y_pred": pred_val.astype(float),
    })
    return result, pred_df


def make_summary(run_df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["node_repr", "relation_mode", "topk"]
    rows = []
    for keys, g in run_df.groupby(group_cols, dropna=False):
        node_repr, relation_mode, topk = keys
        row = {
            "node_repr": node_repr,
            "relation_mode": relation_mode,
            "topk": topk,
            "n_splits": int(len(g)),
        }
        for metric in ["val_RMSE", "val_MAE", "val_R2", "val_Pearson", "val_Spearman"]:
            row[f"{metric}_mean"] = float(g[metric].mean())
            row[f"{metric}_std"] = float(g[metric].std(ddof=1)) if len(g) > 1 else np.nan
            row[f"{metric}_median"] = float(g[metric].median())
        rows.append(row)
    out = pd.DataFrame(rows)
    return out.sort_values(["val_RMSE_mean", "node_repr", "relation_mode"], ascending=[True, True, True])


def make_pairwise_deltas(run_df: pd.DataFrame) -> pd.DataFrame:
    """Negative delta RMSE means the model named in the delta is better."""
    rows = []
    for seed in sorted(run_df["split_seed"].unique()):
        d = run_df.loc[run_df["split_seed"].eq(seed)].copy()
        for node_repr in sorted(d["node_repr"].unique()):
            dn = d.loc[d["node_repr"].eq(node_repr)]
            self_row = dn.loc[dn["relation_mode"].eq("self")]
            if len(self_row) != 1:
                continue
            self_rmse = float(self_row.iloc[0]["val_RMSE"])

            for topk in sorted(dn.loc[dn["relation_mode"].ne("self"), "topk"].dropna().unique()):
                du = dn.loc[(dn["relation_mode"].eq("unweighted")) & (dn["topk"].eq(topk))]
                dw = dn.loc[(dn["relation_mode"].eq("weighted")) & (dn["topk"].eq(topk))]
                if len(du) == 1:
                    u = float(du.iloc[0]["val_RMSE"])
                    rows.append({
                        "split_seed": int(seed), "node_repr": node_repr, "topk": int(topk),
                        "comparison": "unweighted_minus_self", "delta_val_RMSE": u - self_rmse,
                    })
                if len(dw) == 1:
                    w = float(dw.iloc[0]["val_RMSE"])
                    rows.append({
                        "split_seed": int(seed), "node_repr": node_repr, "topk": int(topk),
                        "comparison": "weighted_minus_self", "delta_val_RMSE": w - self_rmse,
                    })
                if len(du) == 1 and len(dw) == 1:
                    u = float(du.iloc[0]["val_RMSE"])
                    w = float(dw.iloc[0]["val_RMSE"])
                    rows.append({
                        "split_seed": int(seed), "node_repr": node_repr, "topk": int(topk),
                        "comparison": "weighted_minus_unweighted", "delta_val_RMSE": w - u,
                    })
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    if args.disable_rdkit_warnings:
        RDLogger.DisableLog("rdApp.*")

    input_table = resolve_path(args.input_table)
    split_root = resolve_path(args.split_root)
    out_dir = resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")

    df = pd.read_csv(input_table, encoding="utf-8-sig", low_memory=False).reset_index(drop=True)
    for col in [args.smiles_col, args.y_col]:
        if col not in df.columns:
            raise KeyError(f"Missing required column: {col}")

    y_all = pd.to_numeric(df[args.y_col], errors="raise").to_numpy(dtype=np.float32)
    if not np.isfinite(y_all).all():
        raise ValueError("Non-finite target values found")

    X_morgan, fps = build_count_morgan(
        df[args.smiles_col].astype(str).tolist(),
        radius=args.radius,
        n_bits=args.n_bits,
    )

    run_rows: List[Dict] = []
    graph_qc_rows: List[Dict] = []
    topology_rows: List[Dict] = []
    prediction_rows: List[pd.DataFrame] = []
    split_audit_rows: List[Dict] = []

    for split_seed in args.split_seeds:
        idx = load_split_indices(split_root, split_seed, len(df))
        train_idx, val_idx, test_idx = idx["train"], idx["val"], idx["test"]

        # Label-free topology audit across a broader K grid. No y and no model fitting.
        for k_top in sorted(set(args.topology_k)):
            if k_top <= 0:
                raise ValueError("topology_k must be positive")
            effective_k_top = min(int(k_top), max(len(train_idx) - 1, 1))
            _, _, topo_qc, _ = build_train_undirected_graph(
                train_global_idx=train_idx,
                fps=fps,
                topk=effective_k_top,
                min_similarity=args.min_similarity,
            )
            topology_rows.append({
                "split_seed": int(split_seed),
                "requested_topk": int(k_top),
                "effective_topk": int(effective_k_top),
                **topo_qc,
            })

        # Train-only SVD. The frozen test subset is transformed nowhere in this audit.
        max_svd_dim = min(args.svd_dim, len(train_idx) - 1, X_morgan.shape[1] - 1)
        if max_svd_dim < 2:
            raise RuntimeError(f"SVD dimension became invalid for split {split_seed}: {max_svd_dim}")

        svd = TruncatedSVD(n_components=max_svd_dim, random_state=split_seed)
        svd.fit(X_morgan[train_idx])
        X_svd = svd.transform(X_morgan).astype(np.float32)
        svd_ev = float(svd.explained_variance_ratio_.sum())

        split_audit_rows.append({
            "split_seed": int(split_seed),
            "n_train": int(len(train_idx)),
            "n_val": int(len(val_idx)),
            "n_test_frozen_not_evaluated": int(len(test_idx)),
            "svd_dim": int(max_svd_dim),
            "svd_explained_variance_ratio_train_fit": svd_ev,
        })

        feature_sets = {
            "Morgan2048": X_morgan,
            f"SVD{max_svd_dim}": X_svd,
        }

        # SELF-only: no K dependency, one run per node representation.
        empty_ei_train = torch.empty((2, 0), dtype=torch.long)
        empty_ew_train = torch.empty((0,), dtype=torch.float32)
        empty_ei_val = torch.empty((2, 0), dtype=torch.long)
        empty_ew_val = torch.empty((0,), dtype=torch.float32)

        for node_repr, X_use in feature_sets.items():
            result, pred_df = fit_one_run(
                X_all=X_use,
                y_all=y_all,
                train_idx=train_idx,
                val_idx=val_idx,
                train_edges=(empty_ei_train, empty_ew_train),
                val_edges=(empty_ei_val, empty_ew_val),
                mode="self",
                args=args,
                device=device,
            )
            row = {
                "split_seed": int(split_seed),
                "node_repr": node_repr,
                "relation_mode": "self",
                "topk": np.nan,
                "n_train": int(len(train_idx)),
                "n_val": int(len(val_idx)),
                "test_evaluated": False,
                "svd_explained_variance_ratio_train_fit": svd_ev if node_repr.startswith("SVD") else np.nan,
                **result,
            }
            run_rows.append(row)
            pred_df["split_seed"] = split_seed
            pred_df["node_repr"] = node_repr
            pred_df["relation_mode"] = "self"
            pred_df["topk"] = np.nan
            prediction_rows.append(pred_df)

        # Graph conditions.
        for k in sorted(set(args.topk)):
            if k <= 0:
                raise ValueError("topk must be positive")
            effective_k = min(int(k), max(len(train_idx) - 1, 1))

            tr_ei, tr_ew, tr_qc = build_inductive_edges(
                train_global_idx=train_idx,
                query_global_idx=np.array([], dtype=int),
                fps=fps,
                topk=effective_k,
                min_similarity=args.min_similarity,
            )
            va_ei, va_ew, va_qc = build_inductive_edges(
                train_global_idx=train_idx,
                query_global_idx=val_idx,
                fps=fps,
                topk=effective_k,
                min_similarity=args.min_similarity,
            )

            graph_qc_rows.append({
                "split_seed": int(split_seed),
                "requested_topk": int(k),
                "effective_topk": int(effective_k),
                "n_train": int(len(train_idx)),
                "n_val": int(len(val_idx)),
                **{f"train_graph_{kk}": vv for kk, vv in tr_qc.items()},
                **{f"val_graph_{kk}": vv for kk, vv in va_qc.items()},
            })

            for node_repr, X_use in feature_sets.items():
                for mode in ["unweighted", "weighted"]:
                    result, pred_df = fit_one_run(
                        X_all=X_use,
                        y_all=y_all,
                        train_idx=train_idx,
                        val_idx=val_idx,
                        train_edges=(tr_ei, tr_ew),
                        val_edges=(va_ei, va_ew),
                        mode=mode,
                        args=args,
                        device=device,
                    )
                    row = {
                        "split_seed": int(split_seed),
                        "node_repr": node_repr,
                        "relation_mode": mode,
                        "topk": int(effective_k),
                        "n_train": int(len(train_idx)),
                        "n_val": int(len(val_idx)),
                        "test_evaluated": False,
                        "svd_explained_variance_ratio_train_fit": svd_ev if node_repr.startswith("SVD") else np.nan,
                        **result,
                    }
                    run_rows.append(row)
                    pred_df["split_seed"] = split_seed
                    pred_df["node_repr"] = node_repr
                    pred_df["relation_mode"] = mode
                    pred_df["topk"] = int(effective_k)
                    prediction_rows.append(pred_df)

        print(f"[split {split_seed}] completed", flush=True)

    run_df = pd.DataFrame(run_rows)
    summary_df = make_summary(run_df)
    delta_df = make_pairwise_deltas(run_df)
    graph_qc_df = pd.DataFrame(graph_qc_rows)
    topology_df = pd.DataFrame(topology_rows)
    pred_all_df = pd.concat(prediction_rows, ignore_index=True) if prediction_rows else pd.DataFrame()
    split_audit_df = pd.DataFrame(split_audit_rows)

    topology_path = out_dir / "00_topology_sweep.csv"
    run_path = out_dir / "01_validation_runs.csv"
    summary_path = out_dir / "02_validation_summary.csv"
    delta_path = out_dir / "03_pairwise_deltas.csv"
    graph_qc_path = out_dir / "04_graph_qc.csv"
    pred_path = out_dir / "05_validation_predictions.csv"
    split_audit_path = out_dir / "06_split_feature_audit.csv"

    topology_df.to_csv(topology_path, index=False, encoding="utf-8-sig")
    run_df.to_csv(run_path, index=False, encoding="utf-8-sig")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    delta_df.to_csv(delta_path, index=False, encoding="utf-8-sig")
    graph_qc_df.to_csv(graph_qc_path, index=False, encoding="utf-8-sig")
    pred_all_df.to_csv(pred_path, index=False, encoding="utf-8-sig")
    split_audit_df.to_csv(split_audit_path, index=False, encoding="utf-8-sig")

    meta = {
        "input_table": str(input_table),
        "split_root": str(split_root),
        "smiles_col": args.smiles_col,
        "y_col": args.y_col,
        "n_rows": int(len(df)),
        "split_seeds": [int(x) for x in args.split_seeds],
        "morgan": {
            "type": "count",
            "radius": int(args.radius),
            "n_bits": int(args.n_bits),
            "include_chirality": False,
        },
        "topk_model_audit": [int(x) for x in args.topk],
        "topk_topology_audit": [int(x) for x in args.topology_k],
        "min_similarity_strict_gt": float(args.min_similarity),
        "graph_policy": "training graph = undirected union of Top-K Tanimoto selections (max score on duplicate pair), expanded both directions for message passing; validation receives TRAIN->VAL edges only; no VAL->TRAIN or VAL->VAL edges",
        "weighted_policy": "normalized Tanimoto-weighted neighbor mean",
        "svd_policy": "TruncatedSVD fit on training Morgan X only per split",
        "test_policy": "frozen test partition is not evaluated by this audit script",
        "model": {
            "family": "controlled GraphSAGE-style self + neighbor aggregation",
            "hidden_dim": int(args.hidden_dim),
            "num_layers": int(args.num_layers),
            "dropout": float(args.dropout),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "epochs_max": int(args.epochs),
            "patience": int(args.patience),
            "model_seed": int(args.model_seed),
        },
        "device": str(device),
    }
    (out_dir / "P05A_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "[P05A G2 TANIMOTO / GRAPHSAGE DESIGN AUDIT]",
        "",
        f"Input chemicals                 : {len(df)}",
        f"Split seeds                     : {args.split_seeds}",
        f"Count Morgan                    : radius={args.radius}, n_bits={args.n_bits}, chirality=False",
        f"Top-K model audit               : {sorted(set(args.topk))}",
        f"Top-K topology-only audit       : {sorted(set(args.topology_k))}",
        f"SVD                             : train-fit, target_dim={args.svd_dim}",
        f"Test partition evaluated        : NO",
        f"Graph policy                    : undirected Top-K union TRAIN graph; TRAIN->VAL query edges only",
        "",
        "[Topology audit: label-free]",
        topology_df.to_string(index=False),
        "",
        "[Validation summary: lower RMSE is better]",
        summary_df.to_string(index=False),
        "",
        "[Pairwise RMSE deltas: negative means first-named graph condition is better]",
    ]

    if len(delta_df):
        delta_summary = (
            delta_df.groupby(["node_repr", "topk", "comparison"], as_index=False)["delta_val_RMSE"]
            .agg(["mean", "std", "median"])
            .reset_index()
        )
        lines.append(delta_summary.to_string(index=False))
    else:
        lines.append("No pairwise deltas generated.")

    summary_txt = "\n".join(lines)
    (out_dir / "P05A_summary.txt").write_text(summary_txt, encoding="utf-8")

    print(summary_txt)
    print(f"\n[DONE] {out_dir}")


if __name__ == "__main__":
    main()
