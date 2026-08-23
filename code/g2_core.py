#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
g2_core.py

Single source of truth for the FROZEN G2 specification.

Frozen scientific specification
-------------------------------
Node X:
  MolLogP, MolWt, TPSA, NumHAcceptors, NumHDonors, NumRotatableBonds

Edge topology:
  Count Morgan radius=4, fpSize=2048, includeChirality=False
  -> Tanimoto
  -> Top-K = 10
  -> training graph = undirected union of directed Top-K selections
  -> duplicate pair similarity reduced by max
  -> expanded both directions for message passing
  -> query graph = TRAIN -> QUERY only
     (no QUERY -> TRAIN, no QUERY -> QUERY)

Model:
  Controlled GraphSAGE-style mean aggregation, UNWEIGHTED
  hidden=128, layers=2, dropout=0.3

This module is the single source of truth for the frozen G2 scientific specification.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


SCHEMA_VERSION = "G2_COMPACT6_TANIMOTO_K10_V1_0"
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

RADIUS = 4
N_BITS = 2048
TOPK = 10
MIN_SIMILARITY = 0.0


def seed_all(seed: int):
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def spec() -> Dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "node_X": {
            "descriptors": COMPACT6,
            "dimension": 6,
            "preprocessing": "TRAIN-only median imputation, zero-variance guard, z-score",
        },
        "edge": {
            "fingerprint": "Count Morgan",
            "radius": RADIUS,
            "n_bits": N_BITS,
            "include_chirality": False,
            "similarity": "Tanimoto",
            "topk": TOPK,
            "min_similarity_strict_gt": MIN_SIMILARITY,
            "train_graph": "undirected union of directed Top-K selections; duplicate pair=max; expand both directions",
            "query_graph": "TRAIN -> QUERY only; no QUERY -> TRAIN; no QUERY -> QUERY",
            "message_weighting": "unweighted",
        },
        "model": {
            "family": "Controlled GraphSAGE-style",
            "aggregation": "mean",
            "hidden_dim": 128,
            "num_layers": 2,
            "dropout": 0.30,
        },
        "rdkit_version": rdBase.rdkitVersion,
    }


def canonical_json_hash(obj: Dict) -> str:
    txt = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(txt.encode("utf-8")).hexdigest()


def schema_hash() -> str:
    return canonical_json_hash(spec())


def smiles_hash(smiles: Sequence[str]) -> str:
    h = hashlib.sha256()
    for i, smi in enumerate(smiles):
        h.update(f"{i}\t{smi}\n".encode("utf-8"))
    return h.hexdigest()


def file_sha256(path: str | Path) -> str:
    path = Path(path)
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build_compact6_raw(smiles: Sequence[str]) -> np.ndarray:
    rows = []
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            raise ValueError(f"Invalid SMILES at row {i}: {smi}")
        vals = []
        for name in COMPACT6:
            v = float(FUNCS[name](mol))
            if not np.isfinite(v):
                raise ValueError(f"Non-finite {name} at row {i}: {smi}")
            vals.append(v)
        rows.append(vals)
    return np.asarray(rows, dtype=np.float64)


def build_count_morgan(smiles: Sequence[str]):
    gen = rdFingerprintGenerator.GetMorganGenerator(
        radius=RADIUS,
        fpSize=N_BITS,
        includeChirality=False,
    )
    fps = []
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            raise ValueError(f"Invalid SMILES at row {i}: {smi}")
        fps.append(gen.GetCountFingerprint(mol))
    return fps


def fit_preprocessor_train_only(
    X_raw: np.ndarray,
    train_idx: np.ndarray,
) -> Dict:
    Xtr = np.asarray(X_raw[train_idx], dtype=np.float64)

    finite_any = np.isfinite(Xtr).any(axis=0)
    if not finite_any.all():
        raise RuntimeError(
            f"Frozen six-descriptor set contains TRAIN-all-missing descriptor(s): "
            f"{[COMPACT6[i] for i in np.where(~finite_any)[0]]}"
        )

    med = np.nanmedian(Xtr, axis=0)
    if not np.isfinite(med).all():
        raise RuntimeError("Non-finite TRAIN median in frozen six-descriptor set.")

    Ximp = Xtr.copy()
    bad = ~np.isfinite(Ximp)
    if bad.any():
        rr, cc = np.where(bad)
        Ximp[rr, cc] = med[cc]

    var = np.var(Ximp, axis=0)
    if np.any(~np.isfinite(var) | (var <= 0)):
        bad_names = [
            COMPACT6[i] for i in np.where(~np.isfinite(var) | (var <= 0))[0]
        ]
        raise RuntimeError(
            f"Frozen six-descriptor set would change due to zero variance: {bad_names}"
        )

    mean = np.mean(Ximp, axis=0)
    std = np.std(Ximp, axis=0, ddof=0)
    if np.any(~np.isfinite(std) | (std <= 0)):
        raise RuntimeError("Invalid TRAIN std in frozen six-descriptor set.")

    return {
        "names": list(COMPACT6),
        "medians": med.astype(float),
        "mean": mean.astype(float),
        "std": std.astype(float),
    }


def transform_rows(X_raw: np.ndarray, idx: np.ndarray, pp: Dict) -> np.ndarray:
    if list(pp["names"]) != COMPACT6:
        raise RuntimeError("Preprocessor descriptor names do not match frozen six-descriptor set.")
    X = np.asarray(X_raw[idx], dtype=np.float64)
    med = np.asarray(pp["medians"], dtype=float)
    mean = np.asarray(pp["mean"], dtype=float)
    std = np.asarray(pp["std"], dtype=float)

    bad = ~np.isfinite(X)
    if bad.any():
        X = X.copy()
        rr, cc = np.where(bad)
        X[rr, cc] = med[cc]
    X = (X - mean) / std
    if not np.isfinite(X).all():
        raise RuntimeError("Non-finite values remain after six-descriptor set transform.")
    return X.astype(np.float32, copy=False)


def select_topk_train_neighbors(
    target_global_idx: int,
    train_global_idx: np.ndarray,
    fps: Sequence,
    topk: int = TOPK,
    min_similarity: float = MIN_SIMILARITY,
    exclude_self: bool = False,
) -> List[Tuple[int, float]]:
    target_fp = fps[int(target_global_idx)]
    train_fps = [fps[int(i)] for i in train_global_idx]
    sims = DataStructs.BulkTanimotoSimilarity(target_fp, train_fps)

    candidates = []
    for train_pos, (global_idx, sim) in enumerate(zip(train_global_idx, sims)):
        if exclude_self and int(global_idx) == int(target_global_idx):
            continue
        sim = float(sim)
        if not np.isfinite(sim) or sim <= float(min_similarity):
            continue
        candidates.append((int(train_pos), sim))

    candidates.sort(key=lambda x: (-x[1], x[0]))
    return candidates[: int(topk)]


def _component_stats(
    n_nodes: int,
    undirected_pairs: List[Tuple[int, int, float]],
) -> Dict:
    parent = list(range(n_nodes))
    rank = [0] * n_nodes

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
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

    comp = {}
    for i in range(n_nodes):
        r = find(i)
        comp[r] = comp.get(r, 0) + 1
    sizes = sorted(comp.values(), reverse=True)
    largest = sizes[0] if sizes else 0

    return {
        "n_components": int(len(sizes)),
        "largest_component_size": int(largest),
        "LCC_ratio": float(largest / n_nodes) if n_nodes else np.nan,
        "isolates": int((degree == 0).sum()),
        "degree_min": int(degree.min()) if n_nodes else 0,
        "degree_mean": float(degree.mean()) if n_nodes else np.nan,
        "degree_median": float(np.median(degree)) if n_nodes else np.nan,
        "degree_max": int(degree.max()) if n_nodes else 0,
    }


def build_train_undirected_graph(
    train_global_idx: np.ndarray,
    fps: Sequence,
    topk: int = TOPK,
    min_similarity: float = MIN_SIMILARITY,
):
    pair_score: Dict[Tuple[int, int], float] = {}
    selected_sims = []

    for target_local, target_global in enumerate(train_global_idx):
        nbrs = select_topk_train_neighbors(
            int(target_global),
            train_global_idx,
            fps,
            topk=topk,
            min_similarity=min_similarity,
            exclude_self=True,
        )
        for nbr_local, sim in nbrs:
            u, v = sorted((int(target_local), int(nbr_local)))
            if u == v:
                continue
            pair_score[(u, v)] = max(
                float(sim), pair_score.get((u, v), -np.inf)
            )
            selected_sims.append(float(sim))

    pairs = [
        (u, v, float(score))
        for (u, v), score in sorted(pair_score.items())
    ]

    src, dst, weights = [], [], []
    pair_sims = []
    for u, v, score in pairs:
        src.extend([u, v])
        dst.extend([v, u])
        weights.extend([score, score])
        pair_sims.append(score)

    if src:
        ei = torch.tensor([src, dst], dtype=torch.long)
        ew = torch.tensor(weights, dtype=torch.float32)
    else:
        ei = torch.empty((2, 0), dtype=torch.long)
        ew = torch.empty((0,), dtype=torch.float32)

    arr = np.asarray(pair_sims, dtype=float)
    qc = {
        "n_train": int(len(train_global_idx)),
        "topk": int(topk),
        "directed_topk_selection_n": int(len(selected_sims)),
        "unique_edge_n": int(len(pairs)),
        "message_edge_n": int(ei.shape[1]),
        "sim_q05": float(np.quantile(arr, 0.05)) if arr.size else np.nan,
        "sim_median": float(np.median(arr)) if arr.size else np.nan,
        "sim_mean": float(np.mean(arr)) if arr.size else np.nan,
        "sim_max": float(np.max(arr)) if arr.size else np.nan,
        **_component_stats(len(train_global_idx), pairs),
    }
    return ei, ew, qc, pairs


def build_query_edges(
    train_global_idx: np.ndarray,
    query_global_idx: np.ndarray,
    fps: Sequence,
    topk: int = TOPK,
    min_similarity: float = MIN_SIMILARITY,
):
    src, dst, weights = [], [], []
    sims_all = []
    degrees = []
    n_train = len(train_global_idx)

    for q_pos, q_global in enumerate(query_global_idx):
        target_local = n_train + q_pos
        nbrs = select_topk_train_neighbors(
            int(q_global),
            train_global_idx,
            fps,
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

    arr = np.asarray(sims_all, dtype=float)
    qc = {
        "n_query": int(len(query_global_idx)),
        "query_message_edge_n": int(len(src)),
        "query_neighbor_count_min": int(np.min(degrees)) if degrees else 0,
        "query_neighbor_count_mean": float(np.mean(degrees)) if degrees else 0.0,
        "query_neighbor_count_max": int(np.max(degrees)) if degrees else 0,
        "query_zero_neighbor_n": int(sum(d == 0 for d in degrees)),
        "query_sim_q05": float(np.quantile(arr, 0.05)) if arr.size else np.nan,
        "query_sim_median": float(np.median(arr)) if arr.size else np.nan,
        "query_sim_mean": float(np.mean(arr)) if arr.size else np.nan,
        "query_sim_max": float(np.max(arr)) if arr.size else np.nan,
    }
    return ei, ew, qc


def build_inductive_edges(
    train_global_idx: np.ndarray,
    query_global_idx: np.ndarray,
    fps: Sequence,
    topk: int = TOPK,
    min_similarity: float = MIN_SIMILARITY,
):
    tr_ei, tr_ew, tr_qc, _ = build_train_undirected_graph(
        train_global_idx, fps, topk, min_similarity
    )
    q_ei, q_ew, q_qc = build_query_edges(
        train_global_idx, query_global_idx, fps, topk, min_similarity
    )

    if tr_ei.numel() == 0:
        ei, ew = q_ei, q_ew
    elif q_ei.numel() == 0:
        ei, ew = tr_ei, tr_ew
    else:
        ei = torch.cat([tr_ei, q_ei], dim=1)
        ew = torch.cat([tr_ew, q_ew], dim=0)

    return ei, ew, {**tr_qc, **q_qc}


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
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.lin_self = nn.Linear(in_dim, out_dim, bias=True)
        self.lin_neigh = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x, edge_index, edge_weight, mode):
        out = self.lin_self(x)
        if mode == "self":
            return out
        if mode not in {"unweighted", "weighted"}:
            raise ValueError(f"Unsupported mode: {mode}")
        neigh = normalized_neighbor_mean(
            x, edge_index, edge_weight,
            weighted=(mode == "weighted"),
        )
        return out + self.lin_neigh(neigh)


class ControlledGraphSAGE(nn.Module):
    def __init__(
        self,
        input_dim: int = 6,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.30,
        mode: str = "unweighted",
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

    def forward(self, x, edge_index, edge_weight):
        h = x
        for layer, bn in zip(self.layers, self.bns):
            h = layer(h, edge_index, edge_weight, self.mode)
            h = bn(h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        return self.head(h).view(-1)


def regression_metrics(y_true, y_pred) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
        "Pearson": float(pd.Series(y_true).corr(pd.Series(y_pred), method="pearson")),
        "Spearman": float(pd.Series(y_true).corr(pd.Series(y_pred), method="spearman")),
    }
