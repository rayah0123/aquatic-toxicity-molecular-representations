#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P05G_00_g1_core.py
v1.1 hotfix

Single source of truth for G1:
- molecular graph schema
- atom/bond feature construction
- graph validation
- GraphSAGE model definition
- schema hashing / asset verification helpers

HOTFIX v1.1
-----------
PyG Data may expose data.y as None even when no target was assigned.
Therefore unlabeled-asset QC must check `getattr(data, "y", None) is not None`
rather than `hasattr(data, "y")`.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rdkit import Chem, rdBase
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv, global_mean_pool


SCHEMA_VERSION = "G1_GRAPH_SCHEMA_V1_1"
MODEL_FAMILY = "GraphSAGE"

ATOM_FEATURE_NAMES = [
    "atomic_number",
    "degree",
    "formal_charge",
    "is_aromatic",
    "total_num_H",
    "is_in_ring",
    "implicit_valence",
    "explicit_valence",
    "total_valence",
    "chiral_tag",
]

EDGE_FEATURE_NAMES = [
    "bond_single",
    "bond_double",
    "bond_triple",
    "bond_aromatic",
    "is_conjugated",
]

ATOM_FEATURE_DIM = len(ATOM_FEATURE_NAMES)
EDGE_FEATURE_DIM = len(EDGE_FEATURE_NAMES)

BOND_MAP = {
    Chem.rdchem.BondType.SINGLE: 0,
    Chem.rdchem.BondType.DOUBLE: 1,
    Chem.rdchem.BondType.TRIPLE: 2,
    Chem.rdchem.BondType.AROMATIC: 3,
}


def _implicit_valence(atom) -> int:
    if hasattr(atom, "GetValence") and hasattr(Chem, "ValenceType"):
        return int(atom.GetValence(Chem.ValenceType.IMPLICIT))
    return int(atom.GetImplicitValence())


def _explicit_valence(atom) -> int:
    if hasattr(atom, "GetValence") and hasattr(Chem, "ValenceType"):
        return int(atom.GetValence(Chem.ValenceType.EXPLICIT))
    return int(atom.GetExplicitValence())


def atom_features(atom) -> List[float]:
    return [
        float(atom.GetAtomicNum()),
        float(atom.GetDegree()),
        float(atom.GetFormalCharge()),
        float(int(atom.GetIsAromatic())),
        float(atom.GetTotalNumHs()),
        float(int(atom.IsInRing())),
        float(_implicit_valence(atom)),
        float(_explicit_valence(atom)),
        float(atom.GetTotalValence()),
        float(int(atom.GetChiralTag())),
    ]


def bond_features(bond) -> List[float]:
    bt = bond.GetBondType()
    onehot = [0.0, 0.0, 0.0, 0.0]
    onehot[BOND_MAP.get(bt, 0)] = 1.0
    onehot.append(float(int(bond.GetIsConjugated())))
    return onehot


def graph_schema_spec() -> Dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "representation": "intramolecular_atom_bond_graph",
        "main_model": MODEL_FAMILY,
        "node_features": {
            "dim": ATOM_FEATURE_DIM,
            "names": ATOM_FEATURE_NAMES,
            "dtype": "float32",
            "encoding_note": "Legacy scalar atom-feature semantics preserved.",
        },
        "edge_index": {
            "policy": "each RDKit bond stored in both directions",
            "main_graphsage_usage": "connectivity_only",
        },
        "edge_attr": {
            "stored": True,
            "used_by_main_graphsage": False,
            "dim": EDGE_FEATURE_DIM,
            "names": EDGE_FEATURE_NAMES,
            "purpose": "provenance/future edge-aware sensitivity analyses only",
        },
        "pooling": "global_mean_pool",
        "rdkit_version": rdBase.rdkitVersion,
    }


def canonical_json_hash(obj: Dict) -> str:
    txt = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(txt.encode("utf-8")).hexdigest()


def graph_schema_hash() -> str:
    return canonical_json_hash(graph_schema_spec())


def smiles_sequence_hash(smiles: Sequence[str]) -> str:
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


def mol_to_pyg(smiles: str, row_idx: int | None = None) -> Data | None:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None

    x = np.asarray([atom_features(a) for a in mol.GetAtoms()], dtype=np.float32)
    if x.ndim == 1:
        x = x.reshape(-1, ATOM_FEATURE_DIM)

    edges = []
    edge_attrs = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bf = bond_features(bond)
        edges.extend([[i, j], [j, i]])
        edge_attrs.extend([bf, bf])

    if edges:
        edge_index = torch.tensor(np.asarray(edges).T, dtype=torch.long)
        edge_attr = torch.tensor(np.asarray(edge_attrs), dtype=torch.float32)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, EDGE_FEATURE_DIM), dtype=torch.float32)

    data = Data(
        x=torch.tensor(x, dtype=torch.float32),
        edge_index=edge_index,
        edge_attr=edge_attr,
    )
    data.smiles = str(smiles)
    if row_idx is not None:
        data.row_idx = torch.tensor([int(row_idx)], dtype=torch.long)

    validate_single_graph(data, row_idx=row_idx, smiles=str(smiles))
    return data


def validate_single_graph(data: Data, row_idx=None, smiles=None) -> Dict:
    if data is None:
        raise ValueError(f"Graph is None | row={row_idx} | smiles={smiles}")
    if data.x is None or data.edge_index is None:
        raise ValueError(f"Missing x/edge_index | row={row_idx} | smiles={smiles}")

    if data.x.ndim != 2 or data.x.size(1) != ATOM_FEATURE_DIM:
        raise ValueError(
            f"Unexpected x shape={tuple(data.x.shape)}; expected (*,{ATOM_FEATURE_DIM})"
        )

    n_nodes = int(data.x.size(0))
    if n_nodes <= 0:
        raise ValueError(f"Zero-node graph | row={row_idx} | smiles={smiles}")

    if data.edge_index.ndim != 2 or data.edge_index.size(0) != 2:
        raise ValueError(f"Invalid edge_index shape={tuple(data.edge_index.shape)}")

    if data.edge_index.numel():
        lo = int(data.edge_index.min().item())
        hi = int(data.edge_index.max().item())
        if lo < 0 or hi >= n_nodes:
            raise ValueError(
                f"edge_index out of bounds [{lo},{hi}] for n_nodes={n_nodes}"
            )

    edge_attr = getattr(data, "edge_attr", None)
    if edge_attr is None:
        raise ValueError("edge_attr missing from provenance graph asset")
    if edge_attr.ndim != 2 or edge_attr.size(1) != EDGE_FEATURE_DIM:
        raise ValueError(
            f"Unexpected edge_attr shape={tuple(edge_attr.shape)}; "
            f"expected (*,{EDGE_FEATURE_DIM})"
        )
    if edge_attr.size(0) != data.edge_index.size(1):
        raise ValueError("edge_attr rows do not match edge_index edge count")

    # HOTFIX: PyG Data can expose y=None even when no label was assigned.
    if getattr(data, "y", None) is not None:
        raise ValueError(
            "Graph asset must be unlabeled. Found non-None data.y; "
            "graph and target must remain separate."
        )

    return {
        "global_row_index": int(row_idx) if row_idx is not None else None,
        "smiles": str(smiles) if smiles is not None else getattr(data, "smiles", None),
        "n_nodes": n_nodes,
        "n_directed_edges": int(data.edge_index.size(1)),
        "n_undirected_bonds": int(data.edge_index.size(1) // 2),
    }


class G1GraphSAGE(nn.Module):
    def __init__(
        self,
        in_dim: int = ATOM_FEATURE_DIM,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.30,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        self.in_dim = int(in_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.convs.append(SAGEConv(self.in_dim, self.hidden_dim))
        self.bns.append(nn.BatchNorm1d(self.hidden_dim))
        for _ in range(self.num_layers - 1):
            self.convs.append(SAGEConv(self.hidden_dim, self.hidden_dim))
            self.bns.append(nn.BatchNorm1d(self.hidden_dim))
        self.head = nn.Linear(self.hidden_dim, 1)

    def forward(self, data):
        x = data.x.float()
        edge_index = data.edge_index
        batch = data.batch

        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        g = global_mean_pool(x, batch)
        return self.head(g).view(-1)


def model_spec(hidden_dim=128, num_layers=2, dropout=0.30) -> Dict:
    return {
        "model_family": MODEL_FAMILY,
        "representation": "G1_intramolecular_graph",
        "in_dim": ATOM_FEATURE_DIM,
        "hidden_dim": int(hidden_dim),
        "num_layers": int(num_layers),
        "dropout": float(dropout),
        "pooling": "global_mean_pool",
        "uses_edge_attr": False,
    }


def assert_bundle_compatible(bundle: Dict, graph_asset_meta: Dict):
    expected = graph_schema_hash()
    bundle_hash = bundle.get("graph_schema_hash")
    asset_hash = graph_asset_meta.get("graph_schema_hash")

    if bundle_hash != expected:
        raise RuntimeError(
            f"Bundle graph schema mismatch: bundle={bundle_hash}, current={expected}"
        )
    if asset_hash != expected:
        raise RuntimeError(
            f"Graph asset schema mismatch: asset={asset_hash}, current={expected}"
        )
    if bundle_hash != asset_hash:
        raise RuntimeError(
            f"Bundle/asset schema mismatch: bundle={bundle_hash}, asset={asset_hash}"
        )

    ms = bundle.get("model_spec", {})
    if int(ms.get("in_dim", -1)) != ATOM_FEATURE_DIM:
        raise RuntimeError(
            f"Bundle expects in_dim={ms.get('in_dim')}, current schema={ATOM_FEATURE_DIM}"
        )
    if ms.get("uses_edge_attr") not in {False, 0}:
        raise RuntimeError("Main G1 bundle unexpectedly declares edge_attr usage.")
