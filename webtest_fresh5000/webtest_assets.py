from __future__ import annotations
from pathlib import Path
from typing import Dict,List,Tuple
import random
import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import Descriptors
COMPACT6=['MolLogP','MolWt','TPSA','NumHAcceptors','NumHDonors','NumRotatableBonds']
FUNCS={name:getattr(Descriptors,name) for name in COMPACT6}
SPLIT_ROOT=None
GRAPH_ROOT=None
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
            print(f"[Compact6] {i}/{len(smiles)}", flush=True)
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

    P10C query assets store:
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
