#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Construct the exact, strictly inductive WebTEST Top-10 Tanimoto graph.

Count Morgan: radius 4, 2048 dimensions, no chirality; similarity > 0.
TRAIN: undirected union of each compound's Top-10 TRAIN selections,
with duplicate similarities combined by maximum.
VAL/TEST: one-way TRAIN-to-query edges only.
Reuse exact Top-50 reference calculations when they guarantee correct Top-10
recovery; otherwise recompute exact similarities to the TRAIN bank.
Save graph/topology/neighborhood QC and cache/fallback provenance.
No model fitting."""

from pathlib import Path
import os
import json
import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger, rdBase
from rdkit.Chem import rdFingerprintGenerator

RDLogger.DisableLog("rdApp.warning")

ROOT = Path(__file__).resolve().parents[1]
# Optional historical exact Top-50 cache source; configure without hard-coded machine paths.
LEGACY_ROOT = Path(os.environ.get("TOXICS_LEGACY_ROOT", str(ROOT / "legacy_cache"))).expanduser().resolve()

CURRENT = (
    ROOT / "03_processed" / "WebTEST"
    / "WebTEST_Daphnia_consensus_unique_connectivity_frozen.csv"
)
SPLIT_ROOT = (
    ROOT / "04_splits" / "WebTEST_Daphnia"
    / "scaffold_split_murcko_topology_refined"
)

LEGACY_TABLE = (
    LEGACY_ROOT / "output" / "00_subsets"
    / "toxicity_master__subset_daphnia.csv"
)
LEGACY_EDGES = (
    LEGACY_ROOT / "output" / "19_g2_raw_edges" / "daphnia"
    / "g2_daphnia__raw_directed.parquet"
)
LEGACY_QC = (
    LEGACY_ROOT / "output" / "19_g2_raw_edges" / "daphnia"
    / "g2_daphnia__raw_directed__qc.json"
)

OUT = (
    ROOT / "07_results" / "WebTEST_revised"
    / "P10C_exact_Top10_graph_assets"
)

SEEDS = [42, 123, 777, 2025, 3407]
N_CURRENT = 119276
N_LEGACY = 153174
K = 10
RADIUS = 4
N_BITS = 2048
MIN_SIM = 0.0
SMILES_COL = "SMILES_RDKit"


def load_split(seed: int, n: int) -> Dict[str, np.ndarray]:
    p = SPLIT_ROOT / f"split_seed{seed}" / "split_membership.csv"
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


def build_count_fps(smiles: List[str]):
    gen = rdFingerprintGenerator.GetMorganGenerator(
        radius=RADIUS,
        fpSize=N_BITS,
        includeChirality=False,
    )
    fps = []
    for i, smi in enumerate(smiles):
        if i % 10000 == 0:
            print(f"  fingerprints {i}/{len(smiles)}", flush=True)
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            raise ValueError(f"Invalid SMILES row={i}: {smi}")
        fps.append(gen.GetCountFingerprint(mol))
    return fps


def exact_topk_against_train(
    source_global: int,
    train_idx: np.ndarray,
    train_fps: List,
    fps_all: List,
    exclude_self: bool,
) -> List[Tuple[int, float]]:
    sims = DataStructs.BulkTanimotoSimilarity(
        fps_all[int(source_global)],
        train_fps,
    )
    candidates = []
    for pos, (gidx, sim) in enumerate(zip(train_idx, sims)):
        gidx = int(gidx)
        if exclude_self and gidx == int(source_global):
            continue
        sim = float(sim)
        if not np.isfinite(sim) or sim <= MIN_SIM:
            continue
        # Current frozen implementation tie-break:
        # train_pos == ascending current global row index because train_idx is sorted.
        candidates.append((gidx, sim))

    candidates.sort(key=lambda x: (-x[1], x[0]))
    return candidates[:K]


def component_stats(
    train_idx: np.ndarray,
    undirected_df: pd.DataFrame,
) -> Dict:
    n = len(train_idx)
    global_to_local = {int(g): i for i, g in enumerate(train_idx)}

    parent = np.arange(n, dtype=np.int64)
    rank = np.zeros(n, dtype=np.int8)
    degree = np.zeros(n, dtype=np.int32)

    def find(a):
        a = int(a)
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = int(parent[a])
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

    for r in undirected_df.itertuples(index=False):
        u = global_to_local[int(r.u_global)]
        v = global_to_local[int(r.v_global)]
        if u == v:
            continue
        union(u, v)
        degree[u] += 1
        degree[v] += 1

    comp = {}
    for i in range(n):
        root = find(i)
        comp[root] = comp.get(root, 0) + 1

    sizes = sorted(comp.values(), reverse=True)
    largest = sizes[0] if sizes else 0
    second = sizes[1] if len(sizes) > 1 else 0

    return {
        "n_components": int(len(sizes)),
        "largest_component_size": int(largest),
        "second_largest_component_size": int(second),
        "LCC_ratio": float(largest / n) if n else np.nan,
        "isolates": int((degree == 0).sum()),
        "degree_min": int(degree.min()) if n else 0,
        "degree_q25": float(np.quantile(degree, .25)) if n else np.nan,
        "degree_median": float(np.median(degree)) if n else np.nan,
        "degree_mean": float(degree.mean()) if n else np.nan,
        "degree_q75": float(np.quantile(degree, .75)) if n else np.nan,
        "degree_max": int(degree.max()) if n else 0,
    }


def density_stats(directed: pd.DataFrame) -> Dict:
    if len(directed) == 0:
        return {
            "n_sources": 0,
            "n_selected_edges": 0,
            "neighbor_count_min": 0,
            "neighbor_count_mean": 0.0,
            "neighbor_count_max": 0,
            "zero_neighbor_sources": 0,
            "edge_sim_min": np.nan,
            "edge_sim_q25": np.nan,
            "edge_sim_median": np.nan,
            "edge_sim_q75": np.nan,
            "edge_sim_max": np.nan,
            "edge_sim_mean": np.nan,
            "source_mean_topk_sim_q25": np.nan,
            "source_mean_topk_sim_median": np.nan,
            "source_mean_topk_sim_q75": np.nan,
            "source_mean_topk_sim_mean": np.nan,
        }

    counts = directed.groupby("source_global_idx").size()
    source_mean = directed.groupby("source_global_idx")["similarity"].mean()
    a = directed["similarity"].to_numpy(dtype=float)

    return {
        "n_sources_with_edges": int(counts.size),
        "n_selected_edges": int(len(directed)),
        "neighbor_count_min": int(counts.min()),
        "neighbor_count_mean": float(counts.mean()),
        "neighbor_count_max": int(counts.max()),
        "edge_sim_min": float(np.min(a)),
        "edge_sim_q25": float(np.quantile(a, .25)),
        "edge_sim_median": float(np.median(a)),
        "edge_sim_q75": float(np.quantile(a, .75)),
        "edge_sim_max": float(np.max(a)),
        "edge_sim_mean": float(np.mean(a)),
        "source_mean_topk_sim_q25": float(source_mean.quantile(.25)),
        "source_mean_topk_sim_median": float(source_mean.median()),
        "source_mean_topk_sim_q75": float(source_mean.quantile(.75)),
        "source_mean_topk_sim_mean": float(source_mean.mean()),
    }


def main():
    t0 = time.time()
    for p in [CURRENT, LEGACY_TABLE, LEGACY_EDGES, LEGACY_QC]:
        if not p.exists():
            raise FileNotFoundError(p)

    OUT.mkdir(parents=True, exist_ok=True)

    cur = pd.read_csv(CURRENT, encoding="utf-8-sig", low_memory=False).reset_index(drop=True)
    old = pd.read_csv(LEGACY_TABLE, encoding="utf-8-sig", low_memory=False).reset_index(drop=True)

    if len(cur) != N_CURRENT:
        raise RuntimeError(f"Current N expected {N_CURRENT}, got {len(cur)}")
    if len(old) != N_LEGACY:
        raise RuntimeError(f"Legacy N expected {N_LEGACY}, got {len(old)}")
    if old[SMILES_COL].duplicated().any():
        raise RuntimeError("Legacy exact SMILES are not unique")

    qc = json.loads(LEGACY_QC.read_text(encoding="utf-8"))
    for k, v in {
        "n_nodes": N_LEGACY,
        "radius": RADIUS,
        "n_bits": N_BITS,
        "topk": 50,
    }.items():
        if int(qc.get(k, -1)) != int(v):
            raise RuntimeError(f"Legacy cache QC mismatch {k}: {qc.get(k)} != {v}")

    # exact-SMILES mapping to the legacy cache source universe
    old_map = pd.Series(
        np.arange(len(old), dtype=np.int64),
        index=old[SMILES_COL].astype(str),
    )
    cur_to_old = cur[SMILES_COL].astype(str).map(old_map)
    if cur_to_old.isna().any():
        raise RuntimeError(
            f"Current rows unmapped to legacy cache: {int(cur_to_old.isna().sum())}"
        )
    cur_to_old = cur_to_old.astype(np.int64).to_numpy()

    old_to_cur = np.full(N_LEGACY, -1, dtype=np.int64)
    old_to_cur[cur_to_old] = np.arange(N_CURRENT, dtype=np.int64)

    print("[1/5] Reading and validating legacy exact Top-50 cache ...", flush=True)
    e = pd.read_parquet(LEGACY_EDGES, columns=["src", "dst", "score"])
    e["src"] = pd.to_numeric(e["src"], errors="raise").astype(np.int64)
    e["dst"] = pd.to_numeric(e["dst"], errors="raise").astype(np.int64)
    e["score"] = pd.to_numeric(e["score"], errors="raise").astype(float)

    expected_edges = N_LEGACY * 50
    if len(e) != expected_edges:
        raise RuntimeError(
            f"Expected exactly {expected_edges} legacy Top50 rows, got {len(e)}"
        )

    # Old builder wrote src in ascending blocks of 50.
    block_ok = (
        np.array_equal(e["src"].to_numpy()[::50], np.arange(N_LEGACY))
        and np.all(
            e["src"].to_numpy().reshape(N_LEGACY, 50)
            == np.arange(N_LEGACY)[:, None]
        )
    )
    if not block_ok:
        print("  legacy rows not block-ordered; sorting once ...", flush=True)
        e = e.sort_values(
            ["src", "score", "dst"],
            ascending=[True, False, True],
            kind="stable",
        ).reset_index(drop=True)
        if not np.all(
            e["src"].to_numpy().reshape(N_LEGACY, 50)
            == np.arange(N_LEGACY)[:, None]
        ):
            raise RuntimeError("Could not establish exact 50-row source blocks")

    edge_dst_old = e["dst"].to_numpy(dtype=np.int64).reshape(N_LEGACY, 50)
    edge_score = e["score"].to_numpy(dtype=float).reshape(N_LEGACY, 50)
    del e

    print("[2/5] Building current Count-Morgan r4/2048 fingerprints once ...", flush=True)
    fps = build_count_fps(cur[SMILES_COL].astype(str).tolist())

    all_graph_qc = []
    all_density_qc = []
    all_prov = []

    print("[3/5] Building exact strict-inductive K10 assets ...", flush=True)

    for seed in SEEDS:
        seed_t0 = time.time()
        split = load_split(seed, N_CURRENT)
        train_idx = split["train"]
        train_mask = np.zeros(N_CURRENT, dtype=bool)
        train_mask[train_idx] = True
        train_fps = [fps[int(i)] for i in train_idx]

        seed_dir = OUT / f"split_seed{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        directed_parts = {}

        for part in ["train", "val", "test"]:
            recs = []
            n_cache = 0
            n_fallback = 0

            source_idx = split[part]
            for j, source_cur in enumerate(source_idx):
                if j % 10000 == 0:
                    print(
                        f"  seed={seed} part={part} {j}/{len(source_idx)}",
                        flush=True,
                    )

                source_cur = int(source_cur)
                source_old = int(cur_to_old[source_cur])

                dst_old_50 = edge_dst_old[source_old]
                sim_50 = edge_score[source_old]
                dst_cur_50 = old_to_cur[dst_old_50]

                eligible = []
                for dst_cur, sim in zip(dst_cur_50, sim_50):
                    dst_cur = int(dst_cur)
                    sim = float(sim)
                    if dst_cur < 0:
                        continue
                    if not train_mask[dst_cur]:
                        continue
                    if part == "train" and dst_cur == source_cur:
                        continue
                    if not np.isfinite(sim) or sim <= MIN_SIM:
                        continue
                    eligible.append((dst_cur, sim))

                # IMPORTANT: current frozen tie-break is current global row index.
                eligible.sort(key=lambda x: (-x[1], x[0]))

                boundary = float(np.min(sim_50))
                exactly_recoverable = (
                    len(eligible) >= K
                    and np.isfinite(boundary)
                    and float(eligible[K - 1][1]) > boundary + 1e-15
                )

                if exactly_recoverable:
                    nbrs = eligible[:K]
                    provenance = "legacy_exact_top50_guaranteed"
                    n_cache += 1
                else:
                    nbrs = exact_topk_against_train(
                        source_global=source_cur,
                        train_idx=train_idx,
                        train_fps=train_fps,
                        fps_all=fps,
                        exclude_self=(part == "train"),
                    )
                    provenance = "exact_current_train_fallback"
                    n_fallback += 1

                for rank, (nbr_cur, sim) in enumerate(nbrs, start=1):
                    recs.append({
                        "source_global_idx": source_cur,
                        "neighbor_train_global_idx": int(nbr_cur),
                        "rank": int(rank),
                        "similarity": float(sim),
                        "provenance": provenance,
                    })

            dpart = pd.DataFrame(recs)
            directed_parts[part] = dpart

            outp = seed_dir / f"{part}_source_to_train_exact_top10.parquet"
            dpart.to_parquet(outp, index=False)

            # compact NPZ for downstream model runner
            np.savez_compressed(
                seed_dir / f"{part}_source_to_train_exact_top10.npz",
                source_global_idx=dpart["source_global_idx"].to_numpy(dtype=np.int64),
                neighbor_train_global_idx=dpart["neighbor_train_global_idx"].to_numpy(dtype=np.int64),
                rank=dpart["rank"].to_numpy(dtype=np.int16),
                similarity=dpart["similarity"].to_numpy(dtype=np.float32),
            )

            expected_sources = len(source_idx)
            source_counts = dpart.groupby("source_global_idx").size()
            zero_sources = expected_sources - int(source_counts.size)

            dens = {
                "split_seed": int(seed),
                "partition": part,
                "n_expected_sources": int(expected_sources),
                "zero_neighbor_sources": int(zero_sources),
                "cache_recovered_sources": int(n_cache),
                "exact_fallback_sources": int(n_fallback),
                **density_stats(dpart),
            }
            all_density_qc.append(dens)
            all_prov.append({
                "split_seed": int(seed),
                "partition": part,
                "n_sources": int(expected_sources),
                "cache_recovered_sources": int(n_cache),
                "exact_fallback_sources": int(n_fallback),
                "cache_fraction": float(n_cache / expected_sources),
            })

        # TRAIN directed selections -> undirected UNION with max score.
        tr = directed_parts["train"].copy()
        uv = np.sort(
            tr[["source_global_idx", "neighbor_train_global_idx"]]
            .to_numpy(dtype=np.int64),
            axis=1,
        )
        tr["u_global"] = uv[:, 0]
        tr["v_global"] = uv[:, 1]

        undirected = (
            tr.groupby(["u_global", "v_global"], as_index=False)["similarity"]
            .max()
            .sort_values(["u_global", "v_global"])
            .reset_index(drop=True)
        )
        undirected.to_parquet(
            seed_dir / "train_undirected_union_exact_top10.parquet",
            index=False,
        )
        np.savez_compressed(
            seed_dir / "train_undirected_union_exact_top10.npz",
            u_global=undirected["u_global"].to_numpy(dtype=np.int64),
            v_global=undirected["v_global"].to_numpy(dtype=np.int64),
            similarity=undirected["similarity"].to_numpy(dtype=np.float32),
        )

        comp = component_stats(train_idx, undirected)
        train_density = next(
            x for x in all_density_qc
            if x["split_seed"] == seed and x["partition"] == "train"
        )

        gqc = {
            "split_seed": int(seed),
            "n_train": int(len(train_idx)),
            "K": K,
            "min_similarity_strict_gt": MIN_SIM,
            "directed_train_selection_edges": int(len(tr)),
            "undirected_union_edges": int(len(undirected)),
            "message_edges_after_bidirectional_expand": int(2 * len(undirected)),
            **comp,
            "selected_edge_sim_mean": float(tr["similarity"].mean()),
            "selected_edge_sim_median": float(tr["similarity"].median()),
            "selected_edge_sim_q25": float(tr["similarity"].quantile(.25)),
            "selected_edge_sim_q75": float(tr["similarity"].quantile(.75)),
            "source_mean_top10_sim_mean": train_density["source_mean_topk_sim_mean"],
            "source_mean_top10_sim_median": train_density["source_mean_topk_sim_median"],
            "source_mean_top10_sim_q25": train_density["source_mean_topk_sim_q25"],
            "source_mean_top10_sim_q75": train_density["source_mean_topk_sim_q75"],
        }
        all_graph_qc.append(gqc)

        print(
            f"  seed={seed} done | LCC={comp['LCC_ratio']:.6f} "
            f"| components={comp['n_components']} "
            f"| fallback="
            f"{sum(x['exact_fallback_sources'] for x in all_prov if x['split_seed']==seed)} "
            f"| {time.time()-seed_t0:.1f}s",
            flush=True,
        )

    print("[4/5] Saving combined QC ...", flush=True)
    graph_qc = pd.DataFrame(all_graph_qc)
    density_qc = pd.DataFrame(all_density_qc)
    prov_qc = pd.DataFrame(all_prov)

    graph_qc.to_csv(
        OUT / "01_train_graph_LCC_degree_QC.csv",
        index=False,
        encoding="utf-8-sig",
    )
    density_qc.to_csv(
        OUT / "02_neighborhood_density_QC.csv",
        index=False,
        encoding="utf-8-sig",
    )
    prov_qc.to_csv(
        OUT / "03_cache_exact_fallback_provenance.csv",
        index=False,
        encoding="utf-8-sig",
    )

    total_sources = int(prov_qc["n_sources"].sum())
    total_cache = int(prov_qc["cache_recovered_sources"].sum())
    total_fallback = int(prov_qc["exact_fallback_sources"].sum())

    lines = [
        "[P10C WebTEST exact strict-inductive Top10 graph assets]",
        f"Current N                         = {N_CURRENT}",
        f"Count Morgan                      = radius {RADIUS}, fpSize {N_BITS}, chirality=False",
        f"Tanimoto                          = exact",
        f"K                                 = {K}",
        f"Minimum similarity               = > {MIN_SIM}",
        f"Total split/partition sources     = {total_sources}",
        f"Cache exact-recovered sources     = {total_cache}",
        f"Exact current-TRAIN fallbacks     = {total_fallback}",
        f"Cache exact-recovery fraction     = {total_cache/total_sources:.6f}",
        "",
        "[TRAIN GRAPH QC]",
        graph_qc.to_string(index=False),
        "",
        "[NEIGHBORHOOD DENSITY QC]",
        density_qc.to_string(index=False),
        "",
        "Graph policy:",
        "- TRAIN: directed Top10 selections -> undirected UNION -> bidirectional message edges",
        "- VAL/TEST: TRAIN -> QUERY only",
        "- No query-to-train or query-to-query edges",
        "- No approximate neighbor accepted",
        "",
        f"RDKit version                     = {rdBase.rdkitVersion}",
        f"Elapsed seconds                   = {time.time()-t0:.1f}",
        "STATUS: EXACT GRAPH ASSETS CREATED; inspect LCC/density before model fitting.",
    ]

    summary = "\n".join(lines)
    (OUT / "P10C_summary.txt").write_text(summary, encoding="utf-8")

    meta = {
        "scientific_spec": {
            "fingerprint": "Count Morgan",
            "radius": RADIUS,
            "fpSize": N_BITS,
            "includeChirality": False,
            "similarity": "exact Tanimoto",
            "K": K,
            "min_similarity_strict_gt": MIN_SIM,
            "train_graph": "undirected union of directed Top10 selections",
            "query_graph": "TRAIN -> QUERY only",
        },
        "implementation": {
            "legacy_cache": "output/19_g2_raw_edges/daphnia/g2_daphnia__raw_directed.parquet",
            "legacy_cache_root_config": "TOXICS_LEGACY_ROOT environment variable or <project>/legacy_cache",
            "legacy_cache_role": "exact computation cache only",
            "fallback": "exact BulkTanimotoSimilarity against current TRAIN",
            "approximation_used": False,
        },
        "rdkit_version": rdBase.rdkitVersion,
    }
    (OUT / "P10C_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("[5/5] DONE")
    print(summary)


if __name__ == "__main__":
    main()
