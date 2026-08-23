#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
characterize_top10_similarity.py

Structural QC only for the already-frozen G2 graph definition.

Question
--------
Does fixed Top-K=10 select neighborhoods of comparable absolute structural
similarity across Daphnia TEST, ADORE external, Fathead TEST, and the revised
WebTEST surrogate regime?

IMPORTANT
---------
- NO model fitting
- NO retraining
- NO graph modification
- NO threshold optimization
- NO toxicity labels used
- NO prediction values used
- NO TEST outcome used

This script only characterizes the absolute Tanimoto similarity of the
TRAIN neighbors that would be / were selected under the frozen G2 policy:

    Count Morgan radius=4, fpSize=2048, includeChirality=False
    exact Tanimoto
    Top-K=10
    similarity > 0
    strict inductive TRAIN -> QUERY

For Daphnia TEST, Fathead TEST, and ADORE external, exact TRAIN->QUERY
similarities are recomputed from the frozen data/split assets.
For WebTEST, the already-created exact WebTEST TEST-to-TRAIN Top-10 assets are
read directly to avoid recomputing the 119,276-chemical search.

Interpretation boundary
-----------------------
The descriptive thresholds (0.2, 0.3, 0.4, 0.5) are NOT model thresholds.
They are used only to describe neighborhood strength and MUST NOT be used
to retune G2 after frozen TEST results have been observed.
"""

from __future__ import annotations

from pathlib import Path
import json
import time
from typing import Dict, List

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger, rdBase
from rdkit.Chem import rdFingerprintGenerator


ROOT = Path(__file__).resolve().parents[1]
OUT = (
    ROOT / "07_results" / "diagnostics"
    / "top10_similarity_characterization"
)

SPLIT_SEEDS = [42, 123, 777, 2025, 3407]
TOPK = 10
RADIUS = 4
N_BITS = 2048
MIN_SIM = 0.0
THRESHOLDS = [0.2, 0.3, 0.4, 0.5]

DAPHNIA = (
    ROOT / "03_processed" / "TEST"
    / "TEST_Daphnia_modeling_pool_no_ADORE_overlap.csv"
)
DAPHNIA_SPLITS = (
    ROOT / "04_splits" / "TEST_Daphnia"
    / "scaffold_split_murcko_topology_refined"
)

FATHEAD = (
    ROOT / "03_processed" / "TEST"
    / "TEST_Fathead_96h_LC50_modeling_pool.csv"
)
FATHEAD_SPLITS = (
    ROOT / "04_splits" / "TEST_Fathead"
    / "scaffold_split_murcko_topology_refined"
)

ADORE = (
    ROOT / "03_processed" / "ADORE"
    / "ADORE_Daphnia_48h_LC50_external_chemical_level.csv"
)

WEBTEST_GRAPH_ROOT = (
    ROOT / "07_results" / "WebTEST_revised"
    / "top10_graph_assets"
)

CONTEXTS = {
    "Daphnia_TEST": {
        "train_table": DAPHNIA,
        "query_table": DAPHNIA,
        "split_root": DAPHNIA_SPLITS,
        "train_smiles_col": "canonical_SMILES_RDKit",
        "query_smiles_col": "canonical_SMILES_RDKit",
        "query_partition": "test",
        "expected_train_dataset_n": 316,
        "expected_query_dataset_n": 316,
        "source": "recompute_exact",
    },
    "ADORE_external": {
        "train_table": DAPHNIA,
        "query_table": ADORE,
        "split_root": DAPHNIA_SPLITS,
        "train_smiles_col": "canonical_SMILES_RDKit",
        "query_smiles_col": "smiles_connectivity",
        "query_partition": "external",
        "expected_train_dataset_n": 316,
        "expected_query_dataset_n": 274,
        "source": "recompute_exact",
    },
    "Fathead_TEST": {
        "train_table": FATHEAD,
        "query_table": FATHEAD,
        "split_root": FATHEAD_SPLITS,
        "train_smiles_col": "canonical_SMILES_RDKit",
        "query_smiles_col": "canonical_SMILES_RDKit",
        "query_partition": "test",
        "expected_train_dataset_n": 821,
        "expected_query_dataset_n": 821,
        "source": "recompute_exact",
    },
    "WebTEST_surrogate": {
        "query_partition": "test",
        "source": "exact_webtest_assets",
    },
}


RDLogger.DisableLog("rdApp.warning")


def load_split_indices(split_root: Path, seed: int, n_rows: int) -> Dict[str, np.ndarray]:
    p = split_root / f"split_seed{seed}" / "split_membership.csv"
    if not p.exists():
        raise FileNotFoundError(p)

    m = pd.read_csv(p, encoding="utf-8-sig", low_memory=False)
    idx_col = next(
        (
            c
            for c in ["original_row_index", "global_row_index", "row_idx", "index"]
            if c in m.columns
        ),
        None,
    )
    if idx_col is None:
        raise KeyError(f"No row-index column in {p}")

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
    if len(all_idx) != n_rows:
        raise RuntimeError(
            f"seed={seed}: split rows={len(all_idx)} expected={n_rows}"
        )
    if len(np.unique(all_idx)) != n_rows:
        raise RuntimeError(f"seed={seed}: duplicate split membership")
    if all_idx.min() < 0 or all_idx.max() >= n_rows:
        raise RuntimeError(f"seed={seed}: split index out of range")

    return out


def build_fps(smiles: List[str]) -> List:
    gen = rdFingerprintGenerator.GetMorganGenerator(
        radius=RADIUS,
        fpSize=N_BITS,
        includeChirality=False,
    )
    fps = []
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            raise ValueError(f"Invalid SMILES row={i}: {smi}")
        fps.append(gen.GetCountFingerprint(mol))
    return fps


def summarize_selected_sims(
    context: str,
    split_seed: int,
    query_row_index: int,
    sims_selected: np.ndarray,
) -> dict:
    sims_selected = np.asarray(sims_selected, dtype=float)
    sims_selected = sims_selected[np.isfinite(sims_selected)]
    sims_selected = np.sort(sims_selected)[::-1]

    n = int(len(sims_selected))
    row = {
        "context": context,
        "split_seed": int(split_seed),
        "query_row_index": int(query_row_index),
        "n_neighbors": n,
        "T1": float(sims_selected[0]) if n else np.nan,
        "mean_Top10": float(np.mean(sims_selected)) if n else np.nan,
        "median_Top10": float(np.median(sims_selected)) if n else np.nan,
        "T10": float(sims_selected[TOPK - 1]) if n >= TOPK else np.nan,
        "selected_min": float(np.min(sims_selected)) if n else np.nan,
        "spread_T1_T10": (
            float(sims_selected[0] - sims_selected[TOPK - 1])
            if n >= TOPK
            else np.nan
        ),
    }
    for t in THRESHOLDS:
        tag = str(t).replace(".", "p")
        row[f"n_ge_{tag}"] = int(np.sum(sims_selected >= t))
        row[f"all10_ge_{tag}"] = bool(n >= TOPK and sims_selected[TOPK - 1] >= t)
    return row


def exact_context(context: str, spec: dict) -> pd.DataFrame:
    td = pd.read_csv(
        spec["train_table"], encoding="utf-8-sig", low_memory=False
    ).reset_index(drop=True)
    qd = pd.read_csv(
        spec["query_table"], encoding="utf-8-sig", low_memory=False
    ).reset_index(drop=True)

    if len(td) != spec["expected_train_dataset_n"]:
        raise RuntimeError(
            f"{context}: train dataset N={len(td)} "
            f"expected={spec['expected_train_dataset_n']}"
        )
    if len(qd) != spec["expected_query_dataset_n"]:
        raise RuntimeError(
            f"{context}: query dataset N={len(qd)} "
            f"expected={spec['expected_query_dataset_n']}"
        )

    for df, col in [
        (td, spec["train_smiles_col"]),
        (qd, spec["query_smiles_col"]),
    ]:
        if col not in df.columns:
            raise KeyError(f"{context}: missing SMILES column {col}")

    train_smiles = td[spec["train_smiles_col"]].astype(str).tolist()
    same_table = spec["train_table"].resolve() == spec["query_table"].resolve()

    if same_table:
        all_fps = build_fps(train_smiles)
        query_fps = all_fps
    else:
        query_smiles = qd[spec["query_smiles_col"]].astype(str).tolist()
        all_fps = build_fps(train_smiles)
        query_fps = build_fps(query_smiles)

    rows = []

    for seed in SPLIT_SEEDS:
        idx = load_split_indices(spec["split_root"], seed, len(td))
        train_idx = idx["train"]
        train_fps = [all_fps[int(i)] for i in train_idx]

        if spec["query_partition"] == "test":
            query_idx = idx["test"]
        elif spec["query_partition"] == "external":
            query_idx = np.arange(len(qd), dtype=int)
        else:
            raise ValueError(spec["query_partition"])

        print(
            f"[{context}] seed={seed} train={len(train_idx)} "
            f"query={len(query_idx)}",
            flush=True,
        )

        for j, q_idx in enumerate(query_idx):
            if j and j % 5000 == 0:
                print(
                    f"  {context} seed={seed}: {j}/{len(query_idx)}",
                    flush=True,
                )

            qfp = (
                query_fps[int(q_idx)]
                if not same_table
                else all_fps[int(q_idx)]
            )

            sims = np.asarray(
                DataStructs.BulkTanimotoSimilarity(qfp, train_fps),
                dtype=float,
            )

            sims = sims[np.isfinite(sims) & (sims > MIN_SIM)]
            if len(sims):
                order = np.argsort(-sims, kind="stable")
                selected = sims[order[:TOPK]]
            else:
                selected = np.asarray([], dtype=float)

            rows.append(
                summarize_selected_sims(
                    context=context,
                    split_seed=seed,
                    query_row_index=int(q_idx),
                    sims_selected=selected,
                )
            )

    return pd.DataFrame(rows)


def webtest_context() -> pd.DataFrame:
    rows = []
    for seed in SPLIT_SEEDS:
        p = (
            WEBTEST_GRAPH_ROOT / f"split_seed{seed}"
            / "test_source_to_train_exact_top10.parquet"
        )
        if not p.exists():
            raise FileNotFoundError(p)

        e = pd.read_parquet(
            p,
            columns=[
                "source_global_idx",
                "rank",
                "similarity",
            ],
        )
        e["source_global_idx"] = pd.to_numeric(
            e["source_global_idx"], errors="raise"
        ).astype(int)
        e["rank"] = pd.to_numeric(e["rank"], errors="raise").astype(int)
        e["similarity"] = pd.to_numeric(
            e["similarity"], errors="raise"
        ).astype(float)

        print(
            f"[WebTEST_surrogate] seed={seed} "
            f"query={e['source_global_idx'].nunique()} "
            f"edges={len(e)} (read existing exact WebTEST asset)",
            flush=True,
        )

        for qidx, g in e.groupby("source_global_idx", sort=True):
            g = g.sort_values(["rank", "similarity"], ascending=[True, False])
            rows.append(
                summarize_selected_sims(
                    context="WebTEST_surrogate",
                    split_seed=seed,
                    query_row_index=int(qidx),
                    sims_selected=g["similarity"].to_numpy(float),
                )
            )

    return pd.DataFrame(rows)


def split_summary(per_query: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (context, seed), g in per_query.groupby(
        ["context", "split_seed"], sort=False
    ):
        n = len(g)
        full = g["n_neighbors"].eq(TOPK)
        t10 = g.loc[full, "T10"].dropna()

        row = {
            "context": context,
            "split_seed": int(seed),
            "n_query": int(n),
            "fraction_full_K10": float(full.mean()),
            "T1_mean": float(g["T1"].mean()),
            "T1_median": float(g["T1"].median()),
            "mean_Top10_mean": float(g["mean_Top10"].mean()),
            "mean_Top10_median": float(g["mean_Top10"].median()),
            "T10_mean": float(t10.mean()) if len(t10) else np.nan,
            "T10_q25": float(t10.quantile(0.25)) if len(t10) else np.nan,
            "T10_median": float(t10.median()) if len(t10) else np.nan,
            "T10_q75": float(t10.quantile(0.75)) if len(t10) else np.nan,
            "spread_T1_T10_mean": float(g["spread_T1_T10"].mean()),
        }

        for t in THRESHOLDS:
            tag = str(t).replace(".", "p")
            row[f"fraction_all10_ge_{tag}"] = float(
                g[f"all10_ge_{tag}"].mean()
            )
            row[f"fraction_T10_lt_{tag}"] = float(
                (g["T10"].lt(t) | g["T10"].isna()).mean()
            )

        rows.append(row)

    return pd.DataFrame(rows)


def context_summary(split_df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [
        c
        for c in split_df.columns
        if c not in {"context", "split_seed"}
    ]

    rows = []
    for context, g in split_df.groupby("context", sort=False):
        row = {
            "context": context,
            "n_structural_splits": int(g["split_seed"].nunique()),
            "n_query_splitmean": float(g["n_query"].mean()),
        }
        for c in numeric_cols:
            if c == "n_query":
                continue
            vals = pd.to_numeric(g[c], errors="coerce")
            row[f"{c}_mean"] = float(vals.mean())
            row[f"{c}_sd"] = (
                float(vals.std(ddof=1)) if vals.notna().sum() > 1 else np.nan
            )
        rows.append(row)

    return pd.DataFrame(rows)


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)

    required = [
        DAPHNIA,
        DAPHNIA_SPLITS,
        FATHEAD,
        FATHEAD_SPLITS,
        ADORE,
        WEBTEST_GRAPH_ROOT,
    ]
    for p in required:
        if not p.exists():
            raise FileNotFoundError(p)

    pieces = []
    for context in ["Daphnia_TEST", "ADORE_external", "Fathead_TEST"]:
        pieces.append(exact_context(context, CONTEXTS[context]))

    pieces.append(webtest_context())
    per_query = pd.concat(pieces, ignore_index=True)

    expected_contexts = set(CONTEXTS)
    if set(per_query["context"].unique()) != expected_contexts:
        raise RuntimeError(
            f"Context mismatch: {sorted(per_query['context'].unique())}"
        )

    split_df = split_summary(per_query)
    context_df = context_summary(split_df)

    per_query.to_csv(
        OUT / "01_per_query_Top10_similarity.csv",
        index=False,
        encoding="utf-8-sig",
    )
    split_df.to_csv(
        OUT / "02_by_split_Top10_similarity_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    context_df.to_csv(
        OUT / "03_cross_context_Top10_similarity_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    display_cols = [
        "context",
        "n_query_splitmean",
        "fraction_full_K10_mean",
        "T1_mean_mean",
        "mean_Top10_mean_mean",
        "T10_mean_mean",
        "T10_median_mean",
        "fraction_T10_lt_0p3_mean",
        "fraction_T10_lt_0p4_mean",
        "fraction_T10_lt_0p5_mean",
    ]
    compact = context_df[display_cols].copy()
    compact.to_csv(
        OUT / "04_compact_manuscript_QC_table.csv",
        index=False,
        encoding="utf-8-sig",
    )

    meta = {
        "module": "characterize_top10_similarity",
        "status": "DESCRIPTIVE_STRUCTURAL_CHARACTERIZATION_ONLY",
        "model_fitting": False,
        "model_retraining": False,
        "graph_modification": False,
        "threshold_optimization": False,
        "toxicity_labels_used": False,
        "predictions_used": False,
        "test_outcomes_used": False,
        "frozen_graph_spec": {
            "fingerprint": "Count Morgan",
            "radius": RADIUS,
            "fpSize": N_BITS,
            "includeChirality": False,
            "similarity": "exact Tanimoto",
            "TopK": TOPK,
            "min_similarity_strict_gt": MIN_SIM,
            "query_policy": "TRAIN -> QUERY only",
            "message_weighting_in_frozen_G2": "unweighted",
        },
        "descriptive_thresholds_only": THRESHOLDS,
        "webtest_source": str(WEBTEST_GRAPH_ROOT),
        "rdkit_version": rdBase.rdkitVersion,
    }
    (OUT / "top10_similarity_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    lines = [
        "[TOP-10 TANIMOTO STRUCTURAL CHARACTERIZATION]",
        "",
        "DESCRIPTIVE STRUCTURAL CHARACTERIZATION ONLY.",
        "NO model fitting. NO retraining. NO graph modification.",
        "NO toxicity labels or predictions used.",
        "Thresholds 0.2/0.3/0.4/0.5 are descriptive only.",
        "",
        "Frozen G2 relation:",
        "Count Morgan r4/2048 -> exact Tanimoto -> Top-K10, similarity > 0",
        "Message passing in G2: unweighted",
        "",
        "Definitions:",
        "T1  = strongest selected TRAIN neighbor similarity.",
        "T10 = rank-10 selected TRAIN neighbor similarity.",
        "mean_Top10 = mean similarity of selected Top-10 TRAIN neighbors.",
        "fraction_T10_lt_X = fraction of query chemicals whose 10th neighbor is below X",
        "                    (or that do not have 10 positive-similarity neighbors).",
        "",
        "[COMPACT CROSS-CONTEXT SUMMARY]",
        compact.to_string(index=False),
        "",
        "[INTERPRETATION BOUNDARY]",
        "- This analysis characterizes neighborhood strength only.",
        "- It does NOT test a new thresholded or weighted graph.",
        "- It must NOT be used to choose a new K/threshold after frozen TEST results.",
        "- Use it to assess whether fixed K=10 implies materially different",
        "  absolute-neighborhood quality across evaluation contexts.",
        "",
        f"Elapsed seconds: {time.time() - t0:.1f}",
        f"Outputs: {OUT}",
    ]
    summary = "\n".join(lines)
    (OUT / "top10_similarity_summary.txt").write_text(summary, encoding="utf-8")

    print("\n" + summary)


if __name__ == "__main__":
    main()
