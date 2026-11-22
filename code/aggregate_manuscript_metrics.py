#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Aggregate metrics from the saved TEST Daphnia, ADORE external, fathead minnow,
and WebTEST prediction records.

For each structural split, average across model seeds before summarizing the
five splits. Report G2 and descriptor-only comparisons on matched splits.
This script reads existing evaluation records without model fitting or selection."""

from __future__ import annotations

from pathlib import Path
from typing import Optional
import re
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "07_results" / "manuscript_tables" / "manuscript_metric_summary"

SPLIT_SEEDS = [42, 123, 777, 2025, 3407]
MODEL_SEEDS = [42, 123, 777, 2025, 3407]

METRICS = ["RMSE", "MAE", "R2", "Pearson", "Spearman", "bias_pred_minus_true"]
PRIMARY_METRICS = ["RMSE", "MAE", "R2", "Pearson", "Spearman"]

EXPECTED_MODELS = {
    "Daphnia_TEST": ["Descriptor-RF", "Morgan-RF", "G1", "G2", "Self-only"],
    "ADORE_external": ["Descriptor-RF", "Morgan-RF", "G1", "G2", "Self-only"],
    "Fathead_TEST": ["Descriptor-RF", "Morgan-RF", "G1", "G2", "Self-only"],
    "WebTEST_surrogate": ["Descriptor-RF", "Morgan-RF", "G2", "Self-only"],
}

SOURCES = [
    # dataset, folder/file, default model, stage
    ("Daphnia_TEST",
     ROOT / "07_results" / "Daphnia" / "rf_test" / "01_rf_frozen_test_all_runs.csv",
     None, "test"),
    ("Daphnia_TEST",
     ROOT / "07_results" / "Daphnia" / "g1_test" / "g1_frozen_test_all_runs.csv",
     "G1", "test"),
    ("Daphnia_TEST",
     ROOT / "07_results" / "Daphnia" / "g2_test" / "g2_frozen_test_all_runs.csv",
     None, "test"),

    ("ADORE_external",
     ROOT / "07_results" / "Daphnia" / "adore_external" / "RF" / "01_rf_external_all_runs.csv",
     None, "external"),
    ("ADORE_external",
     ROOT / "07_results" / "Daphnia" / "adore_external" / "G1" / "01_g1_external_all_runs.csv",
     "G1", "external"),
    ("ADORE_external",
     ROOT / "07_results" / "Daphnia" / "adore_external" / "G2" / "01_g2_external_all_runs.csv",
     None, "external"),

    ("Fathead_TEST",
     ROOT / "07_results" / "Fathead" / "rf_test" / "01_rf_frozen_test_all_runs.csv",
     None, "test"),
    ("Fathead_TEST",
     ROOT / "07_results" / "Fathead" / "g1_test" / "g1_frozen_test_all_runs.csv",
     "G1", "test"),
    ("Fathead_TEST",
     ROOT / "07_results" / "Fathead" / "g2_test" / "g2_frozen_test_all_runs.csv",
     None, "test"),

    ("WebTEST_surrogate",
     ROOT / "07_results" / "WebTEST_revised" / "rf_test" / "01_completed_run_metrics.csv",
     None, "test"),
    ("WebTEST_surrogate",
     ROOT / "07_results" / "WebTEST_revised" / "g2_descriptor_test" / "01_all_frozen_TEST_runs.csv",
     None, "test"),
]


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)


def normalize_model_name(value: object, source_hint: str = "") -> str:
    raw = str(value).lower().replace("_", "-")

    if "descriptor-rf" in raw or ("descriptor" in raw and "rf" in raw):
        return "Descriptor-RF"
    if "morgan-rf" in raw or ("morgan" in raw and "rf" in raw):
        return "Morgan-RF"
    if "tanimoto" in raw or re.search(r"\bg2\b", raw):
        return "G2"
    if "self" in raw:
        return "Self-only"
    if re.search(r"\bg1\b", raw):
        return "G1"

    hint = str(source_hint).lower().replace("_", "-")
    if "descriptor-rf" in hint or ("descriptor" in hint and "rf" in hint):
        return "Descriptor-RF"
    if "morgan-rf" in hint or ("morgan" in hint and "rf" in hint):
        return "Morgan-RF"

    has_g2 = ("tanimoto" in hint) or bool(re.search(r"\bg2\b", hint))
    has_self = "self" in hint
    has_g1 = bool(re.search(r"\bg1\b", hint))

    if has_g2 and not has_self:
        return "G2"
    if has_self and not has_g2:
        return "Self-only"
    if has_g1:
        return "G1"
    return str(value)


def find_column(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    lower_map = {str(c).lower(): c for c in df.columns}
    for c in candidates:
        if c in df.columns:
            return c
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    return None


def metric_candidates(metric: str, stage: str) -> list[str]:
    bases = [
        f"{stage}_{metric}",
        f"{stage.upper()}_{metric}",
        f"test_{metric}",
        f"external_{metric}",
        metric,
    ]

    # Common alternate spellings.
    if metric == "R2":
        bases += [
            f"{stage}_R2_score", "test_R2_score", "external_R2_score",
            "r2", "r2_score",
        ]
    elif metric == "Pearson":
        bases += [
            f"{stage}_pearson_r", "test_pearson_r", "external_pearson_r",
            "pearson", "pearson_r",
        ]
    elif metric == "Spearman":
        bases += [
            f"{stage}_spearman_rho", "test_spearman_rho", "external_spearman_rho",
            "spearman", "spearman_rho",
        ]
    elif metric == "bias_pred_minus_true":
        bases += [
            f"{stage}_bias", "test_bias", "external_bias",
            "bias", "pred_minus_true_bias",
        ]
    return bases


def standardize_run_file(
    dataset: str,
    path: Path,
    default_model: Optional[str],
    stage: str,
) -> pd.DataFrame:
    d = read_csv(path)

    split_col = find_column(
        d, ["split_seed", "structural_split_seed", "seed_split"]
    )
    model_seed_col = find_column(
        d, ["model_seed", "train_seed", "seed_model"]
    )
    label_col = find_column(
        d, ["model", "condition", "relation_mode"]
    )

    if split_col is None:
        raise KeyError(f"No structural split column: {path}")
    if model_seed_col is None:
        raise KeyError(f"No model-seed column: {path}")

    if label_col is None:
        if default_model is None:
            raise KeyError(f"No model/condition column and no default model: {path}")
        d["__model"] = default_model
        label_col = "__model"

    metric_cols = {}
    for metric in METRICS:
        col = find_column(d, metric_candidates(metric, stage))
        if col is None:
            # Bias is useful QC but not required to proceed if a legacy graph
            # run did not save it. Primary metrics are mandatory.
            if metric in PRIMARY_METRICS:
                raise KeyError(
                    f"Missing required metric {metric} in {path}\n"
                    f"Columns: {list(d.columns)}"
                )
            metric_cols[metric] = None
        else:
            metric_cols[metric] = col

    out = pd.DataFrame({
        "dataset": dataset,
        "source_file": str(path.relative_to(ROOT)),
        "split_seed": pd.to_numeric(d[split_col], errors="raise").astype(int),
        "model_seed": pd.to_numeric(d[model_seed_col], errors="raise").astype(int),
        "raw_model": d[label_col].astype(str),
    })
    out["model"] = [
        normalize_model_name(v, str(path)) for v in out["raw_model"]
    ]

    for metric, col in metric_cols.items():
        if col is None:
            out[metric] = np.nan
        else:
            out[metric] = pd.to_numeric(d[col], errors="raise").astype(float)

    return out[
        ["dataset", "source_file", "split_seed", "model_seed",
         "raw_model", "model", *METRICS]
    ]


def qc_run_structure(run_df: pd.DataFrame) -> list[str]:
    lines = []

    dup_cols = ["dataset", "model", "split_seed", "model_seed"]
    dup = run_df.duplicated(dup_cols, keep=False)
    if dup.any():
        bad = run_df.loc[dup, dup_cols + ["source_file"]]
        raise RuntimeError(
            "Duplicate run keys detecte<local-path>" + bad.to_string(index=False)
        )

    for dataset, expected_models in EXPECTED_MODELS.items():
        g = run_df.loc[run_df["dataset"].eq(dataset)].copy()
        got_models = sorted(g["model"].unique())
        missing = [m for m in expected_models if m not in got_models]
        extra = [m for m in got_models if m not in expected_models]
        if missing or extra:
            raise RuntimeError(
                f"{dataset}: model mismatch. missing={missing}, extra={extra}, got={got_models}"
            )

        for model in expected_models:
            gm = g.loc[g["model"].eq(model)]
            got_splits = sorted(gm["split_seed"].unique().tolist())
            if got_splits != SPLIT_SEEDS:
                raise RuntimeError(
                    f"{dataset} / {model}: split seeds {got_splits}, expected {SPLIT_SEEDS}"
                )

            counts = gm.groupby("split_seed")["model_seed"].nunique()
            if not (counts == len(MODEL_SEEDS)).all():
                raise RuntimeError(
                    f"{dataset} / {model}: model-seed counts by spli<local-path>"
                )

            seeds = sorted(gm["model_seed"].unique().tolist())
            if seeds != MODEL_SEEDS:
                raise RuntimeError(
                    f"{dataset} / {model}: model seeds {seeds}, expected {MODEL_SEEDS}"
                )

        lines.append(
            f"[QC PASS] {dataset}: models={len(expected_models)}, "
            f"splits=5, model_seeds_per_split=5"
        )

    if not np.isfinite(run_df[PRIMARY_METRICS].to_numpy(float)).all():
        raise RuntimeError("Non-finite values found in primary metrics.")

    return lines


def make_splitmeans(run_df: pd.DataFrame) -> pd.DataFrame:
    # Hierarchical rule: average model seeds first inside each structural split.
    splitmeans = (
        run_df.groupby(["dataset", "model", "split_seed"], as_index=False)[METRICS]
        .mean(numeric_only=True)
        .sort_values(["dataset", "model", "split_seed"])
        .reset_index(drop=True)
    )
    return splitmeans


def make_summary(splitmeans: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, model), g in splitmeans.groupby(["dataset", "model"]):
        row = {
            "dataset": dataset,
            "model": model,
            "n_structural_splits": int(g["split_seed"].nunique()),
        }
        for metric in METRICS:
            vals = pd.to_numeric(g[metric], errors="coerce")
            if vals.notna().any():
                row[f"{metric}_mean"] = float(vals.mean())
                row[f"{metric}_std"] = float(vals.std(ddof=1))
                row[f"{metric}_median"] = float(vals.median())
            else:
                row[f"{metric}_mean"] = np.nan
                row[f"{metric}_std"] = np.nan
                row[f"{metric}_median"] = np.nan
        rows.append(row)

    return (
        pd.DataFrame(rows)
        .sort_values(["dataset", "RMSE_mean", "model"])
        .reset_index(drop=True)
    )


def make_metric_rank_table(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset, g in summary.groupby("dataset"):
        for metric in PRIMARY_METRICS:
            col = f"{metric}_mean"
            ascending = metric in ["RMSE", "MAE"]
            z = g[["model", col]].copy()
            z["rank"] = z[col].rank(
                method="min", ascending=ascending
            ).astype(int)
            for _, r in z.sort_values("rank").iterrows():
                rows.append({
                    "dataset": dataset,
                    "metric": metric,
                    "model": r["model"],
                    "value": float(r[col]),
                    "rank": int(r["rank"]),
                })
    return pd.DataFrame(rows)


def make_g2_self_splitwise(splitmeans: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for dataset in EXPECTED_MODELS:
        g = splitmeans.loc[
            splitmeans["dataset"].eq(dataset)
            & splitmeans["model"].isin(["G2", "Self-only"])
        ].copy()

        if set(g["model"]) != {"G2", "Self-only"}:
            raise RuntimeError(f"{dataset}: G2/self pair missing.")

        for split_seed in SPLIT_SEEDS:
            z = g.loc[g["split_seed"].eq(split_seed)].set_index("model")
            if set(z.index) != {"G2", "Self-only"}:
                raise RuntimeError(
                    f"{dataset} split={split_seed}: incomplete G2/self pair."
                )

            row = {
                "dataset": dataset,
                "split_seed": split_seed,
            }

            for metric in PRIMARY_METRICS:
                g2 = float(z.loc["G2", metric])
                selfv = float(z.loc["Self-only", metric])
                row[f"G2_{metric}"] = g2
                row[f"Self_{metric}"] = selfv

                # Direction-normalized: positive always means G2 is better.
                if metric in ["RMSE", "MAE"]:
                    adv = selfv - g2
                else:
                    adv = g2 - selfv
                row[f"G2_advantage_{metric}"] = adv

            # Bias: compare absolute magnitude; positive means G2 is closer to zero.
            if z["bias_pred_minus_true"].notna().all():
                g2b = float(z.loc["G2", "bias_pred_minus_true"])
                sb = float(z.loc["Self-only", "bias_pred_minus_true"])
                row["G2_bias"] = g2b
                row["Self_bias"] = sb
                row["G2_advantage_abs_bias"] = abs(sb) - abs(g2b)

            rows.append(row)

    return pd.DataFrame(rows)


def make_g2_self_summary(pair: pd.DataFrame) -> pd.DataFrame:
    advantage_cols = [
        "G2_advantage_RMSE",
        "G2_advantage_MAE",
        "G2_advantage_R2",
        "G2_advantage_Pearson",
        "G2_advantage_Spearman",
    ]
    if "G2_advantage_abs_bias" in pair.columns:
        advantage_cols.append("G2_advantage_abs_bias")

    rows = []
    for dataset, g in pair.groupby("dataset"):
        row = {"dataset": dataset, "n_structural_splits": len(g)}
        for col in advantage_cols:
            vals = pd.to_numeric(g[col], errors="coerce").dropna()
            if len(vals) == 0:
                continue
            row[f"{col}_mean"] = float(vals.mean())
            row[f"{col}_std"] = float(vals.std(ddof=1))
            row[f"{col}_median"] = float(vals.median())
            row[f"{col}_n_G2_better"] = int((vals > 0).sum())
            row[f"{col}_n_tie"] = int(np.isclose(vals, 0.0, atol=1e-12).sum())
            row[f"{col}_n_G2_worse"] = int((vals < 0).sum())
        rows.append(row)

    return pd.DataFrame(rows).sort_values("dataset").reset_index(drop=True)


def make_report(
    run_df: pd.DataFrame,
    summary: pd.DataFrame,
    pair_summary: pd.DataFrame,
    qc_lines: list[str],
) -> str:
    lines = [
        "[MANUSCRIPT METRIC AGGREGATION]",
        "",
        "No model fitting/tuning/split changes were performed.",
        "Hierarchical policy: model seeds averaged within structural split first;",
        "then 5 structural splits summarized by mean / SD / median.",
        "",
        "[SOURCE FILES]",
    ]
    for p in sorted(run_df["source_file"].unique()):
        n = int((run_df["source_file"] == p).sum())
        lines.append(f"- {p} | rows={n}")

    lines += ["", "[QC]"] + qc_lines

    lines += [
        "",
        "[FULL PERFORMANCE SUMMARY]",
        summary.to_string(index=False),
        "",
        "[G2 VS SELF-ONLY: DIRECTION-NORMALIZED ADVANTAGE]",
        "Positive advantage always means G2 performed better.",
        "RMSE/MAE: Self - G2",
        "R2/Pearson/Spearman: G2 - Self",
        "Absolute bias (if available): |Self bias| - |G2 bias|",
        "",
        pair_summary.to_string(index=False),
        "",
        "[INTERPRETATION BOUNDARY]",
        "- This report is descriptive and paired by frozen structural split.",
        "- No significance test is used to turn five structural splits into a claim of universality.",
        "- Do not rename/reframe G2 or self-only for the manuscript until this table is reviewed.",
        "- Do not retune any model after reviewing these frozen results.",
    ]
    return "\n".join(lines)


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    pieces = []
    for dataset, path, default_model, stage in SOURCES:
        print(f"[LOAD] {path.relative_to(ROOT)}")
        pieces.append(
            standardize_run_file(
                dataset=dataset,
                path=path,
                default_model=default_model,
                stage=stage,
            )
        )

    run_df = pd.concat(pieces, ignore_index=True)
    qc_lines = qc_run_structure(run_df)
    for line in qc_lines:
        print(line)

    splitmeans = make_splitmeans(run_df)
    summary = make_summary(splitmeans)
    ranks = make_metric_rank_table(summary)
    pair = make_g2_self_splitwise(splitmeans)
    pair_summary = make_g2_self_summary(pair)

    run_df.to_csv(
        OUT / "01_all_frozen_run_metrics_standardized.csv",
        index=False, encoding="utf-8-sig"
    )
    splitmeans.to_csv(
        OUT / "02_splitmean_all_metrics.csv",
        index=False, encoding="utf-8-sig"
    )
    summary.to_csv(
        OUT / "03_full_performance_summary.csv",
        index=False, encoding="utf-8-sig"
    )
    ranks.to_csv(
        OUT / "04_metric_rank_by_dataset.csv",
        index=False, encoding="utf-8-sig"
    )
    pair.to_csv(
        OUT / "05_G2_vs_self_splitwise_all_metrics.csv",
        index=False, encoding="utf-8-sig"
    )
    pair_summary.to_csv(
        OUT / "06_G2_vs_self_summary_all_metrics.csv",
        index=False, encoding="utf-8-sig"
    )

    report = make_report(run_df, summary, pair_summary, qc_lines)
    (OUT / "manuscript_metric_summary.txt").write_text(
        report, encoding="utf-8"
    )

    print("")
    print("[FULL PERFORMANCE SUMMARY]")
    print(summary.to_string(index=False))
    print("")
    print("[G2 VS SELF-ONLY SUMMARY]")
    print(pair_summary.to_string(index=False))
    print("")
    print(f"[DONE] {OUT}")


if __name__ == "__main__":
    main()
