#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
webtest_evaluate_g2.py

Evaluate the already validation-selected WebTEST G2 bundles on the previously
unopened WebTEST structural TEST partitions.

NO retraining.
NO hyperparameter changes.
NO model selection using TEST.

Uses:
- validation-selected bundle state_dicts
- identical six-descriptor set TRAIN-fitted preprocessing stored in each bundle
- identical target z-scaling stored in each bundle
- exact strict-inductive TEST edges
- G2 TEST policy: TRAIN -> TEST only
- self: no graph edges
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch

try:
    import webtest_train_g2 as webtrain
except Exception as e:
    raise RuntimeError(
        "webtest_train_g2.py must be in the same scripts folder."
    ) from e

ROOT = Path(__file__).resolve().parents[1]

INPUT = (
    ROOT / "03_processed" / "WebTEST"
    / "WebTEST_Daphnia_consensus_unique_connectivity_frozen.csv"
)
VAL_ROOT = (
    ROOT / "07_results" / "WebTEST_revised"
    / "g2_descriptor_validation"
)
BUNDLE_ROOT = VAL_ROOT / "model_bundles"
OUT = (
    ROOT / "07_results" / "WebTEST_revised"
    / "g2_descriptor_test"
)

EXPECTED_N = 119276
SPLIT_SEEDS = [42, 123, 777, 2025, 3407]
MODEL_SEEDS = [42, 123, 777, 2025, 3407]
CONDITIONS = ["descriptor_self", "descriptor_tanimoto_K10"]


def load_bundle(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def summarize_hierarchical(run_df):
    metrics = ["RMSE", "MAE", "R2", "Pearson", "Spearman"]
    rows = []

    for condition, g in run_df.groupby("condition", sort=False):
        split_means = (
            g.groupby("split_seed", as_index=False)[
                [f"test_{m}" for m in metrics]
            ]
            .mean()
        )
        row = {
            "condition": condition,
            "n_structural_splits": int(split_means["split_seed"].nunique()),
            "model_seeds_per_split": int(
                g.groupby("split_seed")["model_seed"].nunique().min()
            ),
        }
        for m in metrics:
            s = split_means[f"test_{m}"]
            row[f"test_{m}_splitmean_mean"] = float(s.mean())
            row[f"test_{m}_splitmean_std"] = float(s.std(ddof=1))
            row[f"test_{m}_splitmean_median"] = float(s.median())
        rows.append(row)

    return pd.DataFrame(rows)


def main():
    for p in [INPUT, VAL_ROOT, BUNDLE_ROOT]:
        if not p.exists():
            raise FileNotFoundError(p)

    OUT.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(
        INPUT, encoding="utf-8-sig", low_memory=False
    ).reset_index(drop=True)
    if len(df) != EXPECTED_N:
        raise RuntimeError(
            f"Expected N={EXPECTED_N}, found {len(df)}"
        )

    y_all = pd.to_numeric(
        df[webtrain.Y_COL], errors="raise"
    ).to_numpy(dtype=float)

    print("[1/4] Rebuilding raw six-descriptor set deterministically ...", flush=True)
    X_raw = webtrain.build_compact6_raw(
        df[webtrain.SMILES_COL].astype(str).tolist()
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"[2/4] Device = {device}", flush=True)

    run_rows = []
    pred_rows = []

    total = len(SPLIT_SEEDS) * len(MODEL_SEEDS) * len(CONDITIONS)
    done = 0

    print("[3/4] Opening frozen TEST partitions ...", flush=True)

    for split_seed in SPLIT_SEEDS:
        idx = webtrain.load_split(split_seed, len(df))
        train_idx = idx["train"]
        test_idx = idx["test"]

        # Exact precomputed TRAIN graph + strict-inductive TEST query edges.
        real_train, real_test = webtrain.load_precomputed_edges(
            seed=split_seed,
            train_idx=train_idx,
            query_idx=test_idx,
            query_part="test",
        )

        empty_train = (
            torch.empty((2, 0), dtype=torch.long),
            torch.empty((0,), dtype=torch.float32),
        )
        empty_test = (
            torch.empty((2, 0), dtype=torch.long),
            torch.empty((0,), dtype=torch.float32),
        )

        for model_seed in MODEL_SEEDS:
            for condition in CONDITIONS:
                bundle_path = (
                    BUNDLE_ROOT
                    / f"split{split_seed}__model{model_seed}__{condition}.pt"
                )
                if not bundle_path.exists():
                    raise FileNotFoundError(bundle_path)

                b = load_bundle(bundle_path)

                if bool(b.get("test_evaluated", True)):
                    raise RuntimeError(
                        f"Bundle was not marked TEST-unopened: {bundle_path}"
                    )
                if int(b["split_seed"]) != split_seed:
                    raise RuntimeError("split_seed mismatch in bundle")
                if int(b["model_seed"]) != model_seed:
                    raise RuntimeError("model_seed mismatch in bundle")
                if str(b["condition"]) != condition:
                    raise RuntimeError("condition mismatch in bundle")
                if int(b["n_total"]) != EXPECTED_N:
                    raise RuntimeError("N mismatch in bundle")

                pp = b["preprocessor"]
                X_train = webtrain.shared.transform_descriptor_rows(
                    X_raw, train_idx, pp
                )
                X_test = webtrain.shared.transform_descriptor_rows(
                    X_raw, test_idx, pp
                )
                X_graph_np = np.concatenate(
                    [X_train, X_test], axis=0
                )

                if condition == "descriptor_self":
                    mode = "self"
                    graph_edges = empty_test
                elif condition == "descriptor_tanimoto_K10":
                    mode = "unweighted"
                    graph_edges = real_test
                else:
                    raise RuntimeError(condition)

                arch = b["architecture"]
                model = webtrain.base.ControlledGraphSAGE(
                    input_dim=int(arch["input_dim"]),
                    hidden_dim=int(arch["hidden_dim"]),
                    num_layers=int(arch["num_layers"]),
                    dropout=float(arch["dropout"]),
                    mode=mode,
                ).to(device)

                model.load_state_dict(b["state_dict"])
                model.eval()

                X_graph = torch.tensor(
                    X_graph_np, dtype=torch.float32, device=device
                )

                ei, ew = graph_edges
                ei = ei.to(device)
                ew = ew.to(device)

                with torch.no_grad():
                    pred_all_z = (
                        model(X_graph, ei, ew)
                        .detach().cpu().numpy()
                    )

                n_train = len(train_idx)
                pred_test_z = pred_all_z[
                    n_train : n_train + len(test_idx)
                ]

                y_mean = float(b["target_z"]["mean"])
                y_std = float(b["target_z"]["std"])
                pred_test = pred_test_z * y_std + y_mean
                y_test = y_all[test_idx]

                metrics = webtrain.regression_metrics(
                    y_test, pred_test
                )

                run_rows.append({
                    "split_seed": int(split_seed),
                    "model_seed": int(model_seed),
                    "condition": condition,
                    "relation_mode": mode,
                    "n_total": int(len(df)),
                    "n_train": int(len(train_idx)),
                    "n_val": int(len(idx["val"])),
                    "n_test": int(len(test_idx)),
                    "bundle_best_epoch": int(b["best_epoch"]),
                    "test_RMSE": float(metrics["RMSE"]),
                    "test_MAE": float(metrics["MAE"]),
                    "test_R2": float(metrics["R2"]),
                    "test_Pearson": float(metrics["Pearson"]),
                    "test_Spearman": float(metrics["Spearman"]),
                    "retrained_on_test": False,
                    "test_used_for_selection": False,
                })

                for j, global_idx in enumerate(test_idx):
                    pred_rows.append({
                        "split_seed": int(split_seed),
                        "model_seed": int(model_seed),
                        "condition": condition,
                        "global_row_index": int(global_idx),
                        "y_true": float(y_test[j]),
                        "y_pred": float(pred_test[j]),
                    })

                done += 1
                print(
                    f"[{done}/{total}] split={split_seed} "
                    f"model={model_seed} {condition} "
                    f"TEST_RMSE={metrics['RMSE']:.6f}",
                    flush=True,
                )

    run_df = pd.DataFrame(run_rows)
    pred_df = pd.DataFrame(pred_rows)

    summary_df = summarize_hierarchical(run_df)

    # Paired G2-self TEST delta.
    piv = run_df.pivot_table(
        index=["split_seed", "model_seed"],
        columns="condition",
        values="test_RMSE",
        aggfunc="first",
    ).reset_index()

    piv["g2_minus_self_test_RMSE"] = (
        piv["descriptor_tanimoto_K10"]
        - piv["descriptor_self"]
    )
    split_delta = (
        piv.groupby("split_seed", as_index=False)[
            "g2_minus_self_test_RMSE"
        ].mean()
    )
    delta_summary = pd.DataFrame([{
        "comparison": "G2_minus_six-descriptor set_self",
        "n_structural_splits": int(
            split_delta["split_seed"].nunique()
        ),
        "model_seeds_per_split": len(MODEL_SEEDS),
        "delta_splitmean_mean": float(
            split_delta["g2_minus_self_test_RMSE"].mean()
        ),
        "delta_splitmean_std": float(
            split_delta["g2_minus_self_test_RMSE"].std(ddof=1)
        ),
        "delta_splitmean_median": float(
            split_delta["g2_minus_self_test_RMSE"].median()
        ),
        "n_splits_G2_better": int(
            (split_delta["g2_minus_self_test_RMSE"] < 0).sum()
        ),
        "n_splits_self_better": int(
            (split_delta["g2_minus_self_test_RMSE"] > 0).sum()
        ),
    }])

    # Validation-to-test comparison using the saved validation run table.
    val_runs_path = VAL_ROOT / "01_all_runs.csv"
    if not val_runs_path.exists():
        raise FileNotFoundError(val_runs_path)
    val_df = pd.read_csv(
        val_runs_path, encoding="utf-8-sig", low_memory=False
    )

    joined = run_df.merge(
        val_df[
            [
                "split_seed",
                "model_seed",
                "condition",
                "val_RMSE",
                "val_MAE",
                "val_R2",
                "val_Pearson",
                "val_Spearman",
            ]
        ],
        on=["split_seed", "model_seed", "condition"],
        how="left",
        validate="one_to_one",
    )
    joined["TEST_minus_VAL_RMSE"] = (
        joined["test_RMSE"] - joined["val_RMSE"]
    )

    vt_rows = []
    for condition, g in joined.groupby("condition", sort=False):
        sm = (
            g.groupby("split_seed", as_index=False)[
                "TEST_minus_VAL_RMSE"
            ].mean()
        )
        val_sm = (
            g.groupby("split_seed", as_index=False)["val_RMSE"]
            .mean()
        )
        test_sm = (
            g.groupby("split_seed", as_index=False)["test_RMSE"]
            .mean()
        )
        vt_rows.append({
            "condition": condition,
            "val_RMSE_splitmean_mean": float(
                val_sm["val_RMSE"].mean()
            ),
            "test_RMSE_splitmean_mean": float(
                test_sm["test_RMSE"].mean()
            ),
            "TEST_minus_VAL_RMSE_mean": float(
                sm["TEST_minus_VAL_RMSE"].mean()
            ),
            "relative_change_percent": float(
                100.0
                * sm["TEST_minus_VAL_RMSE"].mean()
                / val_sm["val_RMSE"].mean()
            ),
        })
    vt_df = pd.DataFrame(vt_rows)

    run_df.to_csv(
        OUT / "01_all_frozen_TEST_runs.csv",
        index=False, encoding="utf-8-sig"
    )
    summary_df.to_csv(
        OUT / "02_hierarchical_TEST_summary.csv",
        index=False, encoding="utf-8-sig"
    )
    pred_df.to_csv(
        OUT / "03_frozen_TEST_predictions.csv",
        index=False, encoding="utf-8-sig"
    )
    piv.to_csv(
        OUT / "04_paired_G2_self_by_split_modelseed.csv",
        index=False, encoding="utf-8-sig"
    )
    split_delta.to_csv(
        OUT / "05_paired_G2_self_splitmeans.csv",
        index=False, encoding="utf-8-sig"
    )
    delta_summary.to_csv(
        OUT / "06_paired_G2_self_delta_summary.csv",
        index=False, encoding="utf-8-sig"
    )
    joined.to_csv(
        OUT / "07_validation_to_TEST_all_runs.csv",
        index=False, encoding="utf-8-sig"
    )
    vt_df.to_csv(
        OUT / "08_validation_to_TEST_summary.csv",
        index=False, encoding="utf-8-sig"
    )

    meta = {
        "status": "FROZEN_TEST_EVALUATED",
        "retraining": False,
        "test_used_for_model_selection": False,
        "test_used_for_hyperparameter_tuning": False,
        "n_total": EXPECTED_N,
        "structural_split_seeds": SPLIT_SEEDS,
        "model_seeds": MODEL_SEEDS,
        "conditions": CONDITIONS,
        "target": "-log10(LC50 mol/L)",
        "G2_query_policy": "TRAIN -> TEST only",
    }
    (OUT / "g2_test_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    lines = [
        "[WebTEST Daphnia G2 / descriptor-only GraphSAGE test]",
        f"N total              = {EXPECTED_N}",
        f"Structural splits    = {SPLIT_SEEDS}",
        f"Model seeds          = {MODEL_SEEDS}",
        "Retraining on TEST   = NO",
        "TEST used selection  = NO",
        "G2 TEST graph        = TRAIN -> TEST only",
        "",
        "[HIERARCHICAL FROZEN TEST SUMMARY]",
        summary_df.to_string(index=False),
        "",
        "[PAIRED G2 - SELF TEST DELTA]",
        delta_summary.to_string(index=False),
        "",
        "[VALIDATION -> TEST]",
        vt_df.to_string(index=False),
        "",
        "Interpretation:",
        "- G2 relational gain is supported only if the paired TEST delta remains < 0.",
        "- No K/X/architecture/graph-policy change is permitted after this TEST.",
        "",
        "STATUS: FROZEN TEST EVALUATED.",
    ]

    text = "\n".join(lines)
    (OUT / "g2_test_summary.txt").write_text(
        text, encoding="utf-8"
    )

    print("[4/4] DONE")
    print(text)


if __name__ == "__main__":
    main()
