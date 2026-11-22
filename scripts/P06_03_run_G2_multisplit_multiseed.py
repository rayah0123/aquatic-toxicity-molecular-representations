#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run frozen G2 production validation over 5 structural splits x 5 model seeds."""

from __future__ import annotations

import argparse, json, subprocess, sys
from pathlib import Path
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--trainer", default="P06_02_train_G2_validation.py")
    p.add_argument("--x_raw", required=True)
    p.add_argument("--rowmap_csv", required=True)
    p.add_argument("--y_csv", required=True)
    p.add_argument("--split_root", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--split_seeds", nargs="+", type=int, default=[42,123,777,2025,3407])
    p.add_argument("--model_seeds", nargs="+", type=int, default=[42,123,777,2025,3407])
    p.add_argument("--conditions", nargs="+", choices=["g2","self"], default=["g2","self"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--skip_existing", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    trainer = Path(args.trainer)
    if not trainer.is_absolute():
        trainer = (Path(__file__).resolve().parent / trainer).resolve()

    rows = []
    total = len(args.split_seeds)*len(args.model_seeds)*len(args.conditions)
    done = 0

    for split_seed in args.split_seeds:
        split_path = Path(args.split_root).resolve()/f"split_seed{split_seed}"/"split_membership.csv"
        if not split_path.exists():
            raise FileNotFoundError(split_path)

        for model_seed in args.model_seeds:
            for condition in args.conditions:
                run_name = f"{condition}_split{split_seed}_seed{model_seed}"
                run_dir = out/run_name
                result_json = run_dir/"result.json"

                if args.skip_existing and result_json.exists():
                    result = json.loads(result_json.read_text(encoding="utf-8"))
                else:
                    run_dir.mkdir(parents=True, exist_ok=True)
                    cmd = [
                        sys.executable, str(trainer),
                        "--x_raw", str(Path(args.x_raw).resolve()),
                        "--rowmap_csv", str(Path(args.rowmap_csv).resolve()),
                        "--y_csv", str(Path(args.y_csv).resolve()),
                        "--split_membership", str(split_path),
                        "--out_dir", str(run_dir),
                        "--run_name", run_name,
                        "--split_seed", str(split_seed),
                        "--model_seed", str(model_seed),
                        "--condition", condition,
                        "--device", args.device,
                    ]
                    subprocess.run(cmd, check=True)
                    result = json.loads(result_json.read_text(encoding="utf-8"))

                rows.append(result)
                done += 1
                print(f"{int(round(100*done/total))}%", end=" " if done<total else "\n", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(out/"g2_validation_all_runs.csv", index=False, encoding="utf-8-sig")

    metric_cols = ["val_RMSE","val_MAE","val_R2","val_Pearson","val_Spearman"]
    split_means = df.groupby(["condition","split_seed"], as_index=False)[metric_cols].mean()
    counts = df.groupby(["condition","split_seed"], as_index=False)[["n_total","n_train","n_val","n_test"]].first()
    split_means = counts.merge(split_means, on=["condition","split_seed"], how="left")
    split_means.to_csv(out/"g2_validation_splitmeans.csv", index=False, encoding="utf-8-sig")

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
        out/"g2_validation_hierarchical_summary.csv",
        index=False, encoding="utf-8-sig"
    )

    manifest = df[[
        "run_name","condition","split_seed","model_seed",
        "n_total","n_train","n_val","n_test",
        "best_epoch","val_RMSE","val_MAE","val_R2","val_Pearson","val_Spearman",
        "bundle_path","bundle_sha256","schema_hash","test_evaluated",
    ]].copy()
    manifest["bundle_path"] = manifest["bundle_path"].map(lambda x: str(Path(x).resolve()))
    manifest.to_csv(out/"g2_model_manifest.csv", index=False, encoding="utf-8-sig")

    print(f"[DONE] {out/'g2_model_manifest.csv'}")
    print("[TEST evaluated] NO")


if __name__ == "__main__":
    main()
