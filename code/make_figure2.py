#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Generate Figure 2 from the saved splitwise G2-versus-descriptor-only metrics.

Each point represents one structural split after averaging five model seeds.
Diamonds and whiskers show the mean and SD across five splits.
Positive differences indicate better G2 performance:
RMSE/MAE = descriptor-only minus G2;
R2/Pearson/Spearman = G2 minus descriptor-only.

Input: 05_G2_vs_self_splitwise_all_metrics.csv
Usage: python make_figure2.py --input <splitwise_metrics.csv> --outdir <output_dir>
No models are fitted and no predictions are averaged by this plotting script."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


INPUT_BASENAME = "05_G2_vs_self_splitwise_all_metrics.csv"

CONTEXT_ORDER = [
    "Daphnia_TEST",
    "ADORE_external",
    "Fathead_TEST",
    "WebTEST_surrogate",
]

CONTEXT_LABELS = {
    "Daphnia_TEST": r"$D.$ magna TEST",
    "ADORE_external": "ADORE external",
    "Fathead_TEST": "Fathead minnow TEST",
    "WebTEST_surrogate": "WebTEST surrogate",
}

METRICS = [
    ("RMSE", r"$\Delta$RMSE", "G2_advantage_RMSE"),
    ("MAE", r"$\Delta$MAE", "G2_advantage_MAE"),
    ("R2", r"$\Delta R^2$", "G2_advantage_R2"),
    ("Pearson", r"$\Delta$Pearson $r$", "G2_advantage_Pearson"),
    ("Spearman", r"$\Delta$Spearman $\rho$", "G2_advantage_Spearman"),
]

PANEL_LABELS = ["a", "b", "c", "d", "e"]


def mm_to_inch(mm: float) -> float:
    return mm / 25.4


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, default=None)
    p.add_argument("--outdir", type=Path, default=None)
    p.add_argument("--dpi", type=int, default=600)
    p.add_argument("--width-mm", type=float, default=183.0)
    p.add_argument("--height-mm", type=float, default=94.0)
    return p.parse_args()


def resolve_input(explicit: Path | None) -> Path:
    if explicit is not None:
        p = explicit.expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"Input file not foun<local-path>")
        return p

    cwd = Path.cwd().resolve()
    script_dir = Path(__file__).resolve().parent

    roots = []
    for p in [cwd, script_dir, script_dir.parent, cwd.parent]:
        p = p.resolve()
        if p.exists() and p not in roots:
            roots.append(p)

    for root in roots:
        direct = root / INPUT_BASENAME
        if direct.exists():
            print(f"[AUTO] Found input: {direct}")
            return direct.resolve()

    found, seen = [], set()
    for root in roots:
        try:
            for p in root.rglob(INPUT_BASENAME):
                rp = p.resolve()
                if rp not in seen:
                    found.append(rp)
                    seen.add(rp)
        except (PermissionError, OSError):
            pass

    if len(found) == 1:
        print(f"[AUTO] Found input: {found[0]}")
        return found[0]

    if len(found) == 0:
        searched = "\n".join(f"  - {x}" for x in roots)
        raise FileNotFoundError(
            f"Could not find {INPUT_BASENAME}.\n"
            f"Searched recursively unde<local-path>"
            "Use --input to specify the exact frozen CSV."
        )

    choices = "\n".join(f"  - {x}" for x in found)
    raise RuntimeError(
        "Multiple candidate frozen CSVs were found.\n"
        "Specify the intended frozen result with --inpu<local-path>"
        f"{choices}"
    )


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    required = [
        "dataset", "split_seed",
        "G2_RMSE", "Self_RMSE",
        "G2_MAE", "Self_MAE",
        "G2_R2", "Self_R2",
        "G2_Pearson", "Self_Pearson",
        "G2_Spearman", "Self_Spearman",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    out = df.copy()

    out["G2_advantage_RMSE_calc"] = out["Self_RMSE"] - out["G2_RMSE"]
    out["G2_advantage_MAE_calc"] = out["Self_MAE"] - out["G2_MAE"]
    out["G2_advantage_R2_calc"] = out["G2_R2"] - out["Self_R2"]
    out["G2_advantage_Pearson_calc"] = out["G2_Pearson"] - out["Self_Pearson"]
    out["G2_advantage_Spearman_calc"] = out["G2_Spearman"] - out["Self_Spearman"]

    check_map = {
        "G2_advantage_RMSE": "G2_advantage_RMSE_calc",
        "G2_advantage_MAE": "G2_advantage_MAE_calc",
        "G2_advantage_R2": "G2_advantage_R2_calc",
        "G2_advantage_Pearson": "G2_advantage_Pearson_calc",
        "G2_advantage_Spearman": "G2_advantage_Spearman_calc",
    }

    for frozen, calc in check_map.items():
        if frozen in out.columns:
            a = pd.to_numeric(out[frozen], errors="coerce").to_numpy(float)
            b = pd.to_numeric(out[calc], errors="coerce").to_numpy(float)
            if not np.allclose(a, b, rtol=1e-9, atol=1e-10, equal_nan=True):
                max_diff = np.nanmax(np.abs(a - b))
                raise ValueError(
                    f"QC failure: {frozen} does not match recomputed values. "
                    f"max |difference| = {max_diff}"
                )
        out[frozen] = out[calc]

    if out.duplicated(["dataset", "split_seed"]).any():
        raise ValueError(
            "Duplicate dataset/split_seed rows detected. "
            "Expected one seed-averaged row per structural split."
        )

    counts = out.groupby("dataset")["split_seed"].nunique()
    for context in CONTEXT_ORDER:
        if counts.get(context, 0) != 5:
            raise ValueError(
                f"{context}: expected exactly 5 structural splits, "
                f"found {counts.get(context, 0)}"
            )

    return out


def make_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for context in CONTEXT_ORDER:
        sub = df[df["dataset"].eq(context)].sort_values("split_seed")
        for metric_key, metric_label, col in METRICS:
            x = sub[col].to_numpy(float)
            rows.append({
                "dataset": context,
                "dataset_label": CONTEXT_LABELS[context],
                "metric": metric_key,
                "metric_label": metric_label,
                "n_structural_splits": len(x),
                "difference_mean": x.mean(),
                "difference_sd": x.std(ddof=1),
                "difference_median": np.median(x),
                "n_G2_better": int((x > 0).sum()),
                "n_tie": int((x == 0).sum()),
                "n_descriptor_only_better": int((x < 0).sum()),
            })
    return pd.DataFrame(rows)


def nice_step(raw_step: float) -> float:
    if raw_step <= 0:
        return 0.01

    exponent = math.floor(math.log10(raw_step))
    fraction = raw_step / (10 ** exponent)

    candidates = [1.0, 1.25, 1.5, 2.0, 2.5, 4.0, 5.0, 8.0, 10.0]
    for c in candidates:
        if fraction <= c:
            return c * (10 ** exponent)

    return 10.0 * (10 ** exponent)


def symmetric_axis(values: np.ndarray):
    max_abs = float(np.nanmax(np.abs(values)))
    step = nice_step((max_abs * 1.08) / 2.0)
    bound = 2.0 * step
    ticks = np.array([-2, -1, 0, 1, 2], dtype=float) * step
    return (-bound, bound), ticks


def tick_formatter(x, pos):
    if abs(x) < 1e-12:
        return "0"
    ax = abs(x)
    if ax >= 1:
        return f"{x:.1f}"
    if ax >= 0.1:
        return f"{x:.2f}"
    return f"{x:.2f}"


def apply_publication_style():
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 6.5,
        "axes.titlesize": 7.0,
        "axes.labelsize": 6.5,
        "xtick.labelsize": 6.0,
        "ytick.labelsize": 6.2,
        "axes.linewidth": 0.55,
        "xtick.major.width": 0.50,
        "ytick.major.width": 0.50,
        "xtick.major.size": 2.3,
        "ytick.major.size": 2.3,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.transparent": False,
    })


def make_figure(
    df: pd.DataFrame,
    outdir: Path,
    dpi: int,
    width_mm: float,
    height_mm: float,
):
    apply_publication_style()

    fig = plt.figure(
        figsize=(mm_to_inch(width_mm), mm_to_inch(height_mm)),
        facecolor="white",
    )

    # Top: 3 equal panels; bottom: 2 equal panels centered.
    gs = fig.add_gridspec(
        2, 6,
        left=0.108,
        right=0.988,
        bottom=0.135,
        top=0.958,
        wspace=0.82,
        hspace=0.42,
    )

    axes = [
        fig.add_subplot(gs[0, 0:2]),
        fig.add_subplot(gs[0, 2:4]),
        fig.add_subplot(gs[0, 4:6]),
        fig.add_subplot(gs[1, 1:3]),
        fig.add_subplot(gs[1, 3:5]),
    ]

    y_map = {
        "Daphnia_TEST": 3,
        "ADORE_external": 2,
        "Fathead_TEST": 1,
        "WebTEST_surrogate": 0,
    }

    offsets = np.array([-0.095, -0.0475, 0.0, 0.0475, 0.095])

    split_edge = "0.55"
    mean_color = "0.12"
    zero_color = "0.68"

    for idx, (ax, (_, metric_title, col), panel) in enumerate(
        zip(axes, METRICS, PANEL_LABELS)
    ):
        ax.axvline(
            0.0,
            color=zero_color,
            linestyle=(0, (2.0, 2.0)),
            linewidth=0.45,
            zorder=0,
        )

        for context in CONTEXT_ORDER:
            sub = (
                df[df["dataset"].eq(context)]
                .sort_values("split_seed")
            )
            vals = sub[col].to_numpy(float)
            y0 = y_map[context]

            # Five structural-split values.
            ax.scatter(
                vals,
                y0 + offsets,
                s=8.0,
                facecolors="white",
                edgecolors=split_edge,
                linewidths=0.45,
                zorder=2,
            )

            # Mean +/- SD.
            mean = vals.mean()
            sd = vals.std(ddof=1)
            ax.errorbar(
                mean,
                y0,
                xerr=sd,
                fmt="D",
                markersize=3.8,
                markerfacecolor=mean_color,
                markeredgecolor=mean_color,
                markeredgewidth=0.0,
                ecolor=mean_color,
                elinewidth=0.60,
                capsize=1.8,
                capthick=0.60,
                zorder=3,
            )

        xlim, xticks = symmetric_axis(df[col].to_numpy(float))
        ax.set_xlim(*xlim)
        ax.set_xticks(xticks)
        ax.xaxis.set_major_formatter(FuncFormatter(tick_formatter))

        ax.set_ylim(-0.40, 3.40)
        ax.set_yticks([3, 2, 1, 0])

        ax.set_title(metric_title, pad=3.8, fontweight="normal")

        # Context labels only on left-most panel of each row.
        if idx in (0, 3):
            ax.set_yticklabels([
                CONTEXT_LABELS["Daphnia_TEST"],
                CONTEXT_LABELS["ADORE_external"],
                CONTEXT_LABELS["Fathead_TEST"],
                CONTEXT_LABELS["WebTEST_surrogate"],
            ])
            ax.tick_params(axis="y", left=True)
        else:
            ax.set_yticklabels([])
            ax.tick_params(axis="y", left=False)

        ax.grid(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Remove left spine from panels without y labels to reduce box-like repetition.
        if idx not in (0, 3):
            ax.spines["left"].set_visible(False)

        ax.text(
            -0.17,
            1.055,
            panel,
            transform=ax.transAxes,
            fontsize=8.0,
            fontweight="bold",
            va="top",
            ha="left",
        )

    # Minimal directional cue; "paired" terminology intentionally omitted.
    fig.text(
        0.50,
        0.048,
        "Descriptor-only better   \u2190   0   \u2192   G2 better",
        ha="center",
        va="center",
        fontsize=6.2,
    )

    png = outdir / "Figure2_relational_effect.png"
    pdf = outdir / "Figure2_relational_effect.pdf"
    svg = outdir / "Figure2_relational_effect.svg"

    fig.savefig(png, dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    fig.savefig(svg, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    return [png, pdf, svg]


def caption_text() -> str:
    return (
        "Figure 2. Context-dependent contribution of inter-chemical structural "
        "connectivity. Inter-chemical GraphSAGE (G2) and descriptor-only "
        "GraphSAGE were compared within each of five structural splits. "
        "Small open circles show split-level differences after metric values "
        "were first averaged across five model seeds within each structural "
        "split; filled diamonds and horizontal whiskers show the mean +/- SD "
        "across the five split-level means. Differences were signed so that "
        "positive values consistently indicate better performance with G2. "
        "For RMSE and MAE, differences were calculated as descriptor-only "
        "GraphSAGE minus G2; for R2, Pearson r, and Spearman rho, differences "
        "were calculated as G2 minus descriptor-only GraphSAGE."
    )


def main():
    args = parse_args()

    input_path = resolve_input(args.input)
    outdir = (
        args.outdir.expanduser().resolve()
        if args.outdir is not None
        else input_path.parent / "Figure2"
    )
    outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("Figure 2 | final within-split effect plot")
    print("=" * 78)
    print(f"Input : {input_path}")
    print(f"Output: {outdir}")
    print(f"Final size: {args.width_mm:.1f} x {args.height_mm:.1f} mm")

    raw = pd.read_csv(input_path)
    df = prepare_data(raw)

    plotting_cols = [
        "dataset", "split_seed",
        "G2_advantage_RMSE",
        "G2_advantage_MAE",
        "G2_advantage_R2",
        "G2_advantage_Pearson",
        "G2_advantage_Spearman",
    ]

    plot_data_path = outdir / "Figure2_plotting_data.csv"
    df[plotting_cols].to_csv(plot_data_path, index=False)

    summary = make_summary(df)
    summary_path = outdir / "Figure2_summary.csv"
    summary.to_csv(summary_path, index=False)

    fig_paths = make_figure(
        df,
        outdir,
        args.dpi,
        args.width_mm,
        args.height_mm,
    )

    caption_path = outdir / "Figure2_caption_draft.txt"
    caption_path.write_text(caption_text(), encoding="utf-8")

    print("\nQC | G2-better metric-split counts")
    counts = (
        summary.groupby("dataset")["n_G2_better"]
        .sum()
        .reindex(CONTEXT_ORDER)
    )
    for context, n in counts.items():
        print(f"  {context:18s}: {int(n):2d}/25")

    print("\nSaved:")
    for p in fig_paths + [plot_data_path, summary_path, caption_path]:
        print(f"  {p}")

    print("\nDONE")


if __name__ == "__main__":
    main()
