#!/usr/bin/env python
"""
Run the first-stage FDR sensitivity analysis for the LIKA manuscript.

This script intentionally lives in `scripts/` because it is a manuscript
analysis driver, not part of the core LIKA algorithm implementation.
"""

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/lika-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/lika-cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from method import BH_, get_kinase_ranking_new, get_pvalue_through_empirical_bayes
from pipeline import (
    build_lika_network,
    calculate_lika_influence_scores,
    get_rejection_pvalue_cutoff,
    rank_kinase_results,
)

DEFAULT_FDR_LEVELS = (0.01, 0.025, 0.05, 0.1, 0.2)
DEFAULT_TOP_N = 10

mpl.rcParams.update(
    {
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.dpi": 300,
        "savefig.dpi": 300,
    }
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fdr-levels",
        nargs="+",
        type=float,
        default=list(DEFAULT_FDR_LEVELS),
        help="First-stage substrate FDR thresholds to evaluate.",
    )
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N, help="Top-N LIKA rankings to compare.")
    parser.add_argument("--results-dir", default=ROOT / "results", type=Path, help="Directory for generated CSV/PNG/PDF outputs.")
    parser.add_argument(
        "--manuscript-figure",
        default=ROOT / "manuscript" / "figures" / "supplementary" / "supplementary_figure_fdr_sensitivity_top10_overlap.pdf",
        type=Path,
        help="Supplementary manuscript PDF path for the heatmap.",
    )
    parser.add_argument(
        "--manuscript-source-data",
        default=ROOT / "manuscript" / "supplementary_tables" / "supplementary_figure_fdr_sensitivity_top10_overlap_source_data.csv",
        type=Path,
        help="CSV source data for the supplementary sensitivity heatmap.",
    )
    return parser.parse_args()


def run_dataset(dataset_name, intensity_path, log_transform, fdr_levels, top_n):
    intensity_df = pd.read_csv(intensity_path)
    intensity_df.columns = intensity_df.columns.str.lower()

    substrate_p_values, logfc = get_pvalue_through_empirical_bayes(intensity_df, log_transform=log_transform)
    network = build_lika_network(substrate_p_values, logfc, rejection_set=[])
    metrics = calculate_lika_influence_scores(network)

    ranking_frames = []
    top_rows = []

    for fdr in fdr_levels:
        rejection_set = BH_(substrate_p_values.copy(), fdr)
        fixed_pvalue_cutoff = get_rejection_pvalue_cutoff(substrate_p_values, rejection_set)
        kinase_p_values, test_statistics = get_kinase_ranking_new(
            network,
            rejection_set,
            p_fixed=fixed_pvalue_cutoff,
        )

        ranked = pd.DataFrame(
            {
                "Name": list(kinase_p_values.keys()),
                "p_value": list(kinase_p_values.values()),
                "test_statistics": [test_statistics[name] for name in kinase_p_values],
            }
        ).merge(metrics, on="Name", how="left")
        ranked = rank_kinase_results(ranked)
        ranked = ranked.rename(
            columns={
                "p_value": "LIKA_p_value",
                "test_statistics": "LIKA_test_statistics",
                "ranking_p_value": "LIKA_p_value_capped",
                "LIKA_rank": "LIKA_ranking",
                "influence_score": "LIKA_influence_score",
            }
        )
        ranked["dataset"] = dataset_name
        ranked["first_stage_fdr"] = fdr
        ranked["rejection_count"] = len(rejection_set)
        ranked["fixed_pvalue_cutoff"] = fixed_pvalue_cutoff
        ranking_frames.append(ranked)

        for rank, kinase in enumerate(ranked.head(top_n)["Name"], start=1):
            top_rows.append(
                {
                    "dataset": dataset_name,
                    "first_stage_fdr": fdr,
                    "rank": rank,
                    "Name": kinase,
                    "rejection_count": len(rejection_set),
                    "fixed_pvalue_cutoff": fixed_pvalue_cutoff,
                }
            )

        print(
            f"{dataset_name} FDR={fdr:g}: {len(rejection_set)} rejected substrates, "
            f"fixed cutoff={fixed_pvalue_cutoff:.6g}, top1={ranked.iloc[0]['Name']}"
        )

    return pd.concat(ranking_frames, ignore_index=True), pd.DataFrame(top_rows)


def overlap_matrices(top_df, dataset_name, fdr_levels, top_n):
    labels = [f"{level:g}" for level in fdr_levels]
    top_sets = {
        level: set(top_df[(top_df["dataset"] == dataset_name) & (top_df["first_stage_fdr"] == level)]["Name"])
        for level in fdr_levels
    }
    counts = pd.DataFrame(index=labels, columns=labels, dtype=int)
    ratios = pd.DataFrame(index=labels, columns=labels, dtype=float)
    jaccard = pd.DataFrame(index=labels, columns=labels, dtype=float)

    for level_i in fdr_levels:
        for level_j in fdr_levels:
            set_i = top_sets[level_i]
            set_j = top_sets[level_j]
            intersection = len(set_i & set_j)
            union = len(set_i | set_j)
            i_label = f"{level_i:g}"
            j_label = f"{level_j:g}"
            counts.loc[i_label, j_label] = intersection
            ratios.loc[i_label, j_label] = intersection / top_n
            jaccard.loc[i_label, j_label] = intersection / union if union else np.nan
    return counts, ratios, jaccard


def matrix_summary_rows(dataset_name, counts, ratios, jaccard):
    rows = []
    for fdr_y in ratios.index:
        for fdr_x in ratios.columns:
            rows.append(
                {
                    "dataset": dataset_name,
                    "fdr_x": fdr_x,
                    "fdr_y": fdr_y,
                    "overlap_count": counts.loc[fdr_y, fdr_x],
                    "overlap_ratio": ratios.loc[fdr_y, fdr_x],
                    "jaccard": jaccard.loc[fdr_y, fdr_x],
                }
            )
    return rows


def save_heatmap(inka_ratio, scz_ratio, output_pdf, output_png=None):
    sns.set_theme(context="talk", style="white", font_scale=1.2)
    cmap = LinearSegmentedColormap.from_list("lika_blues", ["white", "#0072B2"], N=100)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    for ax, ratio, title in [
        (ax1, inka_ratio, "INKA Dataset\nTop-10 LIKA Ranking Overlap Across FDR Levels"),
        (ax2, scz_ratio, "SCZ Dataset\nTop-10 LIKA Ranking Overlap Across FDR Levels"),
    ]:
        sns.heatmap(
            ratio,
            annot=True,
            fmt=".2f",
            cmap=cmap,
            vmin=0,
            vmax=1,
            cbar_kws={"label": "Top-10 Overlap Ratio"},
            ax=ax,
        )
        ax.set_xlabel("First-stage FDR Level")
        ax.set_ylabel("First-stage FDR Level")
        ax.set_title(title)

    fig.tight_layout()
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf, bbox_inches="tight")
    if output_png is not None:
        output_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_png, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    fdr_levels = tuple(args.fdr_levels)
    results_dir = args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    inka_rankings, inka_top = run_dataset(
        "INKA",
        ROOT / "data" / "intensity_data_INKA.csv",
        log_transform=True,
        fdr_levels=fdr_levels,
        top_n=args.top_n,
    )
    scz_rankings, scz_top = run_dataset(
        "SCZ",
        ROOT / "data" / "residual_data_SCZ.csv",
        log_transform=False,
        fdr_levels=fdr_levels,
        top_n=args.top_n,
    )

    all_rankings = pd.concat([inka_rankings, scz_rankings], ignore_index=True)
    all_top = pd.concat([inka_top, scz_top], ignore_index=True)
    all_rankings.to_csv(results_dir / "fdr_sensitivity_all_rankings.csv", index=False)
    all_top.to_csv(results_dir / "fdr_sensitivity_top10_lists.csv", index=False)

    ratios = {}
    summary_rows = []
    for dataset_name in ["INKA", "SCZ"]:
        counts, ratio, jaccard = overlap_matrices(all_top, dataset_name, fdr_levels, args.top_n)
        ratios[dataset_name] = ratio
        counts.to_csv(results_dir / f"fdr_sensitivity_{dataset_name.lower()}_top10_overlap_counts.csv")
        ratio.to_csv(results_dir / f"fdr_sensitivity_{dataset_name.lower()}_top10_overlap_ratio.csv")
        jaccard.to_csv(results_dir / f"fdr_sensitivity_{dataset_name.lower()}_top10_jaccard.csv")
        summary_rows.extend(matrix_summary_rows(dataset_name, counts, ratio, jaccard))

    source_data = pd.DataFrame(summary_rows)
    source_data.to_csv(results_dir / "fdr_sensitivity_overlap_summary.csv", index=False)
    args.manuscript_source_data.parent.mkdir(parents=True, exist_ok=True)
    source_data.to_csv(args.manuscript_source_data, index=False)

    heatmap_pdf = results_dir / "fdr_sensitivity_top10_overlap_heatmap.pdf"
    heatmap_png = results_dir / "fdr_sensitivity_top10_overlap_heatmap.png"
    save_heatmap(ratios["INKA"], ratios["SCZ"], heatmap_pdf, heatmap_png)
    save_heatmap(ratios["INKA"], ratios["SCZ"], args.manuscript_figure)

    print(f"Wrote {results_dir / 'fdr_sensitivity_all_rankings.csv'}")
    print(f"Wrote {results_dir / 'fdr_sensitivity_top10_lists.csv'}")
    print(f"Wrote {results_dir / 'fdr_sensitivity_overlap_summary.csv'}")
    print(f"Wrote {args.manuscript_source_data}")
    print(f"Wrote {heatmap_pdf}")
    print(f"Wrote {heatmap_png}")
    print(f"Wrote {args.manuscript_figure}")


if __name__ == "__main__":
    main()
