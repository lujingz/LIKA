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
DEFAULT_OUTPUT_DIR = os.environ.get("LIKA_FDR_OUTPUT_DIR")

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
    parser.add_argument(
        "--output-dir",
        dest="results_dir",
        metavar="OUTPUT_DIR",
        default=Path(DEFAULT_OUTPUT_DIR) if DEFAULT_OUTPUT_DIR else None,
        type=Path,
        help="Optional directory for generated intermediate CSV/PNG/PDF outputs.",
    )
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
    parser.add_argument(
        "--use-existing-source-data",
        action="store_true",
        help=(
            "Regenerate heatmaps from existing manuscript source data instead of refitting LIKA. "
            "If the manuscript source data is missing, this can also read FDR sensitivity CSVs from output_dir."
        ),
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


def find_existing_result(results_dir, *stems):
    candidates = []
    for stem in stems:
        candidates.append(results_dir / f"{stem}.csv")
        candidates.extend(sorted(results_dir.glob(f"{stem}_*.csv"), key=lambda path: path.stat().st_mtime, reverse=True))
    for path in candidates:
        if path.exists():
            return path
    return None


def normalize_matrix(matrix):
    matrix = matrix.copy()
    matrix.index = [f"{float(value):g}" for value in matrix.index]
    matrix.columns = [f"{float(value):g}" for value in matrix.columns]
    return matrix.astype(float)


def read_cached_matrix(path):
    return normalize_matrix(pd.read_csv(path, index_col=0))


def source_data_from_matrices(ratios, counts, jaccards, top_n):
    rows = []
    for dataset_name, ratio in ratios.items():
        count = counts.get(dataset_name)
        jaccard = jaccards.get(dataset_name)
        if count is None:
            count = ratio * top_n
        if jaccard is None:
            jaccard = pd.DataFrame(np.nan, index=ratio.index, columns=ratio.columns)
        rows.extend(matrix_summary_rows(dataset_name, count, ratio, jaccard))
    return pd.DataFrame(rows)


def matrix_from_source_data(source_data, dataset_name, value_column):
    if value_column not in source_data.columns:
        return None
    dataset_rows = source_data[source_data["dataset"] == dataset_name]
    if dataset_rows.empty:
        return None
    matrix = dataset_rows.pivot(index="fdr_y", columns="fdr_x", values=value_column)
    return normalize_matrix(matrix)


def load_existing_results(results_dir, top_n):
    paths = {}
    ratios = {}
    counts = {}
    jaccards = {}

    for dataset_name in ["INKA", "SCZ"]:
        lower_name = dataset_name.lower()
        ratio_path = find_existing_result(
            results_dir,
            f"fdr_sensitivity_{dataset_name}_top{top_n}_overlap_ratio",
            f"fdr_sensitivity_{lower_name}_top{top_n}_overlap_ratio",
        )
        if ratio_path is None:
            raise FileNotFoundError(f"Could not find cached overlap-ratio CSV for {dataset_name} in {results_dir}")
        paths[f"{dataset_name}_ratio"] = ratio_path
        ratios[dataset_name] = read_cached_matrix(ratio_path)

        count_path = find_existing_result(
            results_dir,
            f"fdr_sensitivity_{dataset_name}_top{top_n}_overlap_counts",
            f"fdr_sensitivity_{lower_name}_top{top_n}_overlap_counts",
        )
        if count_path is not None:
            paths[f"{dataset_name}_counts"] = count_path
            counts[dataset_name] = read_cached_matrix(count_path)

        jaccard_path = find_existing_result(
            results_dir,
            f"fdr_sensitivity_{dataset_name}_top{top_n}_jaccard",
            f"fdr_sensitivity_{lower_name}_top{top_n}_jaccard",
        )
        if jaccard_path is not None:
            paths[f"{dataset_name}_jaccard"] = jaccard_path
            jaccards[dataset_name] = read_cached_matrix(jaccard_path)

    summary_path = find_existing_result(results_dir, "fdr_sensitivity_overlap_summary")
    if summary_path is not None:
        paths["summary"] = summary_path
        source_data = pd.read_csv(summary_path)
    else:
        source_data = source_data_from_matrices(ratios, counts, jaccards, top_n)

    for dataset_name in ["INKA", "SCZ"]:
        if dataset_name not in counts:
            matrix = matrix_from_source_data(source_data, dataset_name, "overlap_count")
            if matrix is not None:
                counts[dataset_name] = matrix
        if dataset_name not in jaccards:
            matrix = matrix_from_source_data(source_data, dataset_name, "jaccard")
            if matrix is not None:
                jaccards[dataset_name] = matrix

    top_path = find_existing_result(results_dir, "fdr_sensitivity_top10_lists")
    top_df = None
    if top_path is not None:
        paths["top10"] = top_path
        top_df = pd.read_csv(top_path)

    return ratios, counts, jaccards, source_data, top_df, paths


def load_existing_source_data(source_data_path):
    source_data = pd.read_csv(source_data_path)
    ratios = {}
    counts = {}
    jaccards = {}
    for dataset_name in ["INKA", "SCZ"]:
        ratio = matrix_from_source_data(source_data, dataset_name, "overlap_ratio")
        if ratio is None:
            raise ValueError(f"{source_data_path} does not contain overlap_ratio rows for {dataset_name}")
        ratios[dataset_name] = ratio

        count = matrix_from_source_data(source_data, dataset_name, "overlap_count")
        if count is not None:
            counts[dataset_name] = count

        jaccard = matrix_from_source_data(source_data, dataset_name, "jaccard")
        if jaccard is not None:
            jaccards[dataset_name] = jaccard

    return ratios, counts, jaccards, source_data, None, {"source_data": source_data_path}


def save_existing_results_outputs(args, ratios, counts, jaccards, source_data, top_df, used_paths):
    results_dir = args.results_dir
    top_n = args.top_n

    args.manuscript_source_data.parent.mkdir(parents=True, exist_ok=True)
    source_data.to_csv(args.manuscript_source_data, index=False)

    save_heatmap(ratios["INKA"], ratios["SCZ"], args.manuscript_figure)

    print("Used cached FDR sensitivity files:")
    for label, path in sorted(used_paths.items()):
        print(f"  {label}: {path}")
    if results_dir is not None:
        results_dir.mkdir(parents=True, exist_ok=True)
        for dataset_name in ["INKA", "SCZ"]:
            lower_name = dataset_name.lower()
            ratio = ratios[dataset_name]
            ratio.to_csv(results_dir / f"fdr_sensitivity_{lower_name}_top{top_n}_overlap_ratio.csv")
            if dataset_name in counts:
                counts[dataset_name].to_csv(results_dir / f"fdr_sensitivity_{lower_name}_top{top_n}_overlap_counts.csv")
            if dataset_name in jaccards:
                jaccards[dataset_name].to_csv(results_dir / f"fdr_sensitivity_{lower_name}_top{top_n}_jaccard.csv")
        source_data.to_csv(results_dir / "fdr_sensitivity_overlap_summary.csv", index=False)
        if top_df is not None:
            top_df.to_csv(results_dir / "fdr_sensitivity_top10_lists.csv", index=False)
        heatmap_pdf = results_dir / "fdr_sensitivity_top10_overlap_heatmap.pdf"
        heatmap_png = results_dir / "fdr_sensitivity_top10_overlap_heatmap.png"
        save_heatmap(ratios["INKA"], ratios["SCZ"], heatmap_pdf, heatmap_png)
        print(f"Wrote {results_dir / 'fdr_sensitivity_overlap_summary.csv'}")
        if top_df is not None:
            print(f"Wrote {results_dir / 'fdr_sensitivity_top10_lists.csv'}")
        print(f"Wrote {heatmap_pdf}")
        print(f"Wrote {heatmap_png}")
    print(f"Wrote {args.manuscript_source_data}")
    print(f"Wrote {args.manuscript_figure}")


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

    if args.use_existing_source_data:
        if args.manuscript_source_data.exists():
            ratios, counts, jaccards, source_data, top_df, used_paths = load_existing_source_data(args.manuscript_source_data)
        else:
            if results_dir is None:
                raise FileNotFoundError(
                    f"{args.manuscript_source_data} does not exist. Provide --output-dir with cached FDR CSVs "
                    "or rerun without --use-existing-source-data."
                )
            ratios, counts, jaccards, source_data, top_df, used_paths = load_existing_results(results_dir, args.top_n)
        save_existing_results_outputs(args, ratios, counts, jaccards, source_data, top_df, used_paths)
        return

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

    ratios = {}
    summary_rows = []
    for dataset_name in ["INKA", "SCZ"]:
        counts, ratio, jaccard = overlap_matrices(all_top, dataset_name, fdr_levels, args.top_n)
        ratios[dataset_name] = ratio
        summary_rows.extend(matrix_summary_rows(dataset_name, counts, ratio, jaccard))
        if results_dir is not None:
            results_dir.mkdir(parents=True, exist_ok=True)
            counts.to_csv(results_dir / f"fdr_sensitivity_{dataset_name.lower()}_top10_overlap_counts.csv")
            ratio.to_csv(results_dir / f"fdr_sensitivity_{dataset_name.lower()}_top10_overlap_ratio.csv")
            jaccard.to_csv(results_dir / f"fdr_sensitivity_{dataset_name.lower()}_top10_jaccard.csv")

    source_data = pd.DataFrame(summary_rows)
    args.manuscript_source_data.parent.mkdir(parents=True, exist_ok=True)
    source_data.to_csv(args.manuscript_source_data, index=False)

    save_heatmap(ratios["INKA"], ratios["SCZ"], args.manuscript_figure)

    if results_dir is not None:
        results_dir.mkdir(parents=True, exist_ok=True)
        all_rankings.to_csv(results_dir / "fdr_sensitivity_all_rankings.csv", index=False)
        all_top.to_csv(results_dir / "fdr_sensitivity_top10_lists.csv", index=False)
        source_data.to_csv(results_dir / "fdr_sensitivity_overlap_summary.csv", index=False)
        heatmap_pdf = results_dir / "fdr_sensitivity_top10_overlap_heatmap.pdf"
        heatmap_png = results_dir / "fdr_sensitivity_top10_overlap_heatmap.png"
        save_heatmap(ratios["INKA"], ratios["SCZ"], heatmap_pdf, heatmap_png)
        print(f"Wrote {results_dir / 'fdr_sensitivity_all_rankings.csv'}")
        print(f"Wrote {results_dir / 'fdr_sensitivity_top10_lists.csv'}")
        print(f"Wrote {results_dir / 'fdr_sensitivity_overlap_summary.csv'}")
        print(f"Wrote {heatmap_pdf}")
        print(f"Wrote {heatmap_png}")
    print(f"Wrote {args.manuscript_source_data}")
    print(f"Wrote {args.manuscript_figure}")


if __name__ == "__main__":
    main()
