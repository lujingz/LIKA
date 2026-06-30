#!/usr/bin/env python
"""
Shared helpers for manuscript-facing tables and figures.

This module intentionally has no command-line interface. The executable scripts
in this directory split the reproducibility workflow into SCZ, INKA, simulation,
and FDR sensitivity settings while keeping manuscript-specific output code out
of the LIKA core implementation.
"""

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
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from method import BH_, get_pvalue_through_empirical_bayes
from pipeline import pipeline
from simulation_methods import KSEA, plot_precision_recall_summary

MANUSCRIPT_DIR = ROOT / "manuscript"
FIGURE_DIR = MANUSCRIPT_DIR / "figures" / "main"
SUPPLEMENTARY_FIGURE_DIR = MANUSCRIPT_DIR / "figures" / "supplementary"
TABLE_DIR = MANUSCRIPT_DIR / "supplementary_tables"

TOP_N = 10
INFLUENCE_COLUMN = "number_of_efficient_substrates"
INFLUENCE_LABEL = "LIKA Influence Score"
RANKING_CMAP = LinearSegmentedColormap.from_list(
    "lika_influence_blues",
    ["#dcecf7", "#7fb8d6", "#2b7bba", "#08306b"],
)

mpl.rcParams.update(
    {
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "figure.dpi": 100,
    }
)

SIMULATION_TABLE = TABLE_DIR / "supplementary_table_s1_simulation_network_structures.csv"
INKA_RANKING_TABLE = TABLE_DIR / "supplementary_table_s2_inka_kinase_rankings.csv"
SCZ_SUBSTRATE_TABLE = TABLE_DIR / "supplementary_table_s3_scz_substrate_statistics.csv"
INKA_SUBSTRATE_TABLE = TABLE_DIR / "supplementary_table_s4_inka_substrate_statistics.csv"
SCZ_RANKING_TABLE = TABLE_DIR / "supplementary_table_s5_scz_kinase_rankings.csv"
FIGURE3_SOURCE_DATA = TABLE_DIR / "figure_3_simulation_precision_recall_source_data.csv"


def ensure_dirs():
    for directory in [FIGURE_DIR, SUPPLEMENTARY_FIGURE_DIR, TABLE_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def write_simulation_structure_table(output_path=SIMULATION_TABLE):
    df = pd.DataFrame(
        [
            {
                "simulation": 1,
                "network_structure": "non-overlapping substrates with equal kinase degree",
                "number_of_kinases": 10,
                "number_of_substrates": 100,
                "kinase_substrate_counts": "10 substrates per kinase",
                "overlapping_substrates": False,
                "ground_truth_dysregulated_kinases": "K0;K1",
                "case_distribution_for_perturbed_substrates": "N(2, 1)",
                "control_distribution": "N(0, 1)",
                "case_distribution_for_unperturbed_substrates": "N(0, 1)",
            },
            {
                "simulation": 2,
                "network_structure": "non-overlapping substrates with unequal kinase degree",
                "number_of_kinases": 13,
                "number_of_substrates": 127,
                "kinase_substrate_counts": "20;20;15;15;12;11;10;8;6;4;3;2;1",
                "overlapping_substrates": False,
                "ground_truth_dysregulated_kinases": "K0;K2;K5;K8",
                "case_distribution_for_perturbed_substrates": "N(2, 1)",
                "control_distribution": "N(0, 1)",
                "case_distribution_for_unperturbed_substrates": "N(0, 1)",
            },
            {
                "simulation": 3,
                "network_structure": "overlapping substrates with unequal kinase degree",
                "number_of_kinases": 13,
                "number_of_substrates": 40,
                "kinase_substrate_counts": "20;20;15;15;12;11;10;8;6;4;3;2;1",
                "overlapping_substrates": True,
                "ground_truth_dysregulated_kinases": "K0;K2;K5;K8",
                "case_distribution_for_perturbed_substrates": "N(2, 1)",
                "control_distribution": "N(0, 1)",
                "case_distribution_for_unperturbed_substrates": "N(0, 1)",
            },
        ]
    )
    df.to_csv(output_path, index=False)
    return output_path


def load_default_network():
    network_df = pd.read_csv(ROOT / "data" / "KSEA_dataset_processed.csv")
    network_df.columns = network_df.columns.str.lower()
    return network_df


def split_phosphosite_name(name):
    gene = str(name).split("_", 1)[0]
    return gene, gene


def write_substrate_statistics(
    intensity_path,
    output_path,
    log_transform,
    include_all_substrates,
):
    intensity_df = pd.read_csv(intensity_path)
    intensity_df.columns = intensity_df.columns.str.lower()
    p_values, _ = get_pvalue_through_empirical_bayes(intensity_df, log_transform)
    rejection_set = set(BH_(p_values, 0.05))

    network_df = load_default_network()
    upstream = (
        network_df[network_df["to"].isin(p_values.keys())]
        .groupby("to")["from"]
        .apply(lambda values: ",".join(sorted(set(map(str, values)))))
    )

    if include_all_substrates:
        names = sorted(p_values)
    else:
        names = sorted(upstream.index)

    rows = []
    for name in names:
        protein_name, gene_name = split_phosphosite_name(name)
        in_analysis = int(name in upstream.index)
        rows.append(
            {
                "Phosphosite Name": name,
                "Protein Name": protein_name,
                "Gene Name": gene_name,
                "in Analysis": in_analysis,
                "p-value": p_values[name],
                "Significant": int(name in rejection_set),
                "Upstream Kinase": upstream.get(name, np.nan),
            }
        )
    pd.DataFrame(rows).to_csv(output_path, index=False)
    return output_path


def downstream_substrates(network):
    rows = []
    for kinase in network.get_kinase():
        children = sorted(set(network.get_unmissing_children(kinase)))
        rows.append({"Name": kinase, "downstream phosphosites": ";".join(children)})
    return pd.DataFrame(rows)


def observed_raw_network_metrics(substrate_p_values):
    edge_df = load_default_network()
    edge_df = edge_df[edge_df["to"].isin(substrate_p_values.keys())].copy()
    parent_counts = edge_df.groupby("to")["from"].nunique()

    rows = []
    for kinase, kinase_edges in edge_df.groupby("from", sort=True):
        children = sorted(set(kinase_edges["to"]))
        influence_score = sum(1.0 / parent_counts[child] for child in children)
        rows.append(
            {
                "Name": kinase,
                "raw_number_of_substrates": len(children),
                "raw_number_of_efficient_substrates": influence_score,
                "raw_downstream phosphosites": ";".join(children),
            }
        )
    return pd.DataFrame(rows)


def write_kinase_ranking_table(intensity_path, output_path, log_transform):
    intensity_df = pd.read_csv(intensity_path)
    network, _, lika_df, _, substrate_p_values = pipeline(intensity_df, log_transform=log_transform, CI=None)
    ksea_df, _ = KSEA(intensity_df, log_transform=log_transform)

    lika = lika_df.rename(
        columns={
            "p_value": "LIKA_p_value",
            "test_statistics": "LIKA_test_statistics_fixed",
            "ranking_p_value": "LIKA_p_value_capped",
            "LIKA_rank": "LIKA_ranking",
        }
    )
    if "number_of_efficient_substrates" not in lika.columns and "influence_score" in lika.columns:
        lika["number_of_efficient_substrates"] = lika["influence_score"]

    lika_columns = [
        "Name",
        "LIKA_p_value",
        "LIKA_ranking",
        "number_of_substrates",
        "number_of_efficient_substrates",
        "LIKA_test_statistics_fixed",
        "LIKA_p_value_capped",
    ]
    lika = lika[[col for col in lika_columns if col in lika.columns]]

    combined = ksea_df[["Name", "p_value"]].rename(columns={"p_value": "KSEA_p_value"}).merge(
        lika,
        on="Name",
        how="outer",
    )
    combined = combined.merge(observed_raw_network_metrics(substrate_p_values), on="Name", how="left")
    combined = combined.merge(downstream_substrates(network), on="Name", how="left")

    combined["number_of_substrates"] = combined.get("number_of_substrates").combine_first(
        combined["raw_number_of_substrates"]
    )
    combined["number_of_efficient_substrates"] = combined.get("number_of_efficient_substrates").combine_first(
        combined["raw_number_of_efficient_substrates"]
    )
    combined["downstream phosphosites"] = combined.get("downstream phosphosites").combine_first(
        combined["raw_downstream phosphosites"]
    )

    combined["KSEA_ranking"] = pd.NA
    ksea_sorted = combined[combined["KSEA_p_value"].notna()].sort_values(
        ["KSEA_p_value", "Name"],
        ascending=[True, True],
    )
    combined.loc[ksea_sorted.index, "KSEA_ranking"] = range(1, len(ksea_sorted) + 1)
    combined["KSEA_ranking"] = combined["KSEA_ranking"].astype("Int64")
    if "LIKA_ranking" in combined.columns:
        combined["LIKA_ranking"] = combined["LIKA_ranking"].astype("Int64")
    combined = combined.sort_values(["LIKA_ranking", "KSEA_ranking", "Name"], na_position="last").reset_index(drop=True)

    output_columns = [
        "Name",
        "KSEA_p_value",
        "LIKA_p_value",
        "LIKA_ranking",
        "KSEA_ranking",
        "number_of_substrates",
        "number_of_efficient_substrates",
        "LIKA_test_statistics_fixed",
        "LIKA_p_value_capped",
    ]
    if "downstream phosphosites" in combined.columns:
        output_columns.insert(-2, "downstream phosphosites")
    combined[[col for col in output_columns if col in combined.columns]].to_csv(output_path, index=False)
    return output_path


def finite_log_p(all_pvalues, top_pvalues):
    all_pvalues = pd.to_numeric(all_pvalues, errors="coerce")
    top_pvalues = pd.to_numeric(top_pvalues, errors="coerce")
    positive = all_pvalues[all_pvalues > 0]
    floor = positive.min() / 10 if len(positive) else 1e-300
    floored = top_pvalues.mask(top_pvalues <= 0, floor)
    return -np.log10(floored), floor


def norm_for(values):
    vmin = float(np.nanmin(values))
    vmax = float(np.nanmax(values))
    if np.isclose(vmin, vmax):
        return Normalize(vmin=vmin - 0.5, vmax=vmax + 0.5)
    return Normalize(vmin=vmin, vmax=vmax)


def pvalue_label(pvalue_column):
    if pvalue_column == "LIKA_p_value_capped":
        return "capped LIKA p-value"
    if pvalue_column.startswith("LIKA"):
        return "LIKA p-value"
    if pvalue_column.startswith("KSEA"):
        return "KSEA p-value"
    return pvalue_column.replace("_", " ")


def ranking_plot(
    ranking_table,
    output_path,
    title,
    rank_column,
    pvalue_column,
    top_n=TOP_N,
    highlight_y_labels=None,
    show_zero_pvalue_note=True,
):
    df = pd.read_csv(ranking_table)
    df = df.sort_values(rank_column).head(top_n).copy()
    df[pvalue_column] = pd.to_numeric(df[pvalue_column], errors="coerce")
    df[INFLUENCE_COLUMN] = pd.to_numeric(df[INFLUENCE_COLUMN], errors="coerce")
    df = df.dropna(subset=[pvalue_column, INFLUENCE_COLUMN])
    color_values, floor = finite_log_p(pd.read_csv(ranking_table)[pvalue_column], df[pvalue_column])
    norm = norm_for(color_values)
    colors = RANKING_CMAP(norm(color_values))

    fig, ax = plt.subplots(figsize=(6.8, 7.6))
    y = np.arange(len(df))
    ax.barh(y, df[INFLUENCE_COLUMN], color=colors)
    ax.set_yticks(y, df["Name"])
    highlighted = set(highlight_y_labels or [])
    for tick in ax.get_yticklabels():
        if tick.get_text() in highlighted:
            tick.set_fontweight("bold")
            tick.set_bbox(
                {
                    "boxstyle": "circle,pad=0.28",
                    "facecolor": "none",
                    "edgecolor": "black",
                    "linewidth": 1.4,
                }
            )
    ax.tick_params(axis="both", labelsize=10)
    ax.invert_yaxis()
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel(INFLUENCE_LABEL, fontsize=12)
    ax.grid(axis="x", color="#d9d9d9", linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    sm = mpl.cm.ScalarMappable(cmap=RANKING_CMAP, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(f"-log10({pvalue_label(pvalue_column)})")
    cbar.ax.tick_params(labelsize=10)
    if show_zero_pvalue_note and (df[pvalue_column] <= 0).any():
        ax.text(
            0.0,
            -0.09,
            f"0 p-values plotted at {floor:.1e}",
            transform=ax.transAxes,
            fontsize=8,
            ha="left",
            va="top",
        )
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def figure_5_subnetworks(output_path=FIGURE_DIR / "figure_5_abl1_egfr_subnetworks.pdf"):
    substrate_df = pd.read_csv(INKA_SUBSTRATE_TABLE)
    kinases = ["ABL1", "EGFR"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    for ax, kinase in zip(axes, kinases):
        kinase_rows = substrate_df[
            substrate_df["Upstream Kinase"].fillna("").str.split(",").apply(lambda values: kinase in values)
        ].copy()
        kinase_rows = kinase_rows.sort_values("p-value")
        label_nodes = set(kinase_rows.head(5)["Phosphosite Name"])

        graph = nx.Graph()
        graph.add_node(kinase, kind="kinase")
        for _, row in kinase_rows.iterrows():
            substrate = row["Phosphosite Name"]
            graph.add_node(
                substrate,
                kind="substrate",
                significant=bool(row["Significant"]),
                label=substrate if substrate in label_nodes else "",
            )
            graph.add_edge(kinase, substrate)

        pos = nx.spring_layout(graph, seed=7, k=0.55)
        node_colors = []
        node_sizes = []
        labels = {kinase: kinase}
        for node, attrs in graph.nodes(data=True):
            if attrs.get("kind") == "kinase":
                node_colors.append("#E69F00")
                node_sizes.append(700)
            elif attrs.get("significant"):
                node_colors.append("#F0E442")
                node_sizes.append(90)
                if attrs.get("label"):
                    labels[node] = attrs["label"]
            else:
                node_colors.append("#222222")
                node_sizes.append(70)

        nx.draw_networkx_edges(graph, pos, ax=ax, edge_color="#999999", width=0.6, alpha=0.7)
        nx.draw_networkx_nodes(graph, pos, ax=ax, node_color=node_colors, node_size=node_sizes, linewidths=0.3)
        nx.draw_networkx_labels(graph, pos, labels=labels, ax=ax, font_size=7)
        ax.set_title(f"{kinase} subnetwork")
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def figure_3_from_summary(
    summary_path=FIGURE3_SOURCE_DATA,
    output_path=FIGURE_DIR / "figure_3_simulation_precision_recall.pdf",
):
    if not summary_path.exists():
        raise FileNotFoundError(
            f"{summary_path} does not exist. Run `python scripts/generate_simulation_assets.py --runs 100 --max-k 10` first."
        )
    summary_df = pd.read_csv(summary_path)
    plot_precision_recall_summary(summary_df, output_path)
    return output_path


def ranking_pvalue_column(ranking_table, preferred_column):
    columns = pd.read_csv(ranking_table, nrows=1).columns
    if preferred_column in columns:
        return preferred_column
    return "LIKA_p_value"


def generate_inka_figures():
    return [
        ranking_plot(
            INKA_RANKING_TABLE,
            FIGURE_DIR / "figure_4a_inka_ksea_pvalue_ranking.pdf",
            "INKA Dataset KSEA p-value ranking",
            "KSEA_ranking",
            "KSEA_p_value",
            highlight_y_labels=["ABL1", "EGFR"],
            show_zero_pvalue_note=False,
        ),
        ranking_plot(
            INKA_RANKING_TABLE,
            FIGURE_DIR / "figure_4b_inka_lika_ranking.pdf",
            "INKA Dataset LIKA ranking",
            "LIKA_ranking",
            ranking_pvalue_column(INKA_RANKING_TABLE, "LIKA_p_value_capped"),
            highlight_y_labels=["ABL1", "EGFR"],
            show_zero_pvalue_note=False,
        ),
        figure_5_subnetworks(),
    ]


def generate_scz_figures():
    return [
        ranking_plot(
            SCZ_RANKING_TABLE,
            FIGURE_DIR / "figure_6a_scz_ksea_pvalue_ranking.pdf",
            "SCZ Dataset KSEA p-value ranking",
            "KSEA_ranking",
            "KSEA_p_value",
        ),
        ranking_plot(
            SCZ_RANKING_TABLE,
            FIGURE_DIR / "figure_6b_scz_lika_ranking.pdf",
            "SCZ Dataset LIKA ranking",
            "LIKA_ranking",
            ranking_pvalue_column(SCZ_RANKING_TABLE, "LIKA_p_value_capped"),
        ),
    ]


def audit_outputs():
    checks = {
        "S1 simulation table": SIMULATION_TABLE,
        "S2 INKA kinase rankings": INKA_RANKING_TABLE,
        "S3 SCZ substrate statistics": SCZ_SUBSTRATE_TABLE,
        "S4 INKA substrate statistics": INKA_SUBSTRATE_TABLE,
        "S5 SCZ kinase rankings": SCZ_RANKING_TABLE,
        "Figure 3 source data": FIGURE3_SOURCE_DATA,
        "Figure 3 simulation precision/recall": FIGURE_DIR / "figure_3_simulation_precision_recall.pdf",
        "Figure 4A INKA KSEA ranking": FIGURE_DIR / "figure_4a_inka_ksea_pvalue_ranking.pdf",
        "Figure 4B INKA LIKA ranking": FIGURE_DIR / "figure_4b_inka_lika_ranking.pdf",
        "Figure 5 ABL1/EGFR subnetworks": FIGURE_DIR / "figure_5_abl1_egfr_subnetworks.pdf",
        "Figure 6A SCZ KSEA ranking": FIGURE_DIR / "figure_6a_scz_ksea_pvalue_ranking.pdf",
        "Figure 6B SCZ LIKA ranking": FIGURE_DIR / "figure_6b_scz_lika_ranking.pdf",
        "FDR sensitivity figure": SUPPLEMENTARY_FIGURE_DIR / "supplementary_figure_fdr_sensitivity_top10_overlap.pdf",
        "FDR sensitivity source data": TABLE_DIR / "supplementary_figure_fdr_sensitivity_top10_overlap_source_data.csv",
    }
    for label, path in checks.items():
        status = "present" if path.exists() else "missing"
        print(f"{status:8} {label}: {path.relative_to(ROOT)}")
    print(
        "\nFocused reproducibility commands:\n"
        "- SCZ/control: `python scripts/generate_scz_assets.py`\n"
        "- INKA: `python scripts/generate_inka_assets.py`\n"
        "- Simulation: `python scripts/generate_simulation_assets.py --runs 100 --max-k 10`\n"
        "- FDR sensitivity heatmap from source data: "
        "`python scripts/run_fdr_sensitivity_analysis.py --use-existing-source-data`, "
        "or fully recompute with `python scripts/run_fdr_sensitivity_analysis.py`."
    )
