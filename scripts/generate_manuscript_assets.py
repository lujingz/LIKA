#!/usr/bin/env python
"""
Generate manuscript-facing tables and figures without changing LIKA core code.

Default behavior uses included supplementary ranking tables where possible and
writes figures/tables into the `manuscript/` folder. Recomputing kinase ranking
tables from raw processed data can be slow; use `--recompute-rankings` only when
you intentionally want fresh LIKA/KSEA results.
"""

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/lika-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/lika-cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from method import BH_, get_pvalue_through_empirical_bayes
from pipeline import pipeline
from simulation_experiment import KSEA, plot_precision_recall_summary

MANUSCRIPT_DIR = ROOT / "manuscript"
FIGURE_DIR = MANUSCRIPT_DIR / "figures" / "main"
SUPPLEMENTARY_FIGURE_DIR = MANUSCRIPT_DIR / "figures" / "supplementary"
TABLE_DIR = MANUSCRIPT_DIR / "supplementary_tables"
RESULTS_DIR = ROOT / "results"

SIMULATION_TABLE = TABLE_DIR / "supplementary_table_s1_simulation_network_structures.csv"
INKA_RANKING_TABLE = TABLE_DIR / "supplementary_table_s2_inka_kinase_rankings.csv"
SCZ_SUBSTRATE_TABLE = TABLE_DIR / "supplementary_table_s3_scz_substrate_statistics.csv"
INKA_SUBSTRATE_TABLE = TABLE_DIR / "supplementary_table_s4_inka_substrate_statistics.csv"
SCZ_RANKING_TABLE = TABLE_DIR / "supplementary_table_s5_scz_kinase_rankings.csv"


def ensure_dirs():
    for directory in [FIGURE_DIR, SUPPLEMENTARY_FIGURE_DIR, TABLE_DIR, RESULTS_DIR]:
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


def write_kinase_ranking_table(intensity_path, output_path, log_transform):
    intensity_df = pd.read_csv(intensity_path)
    network, _, lika_df, _, _ = pipeline(intensity_df, log_transform=log_transform, CI=None)
    ksea_df, _ = KSEA(intensity_df, log_transform=log_transform)

    combined = (
        lika_df.rename(
            columns={
                "p_value": "LIKA_p_value",
                "test_statistics": "LIKA_test_statistics_fixed",
                "ranking_p_value": "LIKA_p_value_capped",
                "LIKA_rank": "LIKA_ranking",
                "influence_score": "number_of_efficient_substrates",
            }
        )
        .merge(
            ksea_df[["Name", "p_value"]].rename(columns={"p_value": "KSEA_p_value"}),
            on="Name",
            how="left",
        )
        .merge(downstream_substrates(network), on="Name", how="left")
    )
    combined = combined.sort_values(["KSEA_p_value", "Name"], na_position="last").reset_index(drop=True)
    combined["KSEA_ranking"] = np.arange(1, len(combined) + 1)
    combined = combined.sort_values("LIKA_ranking").reset_index(drop=True)

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


def finite_log_p(values):
    values = pd.Series(values).astype(float)
    positive = values[(values > 0) & np.isfinite(values)]
    floor = positive.min() if not positive.empty else np.nextafter(0, 1)
    return -np.log10(values.replace(0, floor).fillna(1.0))


def ranking_plot(
    ranking_table,
    output_path,
    title,
    rank_column,
    pvalue_column,
    top_n=10,
):
    df = pd.read_csv(ranking_table)
    df = df.sort_values(rank_column).head(top_n).copy()
    score_col = "number_of_efficient_substrates"
    color_values = finite_log_p(df[pvalue_column])
    color_max = max(float(color_values.max()), 1.0)

    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    ax.barh(
        df["Name"],
        df[score_col],
        color=plt.cm.viridis(color_values / color_max),
    )
    ax.invert_yaxis()
    ax.set_title(title)
    ax.set_xlabel("LIKA Influence Score")
    ax.spines[["top", "right"]].set_visible(False)
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(vmin=0, vmax=color_max))
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label=f"-log10({pvalue_column.replace('_', ' ')})")
    fig.tight_layout()
    fig.savefig(output_path)
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
    summary_path=RESULTS_DIR / "simulation_precision_recall_summary_100runs.csv",
    output_path=FIGURE_DIR / "figure_3_simulation_precision_recall.pdf",
):
    if not summary_path.exists():
        raise FileNotFoundError(
            f"{summary_path} does not exist. Run `python src/simulation_experiment.py --runs 100 --output-dir results --max-k 10` first."
        )
    summary_df = pd.read_csv(summary_path)
    plot_precision_recall_summary(summary_df, output_path)
    return output_path


def generate_figures(include_figure3=False):
    generated = []
    generated.append(
        ranking_plot(
            INKA_RANKING_TABLE,
            FIGURE_DIR / "figure_4a_inka_ksea_pvalue_ranking.pdf",
            "INKA Dataset KSEA p-value ranking",
            "KSEA_ranking",
            "KSEA_p_value",
        )
    )
    generated.append(
        ranking_plot(
            INKA_RANKING_TABLE,
            FIGURE_DIR / "figure_4b_inka_lika_ranking.pdf",
            "INKA Dataset LIKA ranking",
            "LIKA_ranking",
            "LIKA_p_value_capped" if "LIKA_p_value_capped" in pd.read_csv(INKA_RANKING_TABLE, nrows=1).columns else "LIKA_p_value",
        )
    )
    generated.append(figure_5_subnetworks())
    generated.append(
        ranking_plot(
            SCZ_RANKING_TABLE,
            FIGURE_DIR / "figure_6a_scz_ksea_pvalue_ranking.pdf",
            "SCZ Dataset KSEA p-value ranking",
            "KSEA_ranking",
            "KSEA_p_value",
        )
    )
    generated.append(
        ranking_plot(
            SCZ_RANKING_TABLE,
            FIGURE_DIR / "figure_6b_scz_lika_ranking.pdf",
            "SCZ Dataset LIKA ranking",
            "LIKA_ranking",
            "LIKA_p_value_capped" if "LIKA_p_value_capped" in pd.read_csv(SCZ_RANKING_TABLE, nrows=1).columns else "LIKA_p_value",
        )
    )
    if include_figure3:
        generated.append(figure_3_from_summary())
    return generated


def audit_outputs():
    checks = {
        "S1 simulation table": SIMULATION_TABLE,
        "S2 INKA kinase rankings": INKA_RANKING_TABLE,
        "S3 SCZ substrate statistics": SCZ_SUBSTRATE_TABLE,
        "S4 INKA substrate statistics": INKA_SUBSTRATE_TABLE,
        "S5 SCZ kinase rankings": SCZ_RANKING_TABLE,
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
    print("\nFDR sensitivity can be regenerated with `python scripts/run_fdr_sensitivity_analysis.py`.")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", action="store_true", help="Report which manuscript outputs are present.")
    parser.add_argument("--tables", action="store_true", help="Generate supplementary tables.")
    parser.add_argument("--figures", action="store_true", help="Generate manuscript figures from available tables/results.")
    parser.add_argument("--include-figure3", action="store_true", help="Regenerate Figure 3 from results/simulation_precision_recall_summary_100runs.csv.")
    parser.add_argument("--recompute-rankings", action="store_true", help="Recompute S2 and S5 kinase ranking tables from data. This can be slow.")
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_dirs()
    if args.audit:
        audit_outputs()
    if args.tables:
        write_simulation_structure_table()
        write_substrate_statistics(ROOT / "data" / "residual_data_SCZ.csv", SCZ_SUBSTRATE_TABLE, log_transform=False, include_all_substrates=False)
        write_substrate_statistics(ROOT / "data" / "intensity_data_INKA.csv", INKA_SUBSTRATE_TABLE, log_transform=True, include_all_substrates=True)
        if args.recompute_rankings:
            write_kinase_ranking_table(ROOT / "data" / "intensity_data_INKA.csv", INKA_RANKING_TABLE, log_transform=True)
            write_kinase_ranking_table(ROOT / "data" / "residual_data_SCZ.csv", SCZ_RANKING_TABLE, log_transform=False)
    if args.figures:
        generate_figures(include_figure3=args.include_figure3)
    if not any([args.audit, args.tables, args.figures]):
        audit_outputs()


if __name__ == "__main__":
    main()
