"""
Shared simulation methods for LIKA manuscript reproducibility.

This module contains the simulation data generators, LIKA/KSEA ranking logic,
and precision/recall summaries. It does not write files directly; the
manuscript-facing command is `scripts/generate_simulation_assets.py`.
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

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
from scipy import stats

from method import get_intensity_columns
from pipeline import (
    DEFAULT_ALPHA,
    DEFAULT_P_VALUE_RANK_FLOOR,
    new_pipeline,
    resolve_p_value_rank_floor,
)
from utils import my_network


DEFAULT_RUNS = 100
DEFAULT_BASE_SEED = 20260603
DEFAULT_MAX_K = 10


def KSEA(intensity_df, log_transform=False, network_df=None):
    # Get p-values
    logFC = {}
    intensity_df = intensity_df.copy()
    intensity_df.columns = intensity_df.columns.str.lower()
    intensity_cols = get_intensity_columns(intensity_df)

    if log_transform:
        intensity_df['mean'] = intensity_df[intensity_cols].apply(lambda x: x[x != 0].mean(), axis=1)
        intensity_df['mean'] = np.log2(intensity_df['mean'] + 1e-6)
    else:
        intensity_df['mean'] = intensity_df[intensity_cols].mean(axis=1)

    for _, row in intensity_df.iterrows():
        if row['name'] in logFC:
            logFC[row['name']] += row['mean'] * (2 * row['group'] - 1)
        else:
            logFC[row['name']] = row['mean'] * (2 * row['group'] - 1)
    logFC = {key: value for key, value in logFC.items() if not np.isnan(value)}

    if network_df is not None:
        network_df = network_df.copy()
        network_df.columns = network_df.columns.str.lower()
        network_df = network_df[network_df['to'].isin(logFC.keys())].reset_index(drop=True)
        network = nx.from_pandas_edgelist(network_df, source="from", target="to", create_using=my_network)
        edge_df = network_df
    else:
        edge_df = pd.read_csv(ROOT / "data" / "KSEA_dataset_processed.csv")
        edge_df.columns = edge_df.columns.str.lower()
        edge_df = edge_df[edge_df['to'].isin(logFC.keys())].reset_index(drop=True)
        network = nx.from_pandas_edgelist(edge_df, source="from", target="to", create_using=my_network)

    for _, row in edge_df.iterrows():
        network.nodes[row['to']]['Description'] = 'Phosphosite'
        network.nodes[row['from']]['Description'] = 'Kinase'
        network.nodes[row['to']]['p_value'] = logFC[row['to']]

    network.set_kinase_index()
    network.set_unmissing_neighbors_and_children()

    background_values = list(logFC.values())
    background_mean = np.mean(background_values)
    background_std = np.std(background_values)

    kinase_scores = {}
    p_values = {}
    for kinase in network.get_kinase():
        children = network.get_unmissing_children(kinase)
        if not children:
            continue
        kinase_scores[kinase] = sum(network.nodes[node]['p_value'] for node in children) / len(children)
        kinase_scores[kinase] = (kinase_scores[kinase] - background_mean) / (background_std / np.sqrt(len(children)))
        p_values[kinase] = 2 * (1 - stats.norm.cdf(abs(kinase_scores[kinase])))

    results_df = pd.DataFrame({
        'Name': list(p_values.keys()),
        'p_value': list(p_values.values()),
        'KSEA_score': [kinase_scores[name] for name in p_values],
    })
    top_kinases = (
        results_df.sort_values(['p_value', 'Name'], ascending=[True, True])
        .head(10)['Name']
        .tolist()
    )
    return results_df, top_kinases


def experiment_1_inputs(seed):
    rng = np.random.RandomState(seed)
    network_df = pd.DataFrame({
        "from": [f"K{x // 10}" for x in range(100)],
        "to": [str(x) for x in range(100)],
    })

    rows = []
    for substrate in range(100):
        row_group0 = {"name": str(substrate), "group": 0}
        for i, value in enumerate(rng.normal(0, 1, 5)):
            row_group0[f"intensity_{i}"] = value
        rows.append(row_group0)

        row_group1 = {"name": str(substrate), "group": 1}
        mean = 2 if substrate < 20 else 0
        for i, value in enumerate(rng.normal(mean, 1, 5)):
            row_group1[f"intensity_{i}"] = value
        rows.append(row_group1)

    return network_df, pd.DataFrame(rows), ["K0", "K1"]


def experiment_2_inputs(seed):
    rng = np.random.RandomState(seed)
    kinase_substrate_counts = [20, 20, 15, 15, 12, 11, 10, 8, 6, 4, 3, 2, 1]
    total_substrates = sum(kinase_substrate_counts)
    abnormal_kinases = ["K0", "K2", "K5", "K8"]

    from_ = []
    to_ = []
    substrate_idx = 0
    for kinase_idx, substrate_count in enumerate(kinase_substrate_counts):
        for _ in range(substrate_count):
            from_.append(f"K{kinase_idx}")
            to_.append(str(substrate_idx))
            substrate_idx += 1

    network_df = pd.DataFrame({"from": from_, "to": to_})

    abnormal_substrates = set()
    for substrate in range(total_substrates):
        parent_kinases = set(network_df[network_df["to"] == str(substrate)]["from"])
        if parent_kinases:
            probability = len(parent_kinases & set(abnormal_kinases)) / len(parent_kinases)
            if rng.uniform(0, 1) < probability:
                abnormal_substrates.add(str(substrate))

    rows = []
    for substrate in range(total_substrates):
        substrate_name = str(substrate)
        row_group0 = {"name": substrate_name, "group": 0}
        for i, value in enumerate(rng.normal(0, 1, 5)):
            row_group0[f"intensity_{i}"] = value
        rows.append(row_group0)

        row_group1 = {"name": substrate_name, "group": 1}
        mean = 2 if substrate_name in abnormal_substrates else 0
        for i, value in enumerate(rng.normal(mean, 1, 5)):
            row_group1[f"intensity_{i}"] = value
        rows.append(row_group1)

    return network_df, pd.DataFrame(rows), abnormal_kinases


def experiment_3_inputs(seed):
    rng = np.random.RandomState(seed)
    kinase_substrate_counts = [20, 20, 15, 15, 12, 11, 10, 8, 6, 4, 3, 2, 1]
    total_substrates = 40
    abnormal_kinases = ["K0", "K2", "K5", "K8"]

    from_ = []
    to_ = []
    substrate_pool = list(range(total_substrates))
    for kinase_idx, substrate_count in enumerate(kinase_substrate_counts):
        selected_substrates = rng.choice(substrate_pool, size=substrate_count, replace=False)
        for substrate in selected_substrates:
            from_.append(f"K{kinase_idx}")
            to_.append(str(substrate))

    network_df = pd.DataFrame({"from": from_, "to": to_})

    abnormal_substrates = set()
    for substrate in substrate_pool:
        parent_kinases = set(network_df[network_df["to"] == str(substrate)]["from"])
        if parent_kinases:
            probability = len(parent_kinases & set(abnormal_kinases)) / len(parent_kinases)
            if rng.uniform(0, 1) < probability:
                abnormal_substrates.add(str(substrate))

    rows = []
    for substrate in range(total_substrates):
        substrate_name = str(substrate)
        row_group0 = {"name": substrate_name, "group": 0}
        for i, value in enumerate(rng.normal(0, 1, 5)):
            row_group0[f"intensity_{i}"] = value
        rows.append(row_group0)

        row_group1 = {"name": substrate_name, "group": 1}
        mean = 2 if substrate_name in abnormal_substrates else 0
        for i, value in enumerate(rng.normal(mean, 1, 5)):
            row_group1[f"intensity_{i}"] = value
        rows.append(row_group1)

    return network_df, pd.DataFrame(rows), abnormal_kinases


def substrate_metrics(network_df):
    graph = nx.from_pandas_edgelist(
        network_df,
        source="from",
        target="to",
        create_using=my_network,
    )

    rows = []
    for kinase in sorted(network_df["from"].unique()):
        children = list(graph.successors(kinase))
        influence_score = sum(1.0 / graph.in_degree(child) for child in children)
        rows.append({
            "Name": kinase,
            "number_of_substrates": len(children),
            "number_of_efficient_substrates": influence_score,
            "LIKA_influence_score": influence_score,
        })
    return pd.DataFrame(rows)


def add_rankings(df, alpha=DEFAULT_ALPHA, p_value_rank_floor=DEFAULT_P_VALUE_RANK_FLOOR):
    df = df.copy()
    rank_floor = resolve_p_value_rank_floor(p_value_rank_floor, alpha, df["LIKA_p_value"].notna().sum())
    df["LIKA_ranking_p_value"] = df["LIKA_p_value"].clip(lower=rank_floor)

    lika_sorted = df.sort_values(
        ["LIKA_ranking_p_value", "LIKA_influence_score", "Name"],
        ascending=[True, False, True],
        na_position="last",
    )
    df["LIKA_ranking"] = pd.NA
    df.loc[lika_sorted.index, "LIKA_ranking"] = range(1, len(lika_sorted) + 1)

    ksea_sorted = df.sort_values(["KSEA_p_value", "Name"], ascending=[True, True], na_position="last")
    df["KSEA_ranking"] = pd.NA
    df.loc[ksea_sorted.index, "KSEA_ranking"] = range(1, len(ksea_sorted) + 1)

    df["LIKA_ranking"] = df["LIKA_ranking"].astype("Int64")
    df["KSEA_ranking"] = df["KSEA_ranking"].astype("Int64")
    return df


def run_one_experiment(experiment_number, input_fn, runs, base_seed):
    all_rows = []

    for run_index in range(runs):
        seed = base_seed + experiment_number * 100_000 + run_index
        network_df, intensity_df, ground_truth = input_fn(seed)

        ksea_df, _ = KSEA(intensity_df.copy(), log_transform=False, network_df=network_df.copy())
        lika_p_values, lika_test_statistics = new_pipeline(
            intensity_df.copy(),
            log_transform=False,
            network_df=network_df.copy(),
        )

        lika_df = pd.DataFrame({
            "Name": list(lika_p_values.keys()),
            "LIKA_p_value": list(lika_p_values.values()),
            "LIKA_test_statistics": [lika_test_statistics[name] for name in lika_p_values],
        })

        combined = (
            substrate_metrics(network_df)
            .merge(ksea_df[["Name", "p_value"]].rename(columns={"p_value": "KSEA_p_value"}), on="Name", how="left")
            .merge(lika_df, on="Name", how="left")
        )
        combined = add_rankings(combined)
        combined["experiment"] = experiment_number
        combined["run_index"] = run_index + 1
        combined["seed"] = seed
        combined["is_ground_truth"] = combined["Name"].isin(ground_truth)

        output_columns = [
            "experiment",
            "run_index",
            "seed",
            "Name",
            "is_ground_truth",
            "KSEA_p_value",
            "KSEA_ranking",
            "LIKA_p_value",
            "LIKA_ranking_p_value",
            "LIKA_test_statistics",
            "LIKA_ranking",
            "LIKA_influence_score",
            "number_of_substrates",
            "number_of_efficient_substrates",
        ]
        all_rows.append(combined[output_columns])
        print(f"Experiment {experiment_number}, run {run_index + 1}/{runs}, seed {seed}")

    return pd.concat(all_rows, ignore_index=True)


def precision_recall_at_k(ranking_df, max_k=DEFAULT_MAX_K):
    rows = []
    for (experiment, run_index), run_df in ranking_df.groupby(["experiment", "run_index"]):
        ground_truth_count = int(run_df["is_ground_truth"].sum())
        for method in ("LIKA", "KSEA"):
            rank_col = f"{method}_ranking"
            method_df = run_df.dropna(subset=[rank_col]).sort_values(rank_col)
            for k in range(1, min(max_k, len(method_df)) + 1):
                selected = method_df.head(k)
                hits = int(selected["is_ground_truth"].sum())
                rows.append({
                    "experiment": experiment,
                    "run_index": run_index,
                    "method": method,
                    "k": k,
                    "ground_truth_count": ground_truth_count,
                    "precision_at_k": hits / k,
                    "recall_at_k": hits / ground_truth_count if ground_truth_count else np.nan,
                })
    return pd.DataFrame(rows)


def summarize_precision_recall(metric_df):
    return (
        metric_df
        .groupby(["experiment", "method", "k", "ground_truth_count"], as_index=False)
        .agg(
            precision_at_k_mean=("precision_at_k", "mean"),
            precision_at_k_sd=("precision_at_k", "std"),
            recall_at_k_mean=("recall_at_k", "mean"),
            recall_at_k_sd=("recall_at_k", "std"),
        )
    )


def plot_precision_recall_summary(summary_df, output_path):
    experiments = sorted(summary_df["experiment"].unique())
    fig, axes = plt.subplots(len(experiments), 2, figsize=(10, 3.3 * len(experiments)), sharex=True, sharey=True)
    if len(experiments) == 1:
        axes = np.array([axes])

    colors = {"LIKA": "#1b9e77", "KSEA": "#d95f02"}
    for row_idx, experiment in enumerate(experiments):
        experiment_df = summary_df[summary_df["experiment"] == experiment]
        ground_truth_count = int(experiment_df["ground_truth_count"].iloc[0])

        for col_idx, metric in enumerate(("precision_at_k_mean", "recall_at_k_mean")):
            ax = axes[row_idx, col_idx]
            for method in ("LIKA", "KSEA"):
                method_df = experiment_df[experiment_df["method"] == method].sort_values("k")
                ax.plot(method_df["k"], method_df[metric], marker="o", linewidth=2, label=method, color=colors[method])
            ax.axvline(ground_truth_count, color="#555555", linestyle="--", linewidth=1.4, alpha=0.8)
            ax.set_ylim(-0.03, 1.03)
            ax.set_title(f"Simulation Experiment {experiment}")
            ax.set_xlabel("k")
            ax.set_ylabel("Precision@k" if metric.startswith("precision") else "Recall@k")
            ax.grid(True, axis="y", alpha=0.25)
            ax.spines[["top", "right"]].set_visible(False)
            if row_idx == 0 and col_idx == 0:
                ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
