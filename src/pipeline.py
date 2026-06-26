from method import *
import pandas as pd
import numpy as np
from utils import *
import networkx as nx

DEFAULT_ALPHA = 0.05
DEFAULT_TOP_N = 15
DEFAULT_P_VALUE_RANK_FLOOR = "bonferroni"


def get_rejection_pvalue_cutoff(p_values, rejection_set):
    """
    Return the raw substrate p-value cutoff induced by the rejection set.

    This is the realized cutoff from the first-stage BH procedure, not a q-value.
    It is used as the null dysregulation probability in the kinase-level test.
    """
    rejected_p_values = [p_values[node] for node in rejection_set if node in p_values]
    if not rejected_p_values:
        return 0.0
    return max(rejected_p_values)


def build_lika_network(p_values, logFC, rejection_set, network_df=None):
    if network_df is not None:
        df = network_df.copy()
        df.columns = df.columns.str.lower()
    else:
        df = pd.read_csv('data/KSEA_dataset_processed.csv')
        df.columns = df.columns.str.lower()

    df = df[df['to'].isin(p_values.keys())].reset_index(drop=True)
    print('number of substrates:', len(set(df['to']) & set(p_values.keys())))

    network = nx.from_pandas_edgelist(df, source="from", target="to", create_using=my_network)
    for _, row in df.iterrows():
        network.nodes[row['to']]['Description'] = 'Phosphosite'
        network.nodes[row['from']]['Description'] = 'Kinase'
        network.nodes[row['to']]['p_value'] = p_values[row['to']]
        network.nodes[row['to']]['logFC'] = logFC[row['to']]
        network.nodes[row['to']]['significant'] = row['to'] in rejection_set

    network.set_unmissing_neighbors_and_children()
    network.process_indisctinguishable_kinases()
    network.set_unmissing_neighbors_and_children()
    network.set_kinase_index()
    return network


def calculate_lika_influence_scores(network):
    """
    Compute the LIKA influence score for each kinase.

    The score is the effective number of downstream substrates:
    sum_s 1 / parent_degree(s), where s ranges over a kinase's observed substrates.
    """
    rows = []
    for kinase in network.get_kinase():
        children = list(dict.fromkeys(network.get_unmissing_children(kinase)))
        influence_score = 0.0
        for child in children:
            parent_count = len(network.get_kinase_parents(child))
            if parent_count > 0:
                influence_score += 1.0 / parent_count
        rows.append({
            'Name': kinase,
            'number_of_substrates': len(children),
            'influence_score': influence_score,
            'number_of_efficient_substrates': influence_score,
        })
    return pd.DataFrame(rows)


def resolve_p_value_rank_floor(p_value_rank_floor, alpha, number_of_tests):
    if p_value_rank_floor in (None, False):
        return 0.0
    if p_value_rank_floor == "bonferroni":
        return alpha / max(number_of_tests, 1)
    return float(p_value_rank_floor)


def rank_kinase_results(results_df, alpha=DEFAULT_ALPHA,
                        p_value_rank_floor=DEFAULT_P_VALUE_RANK_FLOOR):
    """
    Rank kinases by capped p-value, breaking ties by LIKA influence score.
    """
    results_df = results_df.copy()
    floor = resolve_p_value_rank_floor(p_value_rank_floor, alpha, len(results_df))
    results_df['ranking_p_value'] = results_df['p_value'].clip(lower=floor)
    results_df = results_df.sort_values(
        ['ranking_p_value', 'influence_score', 'Name'],
        ascending=[True, False, True],
        na_position='last'
    ).reset_index(drop=True)
    results_df['LIKA_rank'] = np.arange(1, len(results_df) + 1)
    return results_df


def new_pipeline(intensity_df, log_transform=True, network_df=None):
    # Backward-compatible wrapper around the p-value/influence ranked pipeline.
    _, _, results_df, _, _ = pipeline(
        intensity_df,
        log_transform=log_transform,
        network_df=network_df,
        CI=None,
    )
    new_p_values = dict(zip(results_df['Name'], results_df['p_value']))
    test_statistics = dict(zip(results_df['Name'], results_df['test_statistics']))
    return new_p_values, test_statistics

def pipeline(intensity_df, log_transform=True, network_df=None, CI=None,
             alpha=DEFAULT_ALPHA, top_n=DEFAULT_TOP_N,
             p_value_rank_floor=DEFAULT_P_VALUE_RANK_FLOOR):
    """
    Unified LIKA pipeline.
    - Always computes kinase-level p-values via likelihood profiling.
    - Ranks kinases by p-value and breaks ties with LIKA influence score.
    - If CI is provided (e.g., 0.2 for 80% CI), also attaches lower/upper bounds.

    Returns: (network, rejection_set, results_df, top_kinases, substrate_p_values)
    """
    # process the column names(all lowercase)
    intensity_df = intensity_df.copy()
    intensity_df.columns = intensity_df.columns.str.lower()
    # 1. Get p-value through empirical Bayes (or you can plain one)
    p_values, logFC = get_pvalue_through_empirical_bayes(intensity_df, log_transform)

    # 2. Get rejection set
    rejection_set = BH_(p_values, alpha) # should return a list
    p_fixed = get_rejection_pvalue_cutoff(p_values, rejection_set)

    # 3. Get network data # TODO: allow the user to include more networks than kinase-phosphosite network
    network = build_lika_network(p_values, logFC, rejection_set, network_df=network_df)
    print('number of total substrates:', len(p_values.keys()))
    print('number of nodes in network:', len(network.nodes()))

    # 4. Compute kinase p-values and rank by p-value/influence.
    kinase_p_values, test_statistics = get_kinase_ranking_new(
        network,
        rejection_set,
        p_fixed=p_fixed,
    )
    results_df = pd.DataFrame(kinase_p_values.items(), columns=['Name', 'p_value'])
    results_df['test_statistics'] = results_df['Name'].map(test_statistics)
    results_df['substrate_p_value_cutoff'] = p_fixed
    results_df = results_df.merge(calculate_lika_influence_scores(network), on='Name', how='left')

    if CI is not None:
        _, interval_df = get_kinase_ranking(network, rejection_set, CI)
        results_df = results_df.merge(interval_df, on='Name', how='left')

    results_df = rank_kinase_results(
        results_df,
        alpha=alpha,
        p_value_rank_floor=p_value_rank_floor,
    )
    top_kinases = results_df.head(top_n)['Name'].tolist()

    return network, rejection_set, results_df, top_kinases, p_values
