from method import *
import pandas as pd
import numpy as np
from utils import *
import json
import networkx as nx

def new_pipeline(intensity_df, log_transform=True, network_df=None):
    # Backward-compatible wrapper using unified pipeline ranked by p-values
    network, rejection_set, results_df, top_kinases, _ = pipeline(
        intensity_df, log_transform=log_transform, network_df=network_df, CI=None
    )
    new_p_values = dict(zip(results_df['Name'], results_df['p_value']))
    test_statistics = dict(zip(results_df['Name'], results_df.get('test_statistics', pd.Series(index=results_df['Name']).fillna(np.nan))))
    return new_p_values, test_statistics

def pipeline(intensity_df, log_transform=True, network_df=None, CI=None):
    """
    Unified LIKA pipeline.
    - Always computes kinase-level p-values via likelihood profiling.
    - If CI is provided (e.g., 0.2 for 80% CI), also computes lower/upper bounds and ranks by lower bound.
    - If CI is None, ranks kinases by their p-values.

    Returns: (network, rejection_set, results_df, top_kinases, substrate_p_values)
    """
    # process the column names(all lowercase)
    intensity_df.columns = intensity_df.columns.str.lower()
    # 1. Get p-value through empirical Bayes (or you can plain one)
    p_values, logFC = get_pvalue_through_empirical_bayes(intensity_df, log_transform)

    # 2. Get rejection set
    alpha = 0.05
    rejection_set = BH_(p_values, alpha) # should return a list
    # with open('results/rejection_set_INKA.txt', 'w') as f:
    #     f.write(','.join(rejection_set))

    # 3. Get network data # TODO: allow the user to include more netwokrs than kinase-phosphosite network
    if network_df is not None: # self-defined network
        network_df.columns = network_df.columns.str.lower()
        network = nx.from_pandas_edgelist(network_df, source="from", target="to", create_using=my_network)
        for index, row in network_df.iterrows():
            network.nodes[row['to']]['Description'] = 'Phosphosite'
            network.nodes[row['from']]['Description'] = 'Kinase'
            network.nodes[row['to']]['p_value'] = p_values[row['to']]
            network.nodes[row['to']]['logFC'] = logFC[row['to']]
            network.nodes[row['to']]['significant'] = row['to'] in rejection_set
        network.set_unmissing_neighbors_and_children()
        network.process_indisctinguishable_kinases()
        network.set_kinase_index()
    else: # use default network
        df = pd.read_csv('data/KSEA_dataset_processed.csv')
        df = df[df['to'].isin(p_values.keys())].reset_index(drop=True)
        print('number of substrates:', len(set(df['to']) & set(p_values.keys())))
        network = nx.from_pandas_edgelist(df, source="from", target="to", create_using=my_network)
        for index, row in df.iterrows():
            network.nodes[row['to']]['Description'] = 'Phosphosite'
            network.nodes[row['to']]['p_value'] = p_values[row['to']]
            network.nodes[row['from']]['Description'] = 'Kinase'
            network.nodes[row['to']]['logFC'] = logFC[row['to']]
            network.nodes[row['to']]['significant'] = row['to'] in rejection_set
        network.set_unmissing_neighbors_and_children()
        network.process_indisctinguishable_kinases()
        network.set_kinase_index()
    print('number of total substrates:', len(p_values.keys()))
    print('number of nodes in network:', len(network.nodes()))
    # 4. Always compute kinase p-values (profile test at null)
    kinase_p_values, test_statistics = get_kinase_ranking_new(network, rejection_set)

    if CI is not None:
        # Compute intervals and rank by lower bound
        top_kinases, results_df = get_kinase_ranking(network, rejection_set, CI)
        # Attach p-values for convenience
        results_df['p_value'] = results_df['Name'].map(kinase_p_values)
        # Keep ranking by lower bound as before
    else:
        # Rank solely by p-values
        results_df = pd.DataFrame(kinase_p_values.items(), columns=['Name', 'p_value'])
        results_df['test_statistics'] = results_df['Name'].map(test_statistics)
        results_df = results_df.sort_values(by='p_value', ascending=True)
        top_kinases = results_df.head(15)['Name'].tolist()

    return network, rejection_set, results_df, top_kinases, p_values

def pipeline_INKA():
    intensity_df = pd.read_csv('data/intensity_data_INKA.csv')
    network, rejection_set, results_df, top_kinases, p_values = pipeline(intensity_df, CI=0.2)
    with open('results/rejection_set_INKA.txt', 'a') as f:
        f.write(','.join(rejection_set))
    with open('results/INKA_p_values.json', 'w') as f:
        json.dump(p_values, f, indent=2)
    network.save_to_graphml('results/INKA_network.graphml')
    plot_top_kinases(results_df, top_kinases)
    results_df.to_csv('results/INKA_results.csv', index=False)

def pipeline_Schizo():
    intensity_df = pd.read_csv('data/residual_data_Schizo.csv')
    network, rejection_set, results_df, top_kinases, p_values = pipeline(intensity_df, log_transform=False, CI=0.2)
    with open('results/rejection_set_Schizo.txt', 'a') as f:
        f.write(','.join(rejection_set))
    with open('results/Schizo_p_values.json', 'w') as f:
        json.dump(p_values, f, indent=2)
    network.save_to_graphml('results/Schizo_network.graphml')
    plot_top_kinases(results_df, top_kinases)
    results_df.to_csv('results/Schizo_results.csv', index=False)

def stability_test_INKA():
    intensity_df = pd.read_csv('data/intensity_data_INKA.csv')
    top_kinase_all = []
    CI_list = [0.1, 0.2, 0.3, 0.4, 0.5]
    for CI in CI_list:
        network, rejection_set, results_df, top_kinases, p_values = pipeline(intensity_df, CI=CI)
        top_kinase_all.append(top_kinases)
    with open('results/top_kinase_INKA.txt', 'w') as f:
        for i in range(len(top_kinase_all)):
            f.write(str(CI_list[i]) + ':' + ','.join(top_kinase_all[i]) + '\n')

def stability_test_Schizo():
    intensity_df = pd.read_csv('data/residual_data_Schizo.csv')
    top_kinase_all = []
    CI_list = [0.1, 0.2, 0.3, 0.4, 0.5]
    for CI in CI_list:
        network, rejection_set, results_df, top_kinases, p_values = pipeline(intensity_df, log_transform=False, CI=CI)
        top_kinase_all.append(top_kinases)
    with open('results/top_kinase_Schizo.txt', 'w') as f:
        for i in range(len(top_kinase_all)):
            f.write(str(CI_list[i]) + ':' + ','.join(top_kinase_all[i]) + '\n')

if __name__ == "__main__":
    pipeline_INKA()
    pipeline_Schizo()
    # stability_test_INKA()
    # stability_test_Schizo()