from method import *
import pandas as pd
import numpy as np
from utils import *
import json
import networkx as nx

def new_pipeline(intensity_df, log_transform=True, network_df=None):
        # process the column names(all lowercase)
    intensity_df.columns = intensity_df.columns.str.lower()
    # 1. Get p-value through empirical Bayes (or you can plain one)
    p_values, logFC = get_pvalue_through_empirical_bayes(intensity_df, log_transform)
    # truncate_p_values = {}
    # i = 0
    # for key, value in p_values.items():
    #     if i < 20:
    #         truncate_p_values[key] = value
    #         i += 1
    #     else:
    #         break
    # p_values = truncate_p_values

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
        network.get_depths_for_all_nodes()
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
        network.get_depths_for_all_nodes()
    print('number of total substrates:', len(p_values.keys()))
    print('number of nodes in network:', len(network.nodes()))
    # 4. Get kinase ranking through max likelihood method
    new_p_values, test_statistics = get_kinase_ranking_new(network, rejection_set) # can add alpha if you want
    return new_p_values, test_statistics

def pipeline(intensity_df, log_transform=True, network_df=None, CI=0.8):
    """
    Main pipeline function that processes intensity data and optional network data.
    
    Args:
        intensity_df (pd.DataFrame): Intensity dataset (required)
        network_df (pd.DataFrame, optional): Network dataset (can be None)
    
    Returns:
        tuple: (result_dataframe, result_list)
            - result_dataframe (pd.DataFrame): Processed results
            - result_list (list): List of important findings/features
    """
    # process the column names(all lowercase)
    intensity_df.columns = intensity_df.columns.str.lower()
    # 1. Get p-value through empirical Bayes (or you can plain one)
    p_values, logFC = get_pvalue_through_empirical_bayes(intensity_df, log_transform)
    # truncate_p_values = {}
    # i = 0
    # for key, value in p_values.items():
    #     if i < 20:
    #         truncate_p_values[key] = value
    #         i += 1
    #     else:
    #         break
    # p_values = truncate_p_values

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
        network.get_depths_for_all_nodes()
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
        network.get_depths_for_all_nodes()
    print('number of total substrates:', len(p_values.keys()))
    print('number of nodes in network:', len(network.nodes()))
    # 4. Get kinase ranking through max likelihood method
    top_kinases, results_df = get_kinase_ranking(network, rejection_set, CI) # can add alpha if you want
    return network, rejection_set, results_df, top_kinases, p_values

def test_pipeline_1():
    intensity_df = pd.read_csv('data/intensity_data_INKA.csv')
    network, rejection_set, results_df, top_kinases, p_values = pipeline(intensity_df, CI=0.2)
    with open('results/rejection_set_INKA.txt', 'a') as f:
        f.write(','.join(rejection_set))
    with open('results/INKA_p_values.json', 'w') as f:
        json.dump(p_values, f, indent=2)
    network.save_to_graphml('results/INKA_network.graphml')
    visualize_rejection(network, rejection_set, 'INKA', None)
    plot_top_kinases(results_df, top_kinases)
    results_df.to_csv('results/INKA_results.csv', index=False)

def test_pipeline_1_new():
    intensity_df = pd.read_csv('data/intensity_data_INKA.csv')
    new_p_values, test_statistics = new_pipeline(intensity_df, log_transform=True)
    df = pd.DataFrame(new_p_values.items(), columns=['Name', 'p_value'])
    df['test_statistics'] = df['Name'].map(test_statistics)
    df = df.sort_values(by='p_value', ascending=True)
    df.to_csv('results/INKA_results_new.csv', index=False)
    ground_truth = ['EGFR', 'ABL1']
    colors_ksea = ["#E69F00" if name in ground_truth else "#0072B2" for name in df['Name']]
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(df['Name'], df['p_value'], color=colors_ksea)
    ax.set_xlabel("p-value by new pipeline", fontsize=12)
    ax.set_xscale('log')
    ax.invert_yaxis()  # highest score on top
    ax.spines[['top', 'right']].set_visible(False)
    # Add p-value labels on the bars
    for i, (name, p_val) in enumerate(zip(df['Name'], df['p_value'])):
        ax.text(p_val * 1.1, i, f'{p_val:.5f}', va='center', fontsize=10)
    plt.tight_layout()
    plt.savefig('results/INKA_new_pipeline.png')
    plt.show()


def test_pipeline_2():
    intensity_df = pd.read_csv('data/residual_data_Schizo.csv')
    network, rejection_set, results_df, top_kinases, p_values = pipeline(intensity_df, log_transform=False, CI=0.2)
    with open('results/rejection_set_Schizo.txt', 'a') as f:
        f.write(','.join(rejection_set))
    with open('results/Schizo_p_values.json', 'w') as f:
        json.dump(p_values, f, indent=2)
    network.save_to_graphml('results/Schizo_network.graphml')
    visualize_rejection(network, rejection_set, 'Schizo', None)
    plot_top_kinases(results_df, top_kinases)
    results_df.to_csv('results/Schizo_results.csv', index=False)

def test_pipeline_2_new():
    intensity_df = pd.read_csv('data/residual_data_Schizo.csv')
    new_p_values, test_statistics = new_pipeline(intensity_df, log_transform=False)
    df = pd.DataFrame(new_p_values.items(), columns=['Name', 'p_value'])
    df['test_statistics'] = df['Name'].map(test_statistics)
    df = df.sort_values(by='p_value', ascending=True)
    df.to_csv('results/Schizo_results_new.csv', index=False)
    df = df[:10]
    ground_truth = []
    colors_ksea = ["#E69F00" if name in ground_truth else "#0072B2" for name in df['Name']]
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(df['Name'], df['p_value'], color=colors_ksea)
    ax.set_xlabel("p-value by new pipeline", fontsize=12)
    ax.set_xscale('log')
    ax.invert_yaxis()  # highest score on top
    ax.spines[['top', 'right']].set_visible(False)
    # Add p-value labels on the bars
    for i, (name, p_val) in enumerate(zip(df['Name'], df['p_value'])):
        ax.text(p_val * 1.1, i, f'{p_val:.5f}', va='center', fontsize=10)
    plt.tight_layout()
    plt.savefig('results/Schizo_new_pipeline.png')
    plt.show()

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
    # test_pipeline_1_new()
    # stability_test_INKA()
    # stability_test_Schizo()
    test_pipeline_2_new()