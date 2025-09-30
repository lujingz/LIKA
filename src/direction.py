import pandas as pd
import networkx as nx
from vash import *
from method import *
from utils import *

'''
Identify the direction of the kinase dysregulation in the Schizo setting
'''
log_transform = False
intensity_df = pd.read_csv('data/residual_data_Schizo.csv')
intensity_df.columns = intensity_df.columns.str.lower()
dataset1 = intensity_df[intensity_df['group'] == 0]
dataset2 = intensity_df[intensity_df['group'] == 1]
merged_data = process_gene_pair_analysis(dataset1, dataset2, log_transform)
vash_results = vash(sehat=merged_data['sehat'], betahat=merged_data['betahat'], \
                                    df=merged_data['degrees_of_freedom'])
p_values = {}
for i, molecule in enumerate(merged_data['molecules']):
    p_values[molecule] = vash_results['pvalue'][i]
# 2. Get rejection set
alpha = 0.05
rejection_set = BH_(p_values, alpha) # should return a list
df = pd.read_csv('data/KSEA_dataset_processed.csv')
df = df[df['to'].isin(p_values.keys())].reset_index(drop=True)
print('number of substrates:', len(set(df['to']) & set(p_values.keys())))
network = nx.from_pandas_edgelist(df, source="from", target="to", create_using=my_network)
for index, row in df.iterrows():
    network.nodes[row['to']]['Description'] = 'Phosphosite'
    network.nodes[row['to']]['p_value'] = p_values[row['to']]
    network.nodes[row['from']]['Description'] = 'Kinase'
network.set_unmissing_neighbors_and_children()
network.process_indisctinguishable_kinases()
network.set_kinase_index()
network.get_depths_for_all_nodes()

kinase_interested = ['CSNK2A2', 'NLK', 'PLK1', 'PRKCZ', 'TTBK2', 'PRKAA1', 'SGK1', 'PRKCD', 'CAMK4', 'MAPK13']
for kinase in kinase_interested:
    # Get unmissing children in rejection set
    unmissing_children = network.get_unmissing_children(kinase)
    rejected_children = [child for child in unmissing_children if child in rejection_set]
    
    if not rejected_children:
        print(f"{kinase}: No rejected children found")
        continue
        
    # Count up/down regulation
    up_count = sum(1 for child in rejected_children if vash_results['pvalue_oneside'][list(merged_data['molecules']).index(child)] > 0.5)
    total = len(rejected_children)
    up_ratio = up_count / total
    
    # Print results
    print(f"{kinase}: Up ratio = {up_ratio:.2f} ({up_count}/{total})")
    print(f"Direction: {'up' if up_ratio > 0.5 else 'down'}")
    print("---")
