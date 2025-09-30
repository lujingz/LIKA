"""
This file contains utility functions for the project.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats
from matplotlib_venn import venn2
from matplotlib_venn import venn3
import matplotlib.colors as mcolors
from matplotlib import cm
import networkx as nx
import sys
from typing import List, Dict, Tuple
from scipy.stats import norm
try:
    import pygraphviz as pgv
    PYGRAPHVIZ_AVAILABLE = True
except ImportError:
    PYGRAPHVIZ_AVAILABLE = False
    pgv = None
import random
import time
import igraph as ig
from networkx.drawing import nx_agraph
from community import community_louvain
import logging
import yaml
import os

class my_network(nx.DiGraph):
    '''node also has attribute like 'Description','p_value','unmissing_children' '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.missing_pvalues = []
        self.missing_pvalues_count = 0

    @classmethod
    def from_graph(cls, g):
        new_g = cls()
        new_g.add_nodes_from(g.nodes(data=True))
        new_g.add_edges_from(g.edges(data=True))
        return new_g

    def __repr__(self):
        name = self.graph.get("name", "")
        info = [
            f"<MyDirectedNetwork name='{name}'>",
            f"  Number of nodes: {self.number_of_nodes()}",
            f"  Number of edges: {self.number_of_edges()}",
            f"  Directed: {self.is_directed()}",
            f"  Graph attributes: {dict(self.graph)}"
        ]
        return "\n".join(info)

    def check_missing_pvalues(self, logger, output_dir=None):
        '''check if there are missing p-values and save them to a CSV file'''
        missing_pvalues = []
        for node in self.nodes():
            if not ('p_value' in self.nodes[node]):
                description = self.nodes[node].get('Description', 'No description')
                in_degree= self.in_degree(node)
                out_degree= self.out_degree(node)
                missing_pvalues.append((node, description, in_degree, out_degree))
        
        missing_pvalues = sorted(missing_pvalues, key=lambda x: x[2], reverse=True)
        if output_dir and missing_pvalues:
            # Create the output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Save to CSV
            output_file = os.path.join(output_dir, 'missing_pvalues.csv')
            df = pd.DataFrame(missing_pvalues, columns=['Node', 'Description', 'In Degree', 'Out Degree'])
            df.to_csv(output_file, index=False)
            logger.info(f"Saved {len(missing_pvalues)} missing p-values to {output_file}")
        self.missing_pvalues = missing_pvalues
        self.missing_pvalues_count = len(missing_pvalues)
        self.set_unmissing_neighbors_and_children()
    
    def set_unmissing_neighbors_and_children(self):
        '''calculate the unmissing neighbors'''
        for node in self.nodes():
            self.nodes[node]['unmissing_neighbors'] = [n for n in self.successors(node) if 'p_value' in self.nodes()[n]] + [n for n in self.predecessors(node) if 'p_value' in self.nodes()[n]]
            self.nodes[node]['unmissing_children'] = [c for c in self.successors(node) if 'p_value' in self.nodes()[c]]
            self.nodes[node]['unmissing_descendants'] = [d for d in nx.descendants(self, node) if 'p_value' in self.nodes()[d]]
            self.nodes[node]['neighbors'] = list(self.successors(node)) + list(self.predecessors(node))
            self.nodes[node]['children'] = list(self.successors(node))
            self.nodes[node]['descendants'] = list(nx.descendants(self, node))

    def get_kinase(self):
        '''get the kinase nodes'''
        return [node for node in self.nodes() if self.nodes[node]['Description'] == 'Kinase']
    
    def get_unmissing_neighbors(self, nodes):
        '''get the unmissing neighbors of the nodes'''
        if isinstance(nodes, str):
            nodes = [nodes]
        return [n for node in nodes for n in self.nodes[node]['unmissing_neighbors']]
    
    def get_unmissing_children(self, nodes):
        '''get the unmissing children of the nodes'''
        if isinstance(nodes, str):
            nodes = [nodes]
        return [c for node in nodes for c in self.nodes[node]['unmissing_children']]

    def get_descendants(self, node_set, rejection_set=None, exclude_missing=True):
        '''calculate the number of the unmissing descendats of the nodes in the set'''
        if isinstance(node_set, str):
            node_set = [node_set]
        descendants = set()
        for node in node_set:
            if exclude_missing:
                descendants.update(set(self.nodes[node]['unmissing_descendants']))
            else:
                descendants.update(set(self.nodes[node]['descendants']))
        if rejection_set:
            return descendants.intersection(rejection_set)
        else:
            return descendants
        
    def get_rejected_connected_descendants(self, node_set, rejection_set):
        '''get the connected descendants of the node_set that are rejected'''
        if isinstance(node_set, str):
            node_set = [node_set]
        descendants = self.get_descendants(node_set, rejection_set, exclude_missing=False)
        # Create a new set that includes both descendants and node_set
        all_nodes = descendants.union(set(node_set))
        subgraph = self.subgraph(all_nodes).copy()
        subgraph.set_descendent()
        return subgraph.get_descendants(node_set, rejection_set, exclude_missing=False).union(set(node_set))
        
    def get_successors(self, node_set, rejection_set=None, exclude_missing=True):
        '''calculate the number of the unmissing descendats of the nodes in the set'''
        if isinstance(node_set, str):
            node_set = [node_set]
        successors = set()
        for node in node_set:
            if exclude_missing:
                successors.update(set(self.nodes[node]['unmissing_children']))
            else:
                successors.update(set(self.nodes[node]['children']))
        if rejection_set:
            return successors.intersection(rejection_set)
        else:
            return successors
        
    def save_original_data(self, path):
        '''save the original data'''
        data = pd.read_csv(path)
        for node in self.nodes():
            try:
                self.nodes[node]['original_data'] = data[data['Molecule']==node].iloc[0].tolist()[1:-1]
            except:
                pass

    def delete_loops(self):
        '''delete the loops in the network'''
        # first, remove the edges between kinases
        for node in self.nodes():
            if self.nodes[node]['Description'] == 'Kinase':
                neighbors = list(self.neighbors(node))
                for neighbor in neighbors:
                    if self.nodes[neighbor]['Description'] == 'Kinase':
                        self.remove_edge(node, neighbor)
        # check if there are still loops
        # cycles = list(nx.simple_cycles(self))
        sccs = list(nx.strongly_connected_components(self))
        cyclic_components = [scc for scc in sccs if len(scc) > 1]
        print("Cyclic components:", cyclic_components)
        # print('Remaining loops:', cycles)

    def merge(self, nodes):
        """
        Merge multiple nodes into a single node in the network.
        
        Args:
            nodes: List of node names to merge
            
        Returns:
            The name of the merged node
        """
        if not nodes:
            return None
        
        # Filter out nodes that don't exist in the network
        nodes = [node for node in nodes if node in self.nodes()]
        if not nodes:
            return None
            
        # Create the merged node name
        merged_name = '_'.join(sorted(nodes))
        
        # If merged node already exists, skip
        if merged_name in self.nodes():
            return merged_name
        
        # Get all edges connected to any of the nodes to be merged
        edges_to_add = []
        for node in nodes:
            # Get all edges where this node is either source or target
            for pred in self.predecessors(node):
                if pred not in nodes:  # Don't add edges between nodes being merged
                    edges_to_add.append((pred, merged_name))
            for succ in self.successors(node):
                if succ not in nodes:  # Don't add edges between nodes being merged
                    edges_to_add.append((merged_name, succ))
        
        # Add the merged node with combined attributes
        self.add_node(merged_name)
        
        # Combine attributes from all nodes (excluding problematic attributes)
        for node in nodes:
            for attr, value in self.nodes[node].items():
                # Skip attributes that contain node references that will become invalid
                if attr in ['neighbors', 'children', 'descendants', 'unmissing_neighbors', 
                           'unmissing_children', 'unmissing_descendants']:
                    continue
                    
                if attr not in self.nodes[merged_name]:
                    self.nodes[merged_name][attr] = value
                elif isinstance(value, (list, set)):
                    # For list/set attributes, combine them
                    if isinstance(self.nodes[merged_name][attr], list):
                        self.nodes[merged_name][attr].extend(value)
                    else:
                        self.nodes[merged_name][attr].update(value)
        
        # Add all the edges to the merged node (remove duplicates)
        edges_to_add = list(set(edges_to_add))
        for edge in edges_to_add:
            self.add_edge(*edge)
        
        # Remove the original nodes
        self.remove_nodes_from(nodes)
        
        return merged_name
    
    def update_node_attributes(self):
        """
        Update all node attributes that depend on network structure.
        Call this after merging operations are complete.
        """
        for node in self.nodes():
            self.nodes[node]['unmissing_neighbors'] = [n for n in self.neighbors(node) if 'p_value' in self.nodes()[n]]
            self.nodes[node]['unmissing_children'] = [c for c in self.successors(node) if 'p_value' in self.nodes()[c]]
            self.nodes[node]['unmissing_descendants'] = [d for d in nx.descendants(self, node) if 'p_value' in self.nodes()[d]]
            self.nodes[node]['neighbors'] = list(self.neighbors(node))
            self.nodes[node]['children'] = list(self.successors(node))
            self.nodes[node]['descendants'] = list(nx.descendants(self, node))

    def create_adjacency_matrix(self):
        """
        Create an adjacency matrix for the network with indexed nodes.
        
        Returns:
            tuple: (adjacency_matrix, node_to_index, index_to_node)
            - adjacency_matrix: numpy array of shape (n_nodes, n_nodes)
            - node_to_index: dictionary mapping node names to indices
            - index_to_node: dictionary mapping indices to node names
        """
        # Create node to index mapping
        nodes = sorted(list(self.nodes()))
        self.node_to_index = {node: idx for idx, node in enumerate(nodes)}
        self.index_to_node = {idx: node for node, idx in self.node_to_index.items()}
        
        # Initialize adjacency matrix
        n_nodes = len(nodes)
        self.adjacency_matrix = np.zeros((n_nodes, n_nodes), dtype=int)
        
        # Fill adjacency matrix
        for source, target in self.edges():
            i = self.node_to_index[source]
            j = self.node_to_index[target]
            self.adjacency_matrix[i, j] = 1

        # add index to the nodes
        for node in self.nodes():
            self.nodes[node]['index'] = self.node_to_index[node]
        
    def save_adjacency_matrix(self, output_path):
        """
        Save the adjacency matrix and node mappings to files.
        
        Args:
            output_path: Directory path to save the files
        """
        adj_matrix, node_to_idx, idx_to_node = self.create_adjacency_matrix()
        
        # Save adjacency matrix
        np.save(f"{output_path}/adjacency_matrix.npy", adj_matrix)
        
        # Save node mappings
        with open(f"{output_path}/node_to_index.txt", 'w') as f:
            for node, idx in node_to_idx.items():
                f.write(f"{node}\t{idx}\n")
                
        with open(f"{output_path}/index_to_node.txt", 'w') as f:
            for idx, node in idx_to_node.items():
                f.write(f"{idx}\t{node}\n")
                
        return adj_matrix, node_to_idx, idx_to_node
    
    def save_to_graphml(self, output_path):
        """
        Save the network to a GraphML file.
        
        Args:
            output_path: Path where the GraphML file will be saved
        """
        # Create a copy of the graph to avoid modifying the original
        graph_copy = self.copy()
        
        # Convert non-serializable attributes to strings
        for node in graph_copy.nodes():
            for attr, value in graph_copy.nodes[node].items():
                if isinstance(value, (list, set, tuple)):
                    # Convert lists, sets, tuples to comma-separated strings
                    graph_copy.nodes[node][attr] = ','.join(map(str, value))
                elif not isinstance(value, (str, int, float, bool)):
                    # Convert other non-serializable types to string
                    graph_copy.nodes[node][attr] = str(value)
        
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Write the graph to GraphML format
        nx.write_graphml(graph_copy, output_path)
    
    def get_depth(self, node):
        '''get the depth of the node'''
        return self.nodes[node]['depth']

    def get_depths_for_all_nodes(self):
        sources = self.get_kinase()
        # Initialize all nodes with infinite depth
        nx.set_node_attributes(self, 9999, name="depth")
        
        # Calculate shortest path from all kinase sources
        depths = {}
        for source in sources:
            try:
                source_depths = nx.single_source_shortest_path_length(self, source)
                for node, depth in source_depths.items():
                    if node not in depths or depth < depths[node]:
                        depths[node] = depth
            except:
                continue
        
        # Set the calculated depths
        nx.set_node_attributes(self, depths, name="depth")

    def set_kinase_index(self):
        '''set the index of the kinase'''
        index = 0
        for node in self.nodes():
            if self.nodes[node]['Description'] == 'Kinase':
                self.nodes[node]['index'] = index
                index += 1

    def process_indisctinguishable_kinases(self):
        '''process the indistinguishable kinases'''
        undistinguishable_kinases = self.find_undistinguishable_kinases()
        for kinase in undistinguishable_kinases:
            self.merge(kinase.split(','))
                     
    def get_kinase_parents(self, node):
        """
        Get all parents of a node that have 'Description' equal to 'Kinase'.
        """
        kinase_parents = []
        for parent in self.predecessors(node):
            if self.nodes[parent]['Description'] == 'Kinase':
                kinase_parents.append(parent)
        return kinase_parents
    
    def get_kinase_parents_index(self, node):
        '''get the index of the kinase parents'''
        return [self.nodes[parent]['index'] for parent in self.get_kinase_parents(node)]
        
    def get_kinase_ancestors(self, node):
        """
        Get all ancestors of a node that have 'Description' equal to 'Kinase'.
        Only considers ancestors that have valid p-values.
        
        Args:
            node: The node to find kinase ancestors for
            
        Returns:
            list: List of kinase ancestor nodes
        """
        kinase_ancestors = []
        
        # Get all ancestors (nodes that can reach this node)
        try:
            all_ancestors = nx.ancestors(self, node)
        except:
            # If there's an issue with finding ancestors, return empty list
            return kinase_ancestors
        
        # Filter ancestors to only include kinases
        for ancestor in all_ancestors:
            if self.nodes[ancestor].get('Description') == 'Kinase':
                kinase_ancestors.append(ancestor)
        return kinase_ancestors
    
    def get_kinase_ancestors_index(self, node):
        '''get the index of the kinase ancestors'''
        return [self.nodes[ancestor]['index'] for ancestor in self.get_kinase_ancestors(node)]
    
    def find_undistinguishable_kinases(self):
        '''find the undistinguishable kinases'''
        children = {}
        undistinguishable_kinases = []
        kinases = self.get_kinase()
        for kinase in kinases:
            if not ','.join(self.get_unmissing_children(kinase)) in children:
                children[','.join(self.get_unmissing_children(kinase))] = kinase
            else:
                children[','.join(self.get_unmissing_children(kinase))] += ','+kinase
        for children_list,kinases in children.items():
            if len(kinases.split(',')) > 1:
                undistinguishable_kinases.append(kinases)
        return undistinguishable_kinases

def load_config(path):
    with open(path, "r") as f: 
        return yaml.safe_load(f)
    
def z_transform(x):
    return norm.ppf(x)

def plot_distribution(case_df, control_df):
    plt.figure(figsize=(15, 10))
    for i in range(6):
        x1 = case_df.iloc[i].tolist()
        x2 = control_df.iloc[i].tolist()
        plt.subplot(2, 3, i+1)
        sns.kdeplot(x1[2:], fill=True, color="blue", alpha=0.5)
        plt.title(x1[0]+' '+x1[1][:4])
        plt.xlabel("Value")
        plt.ylabel("Density")
        plt.hist(x1[2:], bins=20, density=True, alpha=0.3, color="gray", label="Case")
        sns.kdeplot(x2[2:], fill=True, color="red", alpha=0.5)
        plt.title(x2[0]+' '+x1[1][:4])
        plt.xlabel("Value")
        plt.ylabel("Density")
        plt.hist(x2[2:], bins=20, density=True, alpha=0.3, color="pink", label="Control")
    plt.tight_layout()
    # plt.show()  # COMMENTED OUT TO PREVENT AUTOMATIC PLOTTING
    case_df.head()

def rgb_to_hex(rgb):
    return '#{:02x}{:02x}{:02x}'.format(
        int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)
    )

def get_case_control_data(member_path, data_path):
    '''get the data of case and control'''
    member = pd.read_excel(member_path)
    case_ID = member[member['Diag']=='Sz']['ID'].tolist()
    control_ID = member[member['Diag']=='C']['ID'].tolist()
    df = pd.read_csv(data_path)
    case_df = df[['Molecule', 'Description']+case_ID]
    control_df = df[['Molecule', 'Description']+control_ID]
    return case_df, control_df

def p_value_calculation(case_df, control_df):
    '''calculate the p-value for each molecule, return a dictionary'''
    p_values = {}
    for i in range(len(case_df)):
        group1 = np.array(case_df.iloc[i, 2:].dropna().tolist())
        group2 = np.array(control_df.iloc[i, 2:].dropna().tolist())
        if len(group1) == 0 or len(group2) == 0:
            continue
        t_stat, p_value = stats.ttest_ind(group1, group2)
        p_values[case_df.iloc[i, 0]] = p_value    
    return p_values

def p_value_average(set, R_current, p_values):
    '''calculate the average p-value of the set'''
    total, missing = 0, 0
    for node in set:
        if node not in p_values:
            missing += 1
        else:
            total += p_values[node]
    if len(set) - missing == 0:
        return 1
    else:
        return total / (len(set) - missing)

def mark_rejected_nodes(network, rejection_set, attribute='reject'):
    """
    Mark nodes in the network based on whether they are in the rejection set
    
    Parameters:
    - network: NetworkX graph object
    - rejection_set: A set or list containing rejected nodes
    """
    for node in network.nodes():
        if node in rejection_set:
            # Node is in the rejection set, set 'reject' attribute to 0.00001
            network.nodes[node][attribute] = 0.00001
        elif 'p_value' in network.nodes[node]:
            # Node is not in the rejection set, set 'reject' attribute to 1, if p-value isn't missing; otherwise, don't set
            network.nodes[node][attribute] = 1
    
    return network

def visualize_rejection(network, rejection, experiment_name, importance_nodes):
    '''visualize the rejection set'''
    network2 = mark_rejected_nodes(network, rejection, experiment_name)
    visualize_pvalues(network2, experiment_name, attribute=experiment_name, name='rejection', importance_nodes=importance_nodes)

def visualize_category(network, experiment_name, importance_nodes=None):
    '''visualize the category of the network'''
    description_dict = {'Kinase':0, 'Phosphosite':1, 'Protein':2, 'Small-molecule':3}
    for node in network.nodes():
        network.nodes[node]['Description_value'] = description_dict[network.nodes[node]['Description']]
    visualize_pvalues(network, experiment_name, attribute='Description_value', name='category', importance_nodes=importance_nodes)
    
def save_result(logger, rejection, peel_off, config, network):
    '''save the result'''
    # Save rejection set to CSV
    rows = []
    for node in rejection:
        p_value = network.nodes[node].get('p_value', 0.00001)
        rows.append({
            'node': node, 
            'p_value': p_value, 
            'description': network.nodes[node]['Description']
        })
    
    df = pd.DataFrame(rows)
    df.to_csv(f"results/{config['experiment_name']}/rejection_set.csv", index=False)
    logger.info(f"Rejection set saved to 'results/{config['experiment_name']}/rejection_set.csv'")
    logger.info(f"When evaluating the rejection subgraph, we include the missing nodes.")
    
    # Create and save subgraph
    subgraph = network.subgraph(rejection)
    # subgraph_path = f"results/{config['experiment_name']}/rejection_subgraph.gexf"
    # nx.write_gexf(subgraph, subgraph_path)
    # logger.info(f"Subgraph saved to {subgraph_path}")
    logger.info(f"--------------------------------")
    if len(rejection) > 0:
        importance_nodes = get_basic_info(logger, subgraph, config['experiment_name'])
        logger.info(f"--------------------------------")
        # Visualize rejection - COMMENTED OUT TO PREVENT AUTOMATIC PLOTTING
        # visualize_rejection(network, rejection, config['experiment_name'], importance_nodes)
    else:
        logger.info(f"The rejection set is empty")
    logger.info(f"Excluding the missingness, the rejection rate is {(len(rejection)-network.missing_pvalues_count)/(network.number_of_nodes()-network.missing_pvalues_count)}")
    logger.info(f"The number of nodes in the peel off set is {len(peel_off)}")
    with open(f"results/{config['experiment_name']}/peel_off.txt", "w") as f:
        f.write(', '.join(peel_off))

def visualize_rejection_comparison(logger, sets_dict, experiment_name):
    '''
    Create a Venn diagram to visualize the overlap between rejection sets
    
    Args:
        sets_dict: Dictionary with set labels as keys and sets as values
        output_path: Path to save the output image
    
    Returns:
        bool: Whether the visualization was successful
    '''
    if len(sets_dict) == 2:
        plt.figure(figsize=(8, 6))
        labels = list(sets_dict.keys())
        sets = list(sets_dict.values())
        venn2([sets[0], sets[1]], labels)
        
    elif len(sets_dict) == 3:
        plt.figure(figsize=(10, 7))
        labels = list(sets_dict.keys())
        sets = list(sets_dict.values())
        venn3([sets[0], sets[1], sets[2]], labels)
        
    else:
        logger.info(f"Venn diagram visualization only supports 2 or 3 sets, but {len(sets_dict)} were provided.")
        return False
        
    plt.title('Overlap between rejection sets')
    output_path = f"results/{experiment_name}/rejection_sets_comparison.png"
    plt.savefig(output_path)
    logger.info(f"Venn diagram saved to '{output_path}'")
    return True

def reachable_diameter(G):
    '''calculate the reachable diameter'''
    max_len = 0
    for source in G.nodes:
        sp_lengths = nx.single_source_shortest_path_length(G, source)
        # ignore the self
        if sp_lengths:
            local_max = max([l for target, l in sp_lengths.items() if target != source], default=0)
            max_len = max(max_len, local_max)
    return max_len

def get_basic_info(logger, network, experiment_name):
    '''get the basic info of the network'''
    logger.info(f"The network has {network.number_of_nodes()} nodes and {network.number_of_edges()} edges")
    
    # Calculate average degree
    degrees = [d for n, d in network.degree()]
    avg_degree = sum(degrees) / len(degrees)
    logger.info(f"Average node degree: {avg_degree:.2f}")
    
    # Calculate number of weakly connected components (ignoring direction)
    weak_components = list(nx.weakly_connected_components(network))
    logger.info(f"Number of weakly connected components: {len(weak_components)}")
    
    # Calculate number of strongly connected components (considering direction)
    strong_components = list(nx.strongly_connected_components(network))
    logger.info(f"Number of strongly connected components: {len(strong_components)}")
    
    # Print size of the largest component
    largest_weak = max(weak_components, key=len)
    largest_strong = max(strong_components, key=len)
    logger.info(f"Size of largest weakly connected component: {len(largest_weak)} nodes")
    logger.info(f"Size of largest strongly connected component: {len(largest_strong)} nodes")
    
    # Identify nodes with the largest in-degree
    top_k = 10
    in_degrees = dict(network.in_degree())
    top_in_degree_nodes = sorted(in_degrees.items(), key=lambda x: x[1], reverse=True)[:top_k]
    
    logger.info("\nTop %d nodes by in-degree:", top_k)
    for node, degree in top_in_degree_nodes:
        description = network.nodes[node].get('Description', 'Unknown')
        logger.info(f"Node: {node}, In-degree: {degree}, Description: {description}")
    
    # Identify nodes with the largest out-degree
    out_degrees = dict(network.out_degree())
    top_out_degree_nodes = sorted(out_degrees.items(), key=lambda x: x[1], reverse=True)[:top_k]
    
    logger.info("\nTop %d nodes by out-degree:", top_k)
    for node, degree in top_out_degree_nodes:
        description = network.nodes[node].get('Description', 'Unknown')
        logger.info(f"Node: {node}, Out-degree: {degree}, Description: {description}")
    
    # Identify nodes with the largest total degree
    total_degrees = dict(network.degree())
    top_total_degree_nodes = sorted(total_degrees.items(), key=lambda x: x[1], reverse=True)[:top_k]
    
    logger.info("\nTop %d nodes by total degree (in + out):", top_k)
    for node, degree in top_total_degree_nodes:
        description = network.nodes[node].get('Description', 'Unknown')
        in_deg, out_deg = in_degrees[node], out_degrees[node]
        logger.info(f"Node: {node}, Total: {degree} (In: {in_deg}, Out: {out_deg}), Description: {description}")
    
    # Calculate average shortest path length
    avg_shortest_path_length = average_shortest_path_length_ignore_unreachable(network)
    logger.info(f"Average shortest path length: {avg_shortest_path_length:.2f}")

    # Calculate reachable diameter
    reachable_diam = reachable_diameter(network)
    logger.info(f"Reachable diameter: {reachable_diam:.2f}")

    # Calculate closeness centrality
    closeness_centrality = nx.closeness_centrality(network)
    top_k_closeness_centrality_nodes = sorted(closeness_centrality.items(), key=lambda x: x[1], reverse=True)[:top_k]
    logger.info("\nTop %d nodes by closeness centrality:", top_k)
    for node, centrality in top_k_closeness_centrality_nodes:
        description = network.nodes[node].get('Description', 'Unknown')
        logger.info(f"Node: {node}, Closeness centrality: {centrality:.4f}, Description: {description}")

    # articulation_points
    component_increase = component_increase_by_node(network)
    top_k_component_increase_nodes = sorted(
                                                ((k, v) for k, v in component_increase.items() if v != 0),
                                                key=lambda x: x[1],
                                                reverse=True
                                            )[:top_k]
    logger.info("\nTop %d nodes by component increase:", top_k)
    for node, increase in top_k_component_increase_nodes:
        description = network.nodes[node].get('Description', 'Unknown')
        logger.info(f"Node: {node}, Component increase: {increase}, Description: {description}")

    importance_nodes = top_k_component_increase_nodes + top_k_closeness_centrality_nodes + top_total_degree_nodes + top_in_degree_nodes + top_out_degree_nodes
    importance_nodes = set(list(map(lambda x: x[0], importance_nodes)))
    return importance_nodes

def count_components_after_removal(G, node):
    G_copy = G.copy()
    G_copy.remove_node(node)
    return len(list(nx.strongly_connected_components(G)))

def component_increase_by_node(G):
    base_components = len(list(nx.strongly_connected_components(G)))
    increase = {}
    for node in G.nodes:
        new_components = count_components_after_removal(G, node)
        increase[node] = new_components - base_components
    return increase
        
def effective_diameter(G, percentile=90):
    '''calculate the effective diameter'''
    path_lengths = []
    for source in G.nodes:
        sp_lengths = nx.single_source_shortest_path_length(G, source)
        # only keep the path length with source != target
        path_lengths.extend([l for target, l in sp_lengths.items() if target != source])
    
    if not path_lengths:
        return float('inf')
    
    return np.percentile(path_lengths, percentile)

def average_shortest_path_length_ignore_unreachable(G):
    '''calculate the average shortest path length, ignore the unreachable nodes'''
    lengths = []
    for source in G.nodes:
        sp_lengths = nx.single_source_shortest_path_length(G, source)
        # ignore the self (length is 0), only keep the target with path
        lengths.extend([l for target, l in sp_lengths.items() if target != source])
    return sum(lengths) / len(lengths) if lengths else float('inf')

def load_data_and_network(logger, member_path, residual_path, path, category_path, network_name, plot=False):
    """Load data and set up the network."""
    # network = load_network(network_name, path, category_path, plot=plot)
    network = load_network_from_data(logger)
    case_df, control_df = get_case_control_data(member_path, residual_path)
    logger.info("finished loading data and network")
    p_values = p_value_calculation(case_df, control_df)
    # network.save_original_data(residual_path)
    nx.set_node_attributes(network, p_values, 'p_value')
    network.check_missing_pvalues(logger)
    network.set_unmissing_neighbors_and_children()
    return case_df, control_df, network, p_values

def nx_to_graphviz(G):
    """Convert a NetworkX graph to a Graphviz graph."""
    return nx.nx_agraph.to_agraph(G)

def load_network_from_data(logger):
    '''load the network from the data'''
    network_path = 'complex_network.graphml'
    logger.info(f"Loading network from {network_path}")
    network_nx = nx.read_graphml(network_path)
    network = my_network.from_graph(network_nx)
    network.process_indisctinguishable_kinases() # merge the indistinguishable kinases
    # network.set_descendent()
    network.set_kinase_index()
    network.get_depths_for_all_nodes()
    return network

def load_network(network_name, raw_data_path, category_path, plot=False, directed=True):
    """Load a network from a CSV file and visualize it."""
    raw_data = pd.read_csv(raw_data_path)
    network = nx.from_pandas_edgelist(raw_data, source="from", target="to", edge_attr="interaction_type", create_using=my_network if directed else None)
    network.set_descendent()
    network.process_indisctinguishable_kinases()
    network.set_kinase_index()
    network.get_depths_for_all_nodes()
    network.name = network_name
    category = pd.read_csv(category_path)
    for node in network.nodes():
        network.nodes[node]['Description'] = category[category['Molecule'] == node]['Description'].values[0]

    if plot:
        a = nx_to_graphviz(network)
        a.draw(f"result/{network.name}.png", prog="dot")
        for node in a.nodes():
            node.attr['label'] = ""
        a.draw(f"result/{network.name}_without_label.png", prog="dot")

    return network

def analyze_connectivity(logger, network):
    '''analyze the connectivity of the network'''
    # Check if the graph is strongly connected
    if nx.is_strongly_connected(network):
        logger.info("The graph is strongly connected")
    else:
        logger.info("The graph is not strongly connected")

    # Check if the graph is weakly connected
    if nx.is_weakly_connected(network):
        logger.info("The graph is weakly connected")
    else:
        logger.info("The graph is not weakly connected")

    # Get the strongly connected components
    strong_components = list(nx.strongly_connected_components(network))
    logger.info("Strongly connected components: %s", strong_components)
    logger.info(f"There are {len(strong_components)} strongly connected components")

    # Get the weakly connected components
    weak_components = list(nx.weakly_connected_components(network))
    logger.info("Weakly connected components: %s", weak_components)
    logger.info(f"There are {len(weak_components)} weakly connected components")

def graph_metric_of_rejection(logger, network, rejection, rejection_name):
    """
    Input: A directed graph G (nx.DiGraph)
    Output: A dictionary with structural connectivity and reachability metrics

    Add necessary missing p-values to the network to increase the connectivity of the rejection network
    """
    # add missing p-values to the rejection network who is the neighbor of the rejection nodes
    for node in rejection:
        for neighbor in list(network.neighbors(node)):
            if not 'p_value' in network.nodes[neighbor]:
                rejection.add(neighbor)
    G = network.subgraph(rejection).copy()
    
    n = G.number_of_nodes()
    
    # Handle empty graph case
    if n == 0:
        logger.warning(f"Empty rejection network for {rejection_name}")
        return {
            'name': rejection_name,
            'num_strongly_connected_components': 0,
            'max_strong_component_ratio': 0.0,
            'num_weakly_connected_components': 0,
            'global_reachability_ratio': 0.0,
            'average_reachable_ratio': 0.0,
            'modularity': 0.0
        }
    
    # Strongly connected components (SCC)
    sccs = list(nx.strongly_connected_components(G))
    num_scc = len(sccs)
    max_scc_size = max(len(comp) for comp in sccs)
    max_scc_ratio = max_scc_size / n

    # Weakly connected components (WCC) - ignore edge direction
    wccs = list(nx.weakly_connected_components(G))
    num_wcc = len(wccs)

    # Reachability (number of reachable node pairs)
    reachable_pairs = 0
    total_pairs = n * (n - 1)
    reach_ratios = []

    for node in G.nodes():
        reachable = nx.descendants(G, node)  # Nodes reachable from current node (excluding itself)
        reachable_count = len(reachable)
        reachable_pairs += reachable_count
        reach_ratios.append(reachable_count / (n - 1) if n > 1 else 0)

    global_reachability = reachable_pairs / total_pairs if total_pairs > 0 else 0
    avg_reach_ratio = np.mean(reach_ratios)
    
    # Calculate modularity using Leiden algorithm
    # First, convert to undirected graph for Leiden
    G_undirected = G.to_undirected()
    
    # Calculate modularity if graph is not empty and has edges
    try:
        # Convert to igraph for Leiden
        # Create mapping from nodes to integer indices
        nx_nodes = list(G_undirected.nodes())
        node_to_idx = {node: i for i, node in enumerate(nx_nodes)}
        
        # Convert edges using the mapping
        nx_edges_mapped = [(node_to_idx[u], node_to_idx[v]) for u, v in G_undirected.edges()]
        g_ig = ig.Graph(n=len(nx_nodes), edges=nx_edges_mapped, directed=False)
        
        # Run Leiden community detection
        leiden_partition = g_ig.community_leiden(
            objective_function="modularity",
            resolution=1.0,
            beta=0.01,
            n_iterations=10
        )
        
        # Get modularity score
        modularity = leiden_partition.modularity
    except Exception as e:
        logger.warning(f"Error calculating modularity for {rejection_name}: {e}")
        modularity = 0.0

    return {'name': rejection_name,
        'num_strongly_connected_components': num_scc,
        'max_strong_component_ratio': max_scc_ratio,
        'num_weakly_connected_components': num_wcc,
        'global_reachability_ratio': global_reachability,
        'average_reachable_ratio': avg_reach_ratio,
        'modularity': modularity
    }
    
def clustered_layout(G, partition, scale=5.0, seed=42):
    """
    Create a layout based on community detection results, with cluster spacing proportional to cluster size.
    Each cluster is positioned in a circle, with the radius proportional to the number of clusters,
    and the spacing between clusters proportional to their sizes.
    """
    np.random.seed(seed)
    pos = {}
    clusters = {}

    # 1. Group nodes by community
    for node, comm_id in partition.items():
        clusters.setdefault(comm_id, []).append(node)

    # Calculate total nodes for scaling
    total_nodes = sum(len(nodes) for nodes in clusters.values())
    
    # Sort clusters by size for better placement
    sorted_clusters = sorted(clusters.items(), key=lambda x: len(x[1]), reverse=True)
    num_clusters = len(sorted_clusters)
    
    if num_clusters == 0:
        return nx.spring_layout(G, seed=seed)
    
    # 2. Position each cluster in a circle, with spacing proportional to cluster size
    for i, (comm_id, nodes) in enumerate(sorted_clusters):
        subgraph = G.subgraph(nodes)
        cluster_size = len(nodes)
        
        # Calculate angle based on cluster size proportion
        angle = 2 * np.pi * i / num_clusters
        
        # Calculate radius based on cluster size, but with less extreme scaling
        # Base radius is smaller, and size-based variation is reduced
        radius = scale * (1 + 1 * cluster_size / total_nodes)
        
        # Calculate cluster center
        cluster_center = np.array([radius * np.cos(angle), radius * np.sin(angle)])
        
        # Create internal layout for the cluster
        sub_pos = nx.spring_layout(subgraph, seed=seed, k=1.5)
        
        # Scale the internal layout based on cluster size, but with less extreme scaling
        internal_scale = (1 + 1 * np.sqrt(cluster_size / total_nodes)) * scale
        
        # Position nodes around the cluster center
        for node in subgraph.nodes():
            # Scale and offset the position
            pos[node] = cluster_center + internal_scale * (sub_pos[node] - np.array([0.5, 0.5]))

    return pos

def importance_layout(network, importance_nodes, radius=1.5, center_scale=1):
    pos = nx.spring_layout(network)
    center = np.mean(np.array(list(pos.values())), axis=0)

    # unimportant nodes are compressed away from the center
    for node in pos:
        if node not in importance_nodes:
            pos[node] = center + center_scale * (pos[node] - center)

    # lay the important nodes along the circle
    importance_nodes = [n for n in importance_nodes if n in pos]
    angle_step = 2 * np.pi / max(1, len(importance_nodes))
    
    for i, node in enumerate(importance_nodes):
        angle = i * angle_step
        offset = radius * np.array([np.cos(angle), np.sin(angle)])
        pos[node] = center + offset

    return pos

def visualize_clustering(logger, network, communities, partition, experiment_name, importance_nodes=None):
    community_networks = {}
    for community_id, nodes in communities.items():
        if len(nodes) <= 1:  # Skip communities with just one node
            continue
        
        # Create subgraph for this community
        subgraph = network.subgraph(nodes).copy()
        community_networks[community_id] = subgraph
    
        
        # Save basic info about the community
        logger.info(f"\nCommunity {community_id} info:")
        logger.info(f"Number of nodes: {len(nodes)}")
        logger.info(f"Number of edges: {subgraph.number_of_edges()}")
        
        # Visualize category for this community - COMMENTED OUT TO PREVENT AUTOMATIC PLOTTING
        # visualize_category(subgraph, f'{experiment_name}/community_{community_id}', importance_nodes=importance_nodes)
        
        # Export the community as GEXF for external visualization
        # nx.write_gexf(subgraph, os.path.join(community_dir, 'network.gexf'))

    # Visualize the entire network with community coloring - COMMENTED OUT TO PREVENT AUTOMATIC PLOTTING
    # plt.figure(figsize=(12, 8))
    
    # # Calculate node sizes based on degrees (reduced size multiplier)
    # node_sizes = [network.degree(node) * 5 + 5 for node in network.nodes()]
    
    # # Get unique community IDs for coloring
    # community_ids = set(partition.values())
    # colors = plt.cm.rainbow(np.linspace(0, 1, len(community_ids)))
    # community_color_map = {comm_id: colors[i] for i, comm_id in enumerate(sorted(community_ids))}
    
    # # Create a color map for nodes
    # node_colors = [community_color_map[partition[node]] for node in network.nodes()]
    
    # # Use the clustered layout
    # pos = clustered_layout(network, partition, scale=0.3, seed=42)
    
    # # if importance_nodes:
    # #     pos = importance_layout(network, importance_nodes)
    
    # # Draw edges with increased visibility
    # nx.draw_networkx_edges(network, pos, alpha=0.5, width=0.5)
    
    # # Draw nodes
    # nx.draw_networkx_nodes(network, pos, node_size=node_sizes, node_color=node_colors, alpha=0.7)
    
    # # Create legend elements for communities
    # legend_elements = []
    # for comm_id in sorted(community_ids):
    #     if comm_id in communities:
    #         comm_size = len(communities[comm_id])
    #         legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
    #                               label=f'Community {comm_id} ({comm_size} nodes)', 
    #                               markerfacecolor=mcolors.rgb2hex(community_color_map[comm_id]),
    #                               markersize=10))
    
    # # Add community legend
    # plt.legend(handles=legend_elements, title="Communities", 
    #           loc='upper left', bbox_to_anchor=(1, 1), fontsize='small')
    
    # plt.title("Network with Community Structure")
    # plt.axis('off')
    # plt.tight_layout()
    # plt.savefig(f"results/{experiment_name}/full_network_communities.png", dpi=300, bbox_inches='tight')
    # plt.close()

def visualize_pvalues(G, experiment_name, attribute='p_value', name='p_value', importance_nodes=None):
    """
    Visualize the network with colors based on p-values using a Louvain community-based layout.
    
    Parameters:
    - G: NetworkX graph (can be directed)
    - attribute: The node attribute to use for coloring (default: 'p_value')
    """
    # Set the colors
    colors = list(mcolors.TABLEAU_COLORS)  # Using Matplotlib's predefined colors
    color_map = {'Kinase':colors[0], 'Phosphosite':colors[1], 'Protein':colors[2], 'Small-molecule':colors[3]}

    # Get p-values
    p_values = nx.get_node_attributes(G, attribute)
    
    # Convert to undirected graph for community detection
    G_undirected = G.to_undirected()
    
    # Get community partition using Louvain method on undirected graph
    partition = community_louvain.best_partition(G_undirected)
    
    # Create figure with subplots
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Calculate node sizes based on degrees
    node_sizes = [G.degree(node) * 5 + 5 for node in G.nodes()]
    
    node_colors = []
    # take care of the p-value missingness
    for node in G.nodes():
        if node in p_values and p_values[node] is not None:
            if name == 'category':
                color = colors[G.nodes[node]['Description_value']]
            elif name == 'rejection':
                # Use a red-blue colormap for rejection visualization
                if p_values[node] < 0.001:  # Rejected nodes
                    color = 'red'
                elif p_values[node] > 0.999:  # Non-rejected nodes
                    color = 'lightblue'
                else:  # Missing p-values
                    color = '#D8BFD8'
            else:
                color = cm.Blues(1 - p_values[node])
        else:
            color = '#D8BFD8'  # Default color for nodes without p-values
        node_colors.append(color)

    # Get clustered layout
    pos = clustered_layout(G, partition, scale=0.3, seed=42)
    
    if importance_nodes:
        pos = importance_layout(G, importance_nodes)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, alpha=0.5, width=0.5, arrows=True, arrowsize=1, ax=ax)
    
    # Draw nodes (label the importance nodes)
    if importance_nodes:
        labels = {node: str(node) for node in set(importance_nodes) & set(G.nodes())}
    else:
        labels = {}

    # Draw nodes
    nx.draw_networkx_nodes(G, pos,
                        node_size=node_sizes,
                        node_color=node_colors,
                        alpha=0.7,
                        ax=ax)

    # Draw labels of the importance nodes
    if labels:
        nx.draw_networkx_labels(G, pos, labels=labels, ax=ax, 
                               bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=0.5))

    if name == 'category':
        # Create a legend
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                            label=desc, markerfacecolor=color, markersize=10)
                            for desc, color in color_map.items()]
        plt.legend(handles=legend_elements, title="Descriptions")
    elif name == 'rejection':
        # Create legend for rejection visualization
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', label='Rejected', 
                      markerfacecolor='red', markersize=10),
            plt.Line2D([0], [0], marker='o', color='w', label='Not Rejected', 
                      markerfacecolor='lightblue', markersize=10),
            plt.Line2D([0], [0], marker='o', color='w', label='Missing p-value', 
                      markerfacecolor='#D8BFD8', markersize=10)
        ]
        plt.legend(handles=legend_elements, loc='upper left')
    else: 
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=plt.cm.Blues, 
                              norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('p-value')
        cbar.ax.invert_yaxis()  # Invert the colorbar to match the color scheme
        
        # Add legend entry for missing p-values
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                      label='Missing p-values', 
                                      markerfacecolor='#D8BFD8', 
                                      markersize=10)]
        plt.legend(handles=legend_elements, loc='upper left')
    
    ax.set_title(f"Network Visualization: {name.title()} ({experiment_name})")
    ax.axis('off')
    plt.show()
    # # Save the figure
    # if not os.path.exists(f"results/{experiment_name}"):
    #     os.makedirs(f"results/{experiment_name}")
    # output_path = f"results/{experiment_name}/{name}_visualization.png"
    # plt.savefig(output_path, dpi=300, bbox_inches='tight')
    # plt.close()

def plot_p_value_distribution(p_values, experiment_name, log=False):
    '''plot the p-value distribution'''
    plt.figure(figsize=(10, 6))
    if log:
        plt.hist(np.log10(np.array(list(p_values.values()))), bins=20, edgecolor='black', log=True)
    else:
        plt.hist(p_values.values(), bins=20, edgecolor='black')
    plt.title('P-value Distribution')
    plt.xlabel('P-value')
    plt.ylabel('Frequency')
    if log:
        plt.savefig(f"results/{experiment_name}/log_p_value_distribution.png", dpi=300, bbox_inches='tight')
    else:   
        plt.savefig(f"results/{experiment_name}/p_value_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()

def calculate_overlap_metrics(set1, set2, method_name, logger):
    """Calculate overlap metrics between two sets of kinases"""
    set1_set = set(set1)
    set2_set = set(set2)
    
    intersection = set1_set & set2_set
    union = set1_set | set2_set
    
    jaccard_index = len(intersection) / len(union) if len(union) > 0 else 0
    overlap_coefficient = len(intersection) / min(len(set1_set), len(set2_set)) if min(len(set1_set), len(set2_set)) > 0 else 0
    
    logger.info(f"\n{method_name} Method Comparison:")
    logger.info(f"  Original network: {len(set1_set)} kinases")
    logger.info(f"  Complex network: {len(set2_set)} kinases")
    logger.info(f"  Intersection: {len(intersection)} kinases")
    logger.info(f"  Union: {len(union)} kinases")
    logger.info(f"  Jaccard Index: {jaccard_index:.3f}")
    logger.info(f"  Overlap Coefficient: {overlap_coefficient:.3f}")
    logger.info(f"  Common kinases: {sorted(list(intersection))[:10]}{'...' if len(intersection) > 10 else ''}")
    
    return {
        'method': method_name,
        'original_count': len(set1_set),
        'complex_count': len(set2_set),
        'intersection_count': len(intersection),
        'union_count': len(union),
        'jaccard_index': jaccard_index,
        'overlap_coefficient': overlap_coefficient,
        'intersection': intersection,
        'original_only': set1_set - set2_set,
        'complex_only': set2_set - set1_set
    }

def create_kinase_comparison_visualizations(comparison_results, output_dir, logger, 
                                          selected_kinases_orig_binomial, selected_kinases_complex,
                                          selected_kinases_orig_weighted, selected_kinases_complex_weighted,
                                          selected_kinases_orig_iterative, selected_kinases_complex_iterative):
    """Create visualizations comparing the kinase selection methods between networks"""
    import matplotlib.pyplot as plt
    import os
    from matplotlib_venn import venn2
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Jaccard Index comparison
    methods = [result['method'] for result in comparison_results]
    jaccard_scores = [result['jaccard_index'] for result in comparison_results]
    
    axes[0, 0].bar(methods, jaccard_scores, color=['skyblue', 'lightcoral', 'lightgreen'])
    axes[0, 0].set_title('Jaccard Index Comparison')
    axes[0, 0].set_ylabel('Jaccard Index')
    axes[0, 0].set_ylim(0, 1)
    for i, v in enumerate(jaccard_scores):
        axes[0, 0].text(i, v + 0.01, f'{v:.3f}', ha='center')
    
    # Plot 2: Overlap Coefficient comparison
    overlap_scores = [result['overlap_coefficient'] for result in comparison_results]
    
    axes[0, 1].bar(methods, overlap_scores, color=['skyblue', 'lightcoral', 'lightgreen'])
    axes[0, 1].set_title('Overlap Coefficient Comparison')
    axes[0, 1].set_ylabel('Overlap Coefficient')
    axes[0, 1].set_ylim(0, 1)
    for i, v in enumerate(overlap_scores):
        axes[0, 1].text(i, v + 0.01, f'{v:.3f}', ha='center')
    
    # Plot 3: Set sizes comparison
    original_counts = [result['original_count'] for result in comparison_results]
    complex_counts = [result['complex_count'] for result in comparison_results]
    
    x = range(len(methods))
    width = 0.35
    
    axes[1, 0].bar([i - width/2 for i in x], original_counts, width, label='Original Network', color='skyblue')
    axes[1, 0].bar([i + width/2 for i in x], complex_counts, width, label='Complex Network', color='lightcoral')
    axes[1, 0].set_title('Number of Selected Kinases')
    axes[1, 0].set_ylabel('Number of Kinases')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(methods)
    axes[1, 0].legend()
    
    # Plot 4: Intersection sizes
    intersection_counts = [result['intersection_count'] for result in comparison_results]
    
    axes[1, 1].bar(methods, intersection_counts, color=['gold', 'orange', 'darkorange'])
    axes[1, 1].set_title('Number of Common Kinases')
    axes[1, 1].set_ylabel('Number of Common Kinases')
    for i, v in enumerate(intersection_counts):
        axes[1, 1].text(i, v + 0.5, str(v), ha='center')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/kinase_comparison_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create Venn diagrams for each method
    for result in comparison_results:
        method = result['method']
        
        # Get the actual sets for this method
        if method == 'Binomial':
            original_set = set(selected_kinases_orig_binomial)
            complex_set = set(selected_kinases_complex)
        elif method == 'Weighted':
            original_set = set(selected_kinases_orig_weighted)
            complex_set = set(selected_kinases_complex_weighted)
        else:  # Iterative
            original_set = set(selected_kinases_orig_iterative)
            complex_set = set(selected_kinases_complex_iterative)
        
        # Create directory for this method's comparison
        method_dir = f'{output_dir}/{method.lower()}_network_comparison'
        os.makedirs(method_dir, exist_ok=True)
        
        # Create a simple Venn diagram using matplotlib
        plt.figure(figsize=(8, 6))
        venn2([original_set, complex_set], set_labels=('Original Network', 'Complex Network'))
        plt.title(f'{method} Method: Network Comparison')
        plt.savefig(f'{method_dir}/rejection_sets_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Venn diagram for {method} method saved to {method_dir}/rejection_sets_comparison.png")

def save_kinase_comparison_results(comparison_results, output_dir, logger):
    """Save detailed comparison results to files"""
    import pandas as pd
    
    # Save detailed comparison results to CSV
    comparison_df = pd.DataFrame([
        {
            'Method': result['method'],
            'Original_Count': result['original_count'],
            'Complex_Count': result['complex_count'],
            'Intersection_Count': result['intersection_count'],
            'Union_Count': result['union_count'],
            'Jaccard_Index': result['jaccard_index'],
            'Overlap_Coefficient': result['overlap_coefficient']
        }
        for result in comparison_results
    ])
    
    comparison_df.to_csv(f'{output_dir}/kinase_comparison_metrics.csv', index=False)
    logger.info(f"Comparison metrics saved to {output_dir}/kinase_comparison_metrics.csv")
    
    # Create detailed kinase lists for each method
    for result in comparison_results:
        method = result['method'].lower()
        
        # Save intersection (common kinases)
        with open(f'{output_dir}/{method}_common_kinases.txt', 'w') as f:
            f.write(f"Common kinases between original and complex networks ({method} method):\n")
            f.write(', '.join(sorted(result['intersection'])))
        
        # Save original-only kinases
        with open(f'{output_dir}/{method}_original_only_kinases.txt', 'w') as f:
            f.write(f"Kinases only in original network ({method} method):\n")
            f.write(', '.join(sorted(result['original_only'])))
        
        # Save complex-only kinases
        with open(f'{output_dir}/{method}_complex_only_kinases.txt', 'w') as f:
            f.write(f"Kinases only in complex network ({method} method):\n")
            f.write(', '.join(sorted(result['complex_only'])))
    
    logger.info("Detailed kinase lists saved to individual text files")

def plot_top_kinases(results_df, top_kinases):
    '''plot the top kinases'''
    df = results_df[results_df['Name'].isin(top_kinases)]
    # 按 INKA 分数排序
    df = df.sort_values(by='Lower Bound', ascending=False)

    fig, ax = plt.subplots(figsize=(6, 8))
    bars = ax.barh(df['Name'], df['Lower Bound'])

    # for i, (score, p) in enumerate(zip(df['Lower Bound'], df['Upper Bound'])):
    #     ax.text(score + 0.5, i, f"{p:.1e}", va='center', fontsize=9)

    ax.set_xlabel("LIKA Score", fontsize=12)
    # ax.set_title("SK-Mel-28\nINKA", fontsize=14, fontweight='bold', loc='center')
    ax.invert_yaxis()  # highest score on top
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    plt.show()

def plot_top_kinases_streamlit(results_df, top_kinases):
    '''plot the top kinases using streamlit with download functionality'''
    import streamlit as st
    import io
    
    df = results_df[results_df['Name'].isin(top_kinases)]
    # 按 INKA 分数排序
    df = df.sort_values(by='Lower Bound', ascending=False)

    fig, ax = plt.subplots(figsize=(6, 8))
    bars = ax.barh(df['Name'], df['Lower Bound'])

    # for i, (score, p) in enumerate(zip(df['Lower Bound'], df['Upper Bound'])):
    #     ax.text(score + 0.5, i, f"{p:.1e}", va='center', fontsize=9)

    ax.set_xlabel("LIKA Score", fontsize=12)
    ax.set_title("Top Kinases by LIKA Score", fontsize=14, fontweight='bold')
    ax.invert_yaxis()  # highest score on top
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    
    # Display the plot
    st.pyplot(fig)
    
    # Create download button for the figure
    img_buffer = io.BytesIO()
    fig.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
    img_buffer.seek(0)
    
    st.download_button(
        label="📥 Download Top Kinases Plot",
        data=img_buffer.getvalue(),
        file_name="top_kinases_plot.png",
        mime="image/png"
    )
    
    # Close the figure to free memory
    plt.close(fig)

def save_network_to_graphml_streamlit(network, filename="network.graphml"):
    """
    Create a download button for the network GraphML file in streamlit.
    
    Args:
        network: The network object (my_network instance)
        filename: Default filename for download
    
    Returns:
        None (creates streamlit download button)
    """
    import streamlit as st
    import io
    import tempfile
    import os
    
    # Create a copy of the graph to avoid modifying the original
    graph_copy = network.copy()
    
    # Convert non-serializable attributes to strings
    for node in graph_copy.nodes():
        for attr, value in graph_copy.nodes[node].items():
            if isinstance(value, (list, set, tuple)):
                # Convert lists, sets, tuples to comma-separated strings
                graph_copy.nodes[node][attr] = ','.join(map(str, value))
            elif not isinstance(value, (str, int, float, bool)):
                # Convert other non-serializable types to string
                graph_copy.nodes[node][attr] = str(value)
    
    # Create a temporary file to save the GraphML
    with tempfile.NamedTemporaryFile(mode='w', suffix='.graphml', delete=False) as temp_file:
        try:
            # Write the graph to GraphML format
            nx.write_graphml(graph_copy, temp_file.name)
            
            # Read the file content
            with open(temp_file.name, 'rb') as f:
                graphml_data = f.read()
            
            # Create download button
            st.download_button(
                label="🔗 Download Network (GraphML)",
                data=graphml_data,
                file_name=filename,
                mime="application/xml",
                help="Download the network as a GraphML file for visualization in other tools like Cytoscape or Gephi"
            )
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_file.name):
                os.unlink(temp_file.name)

def visualize_rejection_streamlit(network, rejection_set, experiment_name="Analysis", importance_nodes=None):
    '''visualize the rejection set using streamlit'''
    import streamlit as st
    import io
    
    st.subheader("🎯 Rejection Set Visualization")
    st.write(f"Visualizing {len(rejection_set)} rejected nodes out of {network.number_of_nodes()} total nodes")
    
    # Mark rejected nodes
    network_copy = network.copy()
    for node in network_copy.nodes():
        if node in rejection_set:
            network_copy.nodes[node]['rejected'] = 0.00001  # Very low value for rejected
        elif 'p_value' in network_copy.nodes[node]:
            network_copy.nodes[node]['rejected'] = 1  # High value for non-rejected
        else:
            network_copy.nodes[node]['rejected'] = 0.5  # Middle value for missing p-values
    
    # Create the visualization
    fig = visualize_pvalues_streamlit(
        network_copy, 
        experiment_name, 
        attribute='rejected', 
        name='rejection', 
        importance_nodes=importance_nodes,
        return_fig=True
    )
    
    # Display the figure in Streamlit
    st.pyplot(fig)
    
    # Create download button for the figure
    if fig is not None:
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        buf.seek(0)
        
        st.download_button(
            label="📊 Download Rejection Visualization (PNG)",
            data=buf,
            file_name=f"{experiment_name}_rejection_visualization.png",
            mime="image/png",
            help="Download the rejection set visualization as a PNG image"
        )
    
    # Display statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Nodes", network.number_of_nodes())
    with col2:
        st.metric("Rejected Nodes", len(rejection_set))
    with col3:
        rejection_rate = len(rejection_set) / network.number_of_nodes() * 100
        st.metric("Rejection Rate", f"{rejection_rate:.1f}%")
        
def visualize_pvalues_streamlit(G, experiment_name, attribute='p_value', name='p_value', importance_nodes=None, return_fig=False):
    """
    Streamlit version of visualize_pvalues function with download capabilities.
    
    Parameters:
    - G: NetworkX graph (can be directed)
    - experiment_name: Name of the experiment for titles
    - attribute: The node attribute to use for coloring (default: 'p_value')
    - name: Name for the visualization type
    - importance_nodes: Important nodes to highlight
    - return_fig: If True, return the figure instead of displaying it
    """
    import streamlit as st
    import io
    
    # Set the colors
    colors = list(mcolors.TABLEAU_COLORS)  # Using Matplotlib's predefined colors
    color_map = {'Kinase':colors[0], 'Phosphosite':colors[1], 'Protein':colors[2], 'Small-molecule':colors[3]}

    # Get p-values
    p_values = nx.get_node_attributes(G, attribute)
    
    # Convert to undirected graph for community detection
    G_undirected = G.to_undirected()
    
    # Get community partition using Louvain method on undirected graph
    partition = community_louvain.best_partition(G_undirected)
    
    # Create figure with subplots
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Calculate node sizes based on degrees
    node_sizes = [G.degree(node) * 5 + 5 for node in G.nodes()]
    
    node_colors = []
    # take care of the p-value missingness
    for node in G.nodes():
        if node in p_values and p_values[node] is not None:
            if name == 'category':
                color = colors[G.nodes[node]['Description_value']]
            elif name == 'rejection':
                # Use a red-blue colormap for rejection visualization
                if p_values[node] < 0.001:  # Rejected nodes
                    color = 'red'
                elif p_values[node] > 0.999:  # Non-rejected nodes
                    color = 'lightblue'
                else:  # Missing p-values
                    color = '#D8BFD8'
            else:
                color = cm.Blues(1 - p_values[node])
        else:
            color = '#D8BFD8'  # Default color for nodes without p-values
        node_colors.append(color)

    # Get clustered layout
    pos = clustered_layout(G, partition, scale=0.3, seed=42)
    
    if importance_nodes:
        pos = importance_layout(G, importance_nodes)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, alpha=0.3, width=0.5, arrows=True, arrowsize=1, ax=ax)
    
    # Draw nodes (label the importance nodes)
    if importance_nodes:
        labels = {node: str(node) for node in set(importance_nodes) & set(G.nodes())}
    else:
        labels = {}

    # Draw nodes
    nx.draw_networkx_nodes(G, pos,
                        node_size=node_sizes,
                        node_color=node_colors,
                        alpha=0.7,
                        ax=ax)

    # Draw labels of the importance nodes
    if labels:
        nx.draw_networkx_labels(G, pos, labels=labels, ax=ax, 
                               bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=0.5))

    if name == 'category':
        # Create a legend
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                            label=desc, markerfacecolor=color, markersize=10)
                            for desc, color in color_map.items()]
        plt.legend(handles=legend_elements, title="Descriptions")
    elif name == 'rejection':
        # Create legend for rejection visualization
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', label='Rejected', 
                      markerfacecolor='red', markersize=10),
            plt.Line2D([0], [0], marker='o', color='w', label='Not Rejected', 
                      markerfacecolor='lightblue', markersize=10),
            plt.Line2D([0], [0], marker='o', color='w', label='Missing p-value', 
                      markerfacecolor='#D8BFD8', markersize=10)
        ]
        plt.legend(handles=legend_elements, loc='upper left')
    else: 
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=plt.cm.Blues, 
                              norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('p-value')
        cbar.ax.invert_yaxis()  # Invert the colorbar to match the color scheme
        
        # Add legend entry for missing p-values
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                      label='Missing p-values', 
                                      markerfacecolor='#D8BFD8', 
                                      markersize=10)]
        plt.legend(handles=legend_elements, loc='upper left')
    
    ax.set_title(f"Network Visualization: {name.title()} ({experiment_name})")
    ax.axis('off')
    plt.tight_layout()
    
    if return_fig:
        return fig
    
    # Display the plot
    st.pyplot(fig)
    
    # Create download buttons
    col1, col2 = st.columns([1, 1])
    with col1:
        # Save figure to bytes for download
        img_buffer = io.BytesIO()
        fig.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
        img_buffer.seek(0)
        
        st.download_button(
            label="📥 Download Visualization (PNG)",
            data=img_buffer.getvalue(),
            file_name=f"{name}_visualization.png",
            mime="image/png"
        )
    
    with col2:
        # Save as high-resolution PDF
        pdf_buffer = io.BytesIO()
        fig.savefig(pdf_buffer, format='pdf', dpi=300, bbox_inches='tight')
        pdf_buffer.seek(0)
        
        st.download_button(
            label="📄 Download as PDF",
            data=pdf_buffer.getvalue(),
            file_name=f"{name}_visualization.pdf",
            mime="application/pdf"
        )
    
    # Close the figure to free memory
    plt.close(fig)