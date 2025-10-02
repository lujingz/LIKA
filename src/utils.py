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
from networkx.drawing import nx_agraph
import logging
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
    
    def get_unmissing_children(self, nodes):
        '''get the unmissing children of the nodes'''
        if isinstance(nodes, str):
            nodes = [nodes]
        return [c for node in nodes for c in self.nodes[node]['unmissing_children']]
                       
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

def plot_top_kinases(results_df, top_kinases):
    '''plot the top kinases'''
    df = results_df[results_df['Name'].isin(top_kinases)]
    df = df.sort_values(by='Lower Bound', ascending=False)

    fig, ax = plt.subplots(figsize=(6, 8))
    bars = ax.barh(df['Name'], df['Lower Bound'])

    ax.set_xlabel("LIKA Score", fontsize=12)
    ax.invert_yaxis()  # highest score on top
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    plt.show()