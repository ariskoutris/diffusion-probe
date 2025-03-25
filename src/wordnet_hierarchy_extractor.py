"""
Functions for extracting hierarchical structures from WordNet.
"""
import os
import networkx as nx
import pandas as pd
from nltk.corpus import wordnet as wn
from src.synset_processor import get_synset_attributes, find_root_synset, calculate_similarity


def create_hyponym_tree(root_synset, max_depth=1, frequency_threshold=0):
    """
    Create a graph representation of a WordNet synset's hyponym hierarchy.
    
    Args:
        root_synset: The root WordNet synset
        max_depth: Maximum depth to traverse (default: 1)
        frequency_threshold: Minimum frequency for including a synset (default: 0)
        
    Returns:
        A NetworkX graph representing the hierarchy
    """
    def create_tree_recursive(synset, graph, depth):
        graph.add_node(synset.name(), **get_synset_attributes(synset))
        if depth < max_depth:
            for hyponym in synset.hyponyms():
                hyponym_freq = hyponym.lemmas()[0].count()
                if hyponym_freq >= frequency_threshold:
                    create_tree_recursive(hyponym, graph, depth+1)
                    graph.add_edge(synset.name(), hyponym.name(), 
                                 similarity=calculate_similarity(synset, hyponym))

    G = nx.Graph()
    G.add_node(root_synset.name(), **get_synset_attributes(root_synset))
    for hyponym in root_synset.hyponyms():
        hyponym_freq = hyponym.lemmas()[0].count()
        if hyponym_freq >= frequency_threshold:
            create_tree_recursive(hyponym, G, 1)
            G.add_edge(root_synset.name(), hyponym.name(), 
                     similarity=calculate_similarity(root_synset, hyponym))
    return G

def extract_hierarchy(word, max_depth=2, frequency_threshold=0, output_file=None):
    """
    Extract a hierarchy from WordNet and optionally save to CSV.
    
    Args:
        word: The root word to start extraction from
        max_depth: Maximum depth to traverse (default: 2)
        frequency_threshold: Minimum frequency threshold (default: 0)
        output_file: Path to save CSV (default: None)
        
    Returns:
        DataFrame containing the hierarchy and the NetworkX graph
    """
    root_synset = find_root_synset(word)
    hypotree = create_hyponym_tree(root_synset, max_depth, frequency_threshold)
    
    # Create a dataframe representation of the hierarchy
    all_paths = nx.single_source_shortest_path(hypotree, root_synset.name())
    
    # Convert paths to a DataFrame with proper depth columns
    max_path_length = max(len(path) for path in all_paths.values())
    columns = [f'cat_depth_{i}' for i in range(max_path_length)]
    
    hier_df = pd.DataFrame([path + [None] * (max_path_length - len(path)) 
                           for path in all_paths.values()], 
                           columns=columns)
    
    # Add synset ID and frequency
    hier_df.insert(0, 'synset_id', all_paths.keys())
    hier_df.insert(1, 'class', all_paths.keys())
    hier_df.insert(len(columns) + 2, 'frequency', 
                  [wn.synset(key).lemmas()[0].count() for key in all_paths.keys()])
    
    # Add definition column for enriching prompts
    hier_df.insert(len(columns) + 3, 'definition',
                  [wn.synset(key).definition() for key in all_paths.keys()])
    
    # Format node names
    format_node_name = lambda x: x.split('.')[0].replace("_", " ").title() if x is not None else None
    
    # Apply the formatting function to each column individually
    for col in hier_df.iloc[:, 1:len(columns) + 2].columns:
        hier_df[col] = hier_df[col].map(format_node_name)
    
    # Save to file if requested
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        hier_df.to_csv(output_file, index=False)
    
    return hier_df, hypotree

def get_hierarchy_paths(hierarchy_graph, root_node):
    """Get all paths from root to each node in the hierarchy."""
    return nx.single_source_shortest_path(hierarchy_graph, root_node)