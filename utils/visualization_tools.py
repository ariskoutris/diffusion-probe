"""
Tools for visualizing WordNet hierarchies and graph structures.
"""
import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout

def visualize_hierarchy(graph, layout="dot", figsize=(10, 10), node_size=0, 
                       edge_width=0.5, font_size=8, show_bbox=True):
    """
    Visualize a WordNet hierarchy using different layout algorithms.
    
    Args:
        graph: NetworkX graph representing the hierarchy
        layout: Layout algorithm ('dot', 'twopi', 'kamada_kawai', 'spring')
        figsize: Figure size as tuple (width, height)
        node_size: Size of nodes
        edge_width: Width of edges
        font_size: Font size for labels
        show_bbox: Whether to show background box around labels
    """
    custom_labels = {k: v['name'] for k, v in dict(graph.nodes).items()}
    
    options = {
        "node_size": node_size, 
        "alpha": 0.2, 
        "width": edge_width, 
        "node_color": 'blue',
        "arrows": False
    }
    
    fig, ax = plt.subplots(figsize=figsize)
    
    if layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(graph)
    elif layout == "spring":
        pos = nx.spring_layout(graph, k=3, iterations=1000, threshold=1e-7)
    elif layout in ["dot", "twopi", "neato", "circo", "fdp"]:
        if layout == "dot":
            args = '-Grankdir="LR"' if layout == "dot" else ''
            pos = graphviz_layout(graph, prog=layout, args=args)
        else:
            pos = graphviz_layout(graph, prog=layout)
    else:
        raise ValueError(f"Unsupported layout: {layout}")
    
    nx.draw(graph, pos, **options, ax=ax)
    
    bbox_opts = dict(facecolor="skyblue") if show_bbox else None
    nx.draw_networkx_labels(
        graph, 
        pos, 
        labels=custom_labels, 
        font_size=font_size, 
        font_weight="bold", 
        bbox=bbox_opts,
        ax=ax
    )
    
    plt.tight_layout()
    return fig, ax
