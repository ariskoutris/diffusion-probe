import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, adjusted_rand_score
from sklearn.decomposition import PCA
from sklearn import cluster as Cluster
from scipy.cluster.hierarchy import dendrogram
from collections import defaultdict

def report_classification(y_true, y_pred, labels, target_names, digits=2):
    """
    Generate and print a classification report.
    """
    report = classification_report(y_true, y_pred, target_names=target_names, labels=labels, digits=digits)
    print(report)
    return report

def compute_adjusted_rand(y_true, y_pred):
    """
    Compute and print the Adjusted Rand Score.
    """
    score = adjusted_rand_score(y_true, y_pred)
    print("Adjusted Rand Score:", score)
    return score

def plot_scatter(x, y, title='Scatter Plot'):
    """
    Plot a 2D scatter plot given a 2D array x and labels y.
    """
    plt.figure(figsize=(5,5))
    cmap = np.array(sns.color_palette("hls", n_colors=int(np.max(y)+1)))
    plt.scatter(x[:,0], x[:,1], c=cmap[y.astype(int)], s=5)
    plt.title(title)
    plt.show()

def plot_classification_results(x_2d, y_true, y_pred, encoder):
    """
    Plot scatter plots for true labels and predicted labels.
    x_2d : 2D projections of the feature space (e.g. from PCA)
    y_true : ground truth labels
    y_pred : predicted labels
    encoder : fitted OrdinalEncoder (to retrieve category names)
    """
    labels = np.unique(y_true)
    target_names = encoder.categories_[0][labels.astype(int)]
    
    # Plot ground truth labels
    fig, ax = plt.subplots(figsize=(5,5))
    cmap = np.array(sns.color_palette('hls', n_colors=encoder.categories_[0].size))
    plt.scatter(x_2d[:,0], x_2d[:,1], c=cmap[y_true.astype(int)], s=5)
    patches = [plt.Line2D([0], [0], marker='o', color='w', label=l,
                 markerfacecolor=c, markersize=6) for c, l in zip(cmap[labels.astype(int)], target_names)]
    plt.legend(title='Breed', handles=patches, loc='best', fontsize=6)
    plt.title("Ground Truth")
    plt.show()

    # Plot predicted labels
    fig, ax = plt.subplots(figsize=(5,5))
    plt.scatter(x_2d[:,0], x_2d[:,1], c=cmap[y_pred.astype(int)], s=5)
    patches = [plt.Line2D([0], [0], marker='o', color='w', label=l,
                 markerfacecolor=c, markersize=6) for c, l in zip(cmap, encoder.categories_[0])]
    plt.legend(title='Breed', handles=patches, loc='best', fontsize=6)
    plt.title("Predictions")
    plt.show()

def add_dicts(d1, d2):
    """Add values of two dictionaries where keys match."""
    out = d1.copy()
    for key in d2.keys():
        if key in d1:
            out[key] += d2[key]
        else:
            out[key] = d2[key]
    return out

def get_linkage_matrix_with_freqs(model):
    """Create linkage matrix with frequencies for dendrogram visualization."""
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    freqs = [{} for _ in range(model.children_.shape[0])]
    
    for i, merge in enumerate(model.children_):
        current_count = 0
        current_freqs = defaultdict(lambda: 0)
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
                current_freqs[model.labels_[child_idx]] += 1
            else:
                current_count += counts[child_idx - n_samples]
                current_freqs = add_dicts(freqs[child_idx - n_samples], current_freqs)
        counts[i] = current_count
        freqs[i] = dict(current_freqs)

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    return linkage_matrix, freqs

def plot_hierarchical_dendrogram(model, df, label_col, hier_df=None, category_depth=None, 
                               figsize=(5,7), orientation='left'):
    """
    Plot a dendrogram with cluster labels based on most common class in each cluster.
    For depth 2 labels, it will show parent/child format when hier_df is provided.
    """
    mode = 'lastp'
    p = model.n_clusters

    linkage_matrix, node_freqs = get_linkage_matrix_with_freqs(model)
    R = dendrogram(linkage_matrix, truncate_mode=mode, p=p, no_plot=True, 
                  show_contracted=True, distance_sort='descending')

    # Get most common label for each cluster
    cluster_most_common_labels = []
    for cluster_id in range(model.n_clusters):
        ids = np.where((model.labels_ == cluster_id))[0]
        cluster = df.iloc[ids]
        
        if category_depth and hier_df is not None:
            # For depth 2 categories, show parent/child format
            mcl = cluster[label_col].mode().to_list()[0]
            parent = hier_df[hier_df[f'cat_depth_{category_depth}'] == mcl][f'cat_depth_{category_depth-1}'].iloc[0]
            cluster_most_common_labels.append(f"{parent}/{mcl}")
        else:
            # Just show the most common label
            cluster_most_common_labels.append(cluster[label_col].mode().to_list()[0])

    # Map leaf indices to labels
    dendrogram_cluster_ids = [list(node_freqs[l-model.n_leaves_].keys())[0] for l in R['leaves']]
    temp = {R["leaves"][ii]: cluster_most_common_labels[dendrogram_cluster_ids[ii]] 
            for ii in range(model.n_clusters)}
    
    def llf(xx):
        return temp[xx]

    fig, ax = plt.subplots(figsize=figsize)
    dendrogram(
        linkage_matrix,
        truncate_mode=mode,
        p=p,
        leaf_label_func=llf,
        leaf_rotation=0,
        leaf_font_size=8,
        orientation=orientation,
        distance_sort='descending',
        ax=ax)
    plt.show()

def run_clustering_grid_search(x, y, n_components_list, n_clusters_range, algorithm="kmeans", 
                              sample_weights=None, linkage=None, metric=None):
    """
    Run a grid search for clustering parameters.
    Returns list of tuples (score, params)
    """
    scores = []
    
    for n_components in n_components_list:
        print(f"Components: {n_components}")
        
        # Apply dimensionality reduction if needed
        if n_components == -1:
            x_reduced = x
        else:
            pca = PCA(n_components=n_components)
            x_reduced = pca.fit_transform(x)
        
        # Try different cluster counts
        for n_clusters in n_clusters_range:
            if algorithm == "kmeans":
                for algo in ["lloyd", "elkan"]:
                    model = Cluster.KMeans(n_clusters=n_clusters, n_init="auto", 
                                        tol=1e-4, max_iter=300, algorithm=algo)
                    preds = model.fit_predict(x_reduced, sample_weight=sample_weights)
                    score = compute_adjusted_rand(y, preds)
                    scores.append((score, {
                        'n_clusters': n_clusters, 
                        'algo': algo, 
                        'n_components': n_components
                    }))
                    
            elif algorithm == "agglomerative":
                for l in linkage or ['average', 'complete', 'ward']:
                    for m in metric or ["euclidean", "l1", "cosine"]:
                        if l == 'ward' and m != 'euclidean':
                            continue
                        model = Cluster.AgglomerativeClustering(
                            n_clusters=n_clusters, linkage=l, metric=m
                        )
                        preds = model.fit_predict(x_reduced)
                        score = compute_adjusted_rand(y, preds)
                        scores.append((score, {
                            'n_clusters': n_clusters, 
                            'linkage': l, 
                            'metric': m, 
                            'n_components': n_components
                        }))
    
    return sorted(scores, key=lambda x: -x[0])
