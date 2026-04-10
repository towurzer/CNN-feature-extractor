import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix

from plot_hover import enable_thumbnail_hover
from pca_scatterplot import pca_2d


def _run_kmeans(features: np.ndarray, n_clusters: int) -> np.ndarray:
    """Fits K-Means on the feature vectors and returns the cluster label for each image."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(features)    # Assign each image to a cluster (0 to n_clusters-1)
    return cluster_labels


def _match_clusters_to_classes(cluster_labels: np.ndarray, true_labels: np.ndarray, n_clusters: int) -> np.ndarray:
    # Build cost matrix: rows = true classes, cols = clusters
    cost_matrix = confusion_matrix(true_labels, cluster_labels, labels=list(range(n_clusters)))

    # Hungarian algorithm finds the assignment that maximizes the diagonal (best match)
    row_ind, col_ind = linear_sum_assignment(-cost_matrix)  # Negative = maximize

    # Build remapping: cluster_id -> matched class_id
    remap = np.zeros(n_clusters, dtype=int)
    for class_id, cluster_id in zip(row_ind, col_ind):
        remap[cluster_id] = class_id

    return remap[cluster_labels]  # Apply remapping to all labels


def run_clustering_scatter(extraction_results: list[dict], config) -> str:

    if not extraction_results:
        raise ValueError("extraction_results is empty. Run feature extraction first.")

    features = np.asarray([entry["features"] for entry in extraction_results], dtype=np.float32)  # shape: (N, 1280)
    labels   = np.asarray([entry["label"]    for entry in extraction_results], dtype=np.int32)    # ground-truth labels (0-4)
    thumbnails = [entry.get("thumbnail") for entry in extraction_results]

    n_clusters = config.NUM_CLASSES                         # Number of clusters = number of classes (5)
    cluster_labels = _run_kmeans(features, n_clusters)      # Run K-Means on the raw 1280-D feature vectors
    points_2d, explained_ratio = pca_2d(features)          # Reduce to 2D only for plotting

    # --- Remap cluster IDs to best matching class IDs for consistent coloring ---
    remapped_labels = _match_clusters_to_classes(cluster_labels, labels, n_clusters)

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(10, 7))
    cmap = plt.cm.get_cmap("tab10", n_clusters)             # Same colormap as PCA scatter

    for label_id in range(n_clusters):
        mask = remapped_labels == label_id                  # Select points assigned to this class
        class_name = config.SELECTED_CLASSES[label_id]
        ax.scatter(
            points_2d[mask, 0],                             # x coordinates (PC1)
            points_2d[mask, 1],                             # y coordinates (PC2)
            s=22,
            alpha=0.75,
            color=cmap(label_id),                           # Same color as in PCA scatter
            label=class_name,                               # Show class name instead of cluster number
        )

    enable_thumbnail_hover(fig, ax, points_2d, thumbnails, config)

    ratio_pc1 = explained_ratio[0] * 100.0
    ratio_pc2 = explained_ratio[1] * 100.0
    ax.set_xlabel(f"PC1 ({ratio_pc1:.1f}% explained variance)")
    ax.set_ylabel(f"PC2 ({ratio_pc2:.1f}% explained variance)")
    ax.set_title("K-Means Clustering of CNN Features, visualized with PCA and colored by K-means Clustering")
    ax.legend(loc="best", frameon=True)
    fig.tight_layout()

    output_path = os.path.join(config.OUT_DIR, "clustering_scatter.png")
    fig.savefig(output_path, dpi=config.PLOT_DPI)

    if config.DISPLAY_PLOT:
        plt.show()
    else:
        plt.close(fig)

    return output_path


def run_cluster_inspection(extraction_results: list[dict], config) -> str:

    if not extraction_results:
        raise ValueError("extraction_results is empty. Run feature extraction first.")

    features    = np.asarray([entry["features"] for entry in extraction_results], dtype=np.float32)
    true_labels = np.asarray([entry["label"]    for entry in extraction_results], dtype=np.int32)
    thumbnails = [entry.get("thumbnail") for entry in extraction_results]

    n_clusters     = config.NUM_CLASSES
    cluster_labels = _run_kmeans(features, n_clusters)
    remapped_labels = _match_clusters_to_classes(cluster_labels, true_labels, n_clusters)
    points_2d, explained_ratio = pca_2d(features)

    # Boolean mask: True = correctly assigned, False = wrong
    correct_mask = remapped_labels == true_labels

    fig, ax = plt.subplots(figsize=(10, 7))

    # Plot correct points (green)
    ax.scatter(
        points_2d[correct_mask, 0],
        points_2d[correct_mask, 1],
        s=22, alpha=0.75, color="green",
        label=f"Correct ({correct_mask.sum()})",
    )

    # Plot incorrect points (red)
    ax.scatter(
        points_2d[~correct_mask, 0],
        points_2d[~correct_mask, 1],
        s=22, alpha=0.75, color="red",
        label=f"Incorrect ({(~correct_mask).sum()})",
    )

    enable_thumbnail_hover(fig, ax, points_2d, thumbnails, config)

    ratio_pc1 = explained_ratio[0] * 100.0
    ratio_pc2 = explained_ratio[1] * 100.0
    ax.set_xlabel(f"PC1 ({ratio_pc1:.1f}% explained variance)")
    ax.set_ylabel(f"PC2 ({ratio_pc2:.1f}% explained variance)")
    ax.set_title("K-Means Clustering: Correct vs. Incorrect, visualized with PCA")
    ax.legend(loc="best", frameon=True)
    fig.tight_layout()

    output_path = os.path.join(config.OUT_DIR, "clustering_correct_incorrect.png")
    fig.savefig(output_path, dpi=config.PLOT_DPI)

    if config.DISPLAY_PLOT:
        plt.show()
    else:
        plt.close(fig)

    total_correct = int((confusion_matrix(true_labels, remapped_labels)).trace())
    total = len(true_labels)

    return output_path, total_correct, total