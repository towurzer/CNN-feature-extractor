import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix


def _pca_2d(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Reduces features to 2D using sklearn PCA and returns points + explained variance ratio."""
    pca = PCA(n_components=2)
    points_2d = pca.fit_transform(features)          # Reduce to 2D
    explained_ratio = pca.explained_variance_ratio_  # Get variance ratio for PC1 and PC2
    return points_2d, explained_ratio


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

    n_clusters = config.NUM_CLASSES                         # Number of clusters = number of classes (5)
    cluster_labels = _run_kmeans(features, n_clusters)      # Run K-Means on the raw 1280-D feature vectors
    points_2d, explained_ratio = _pca_2d(features)          # Reduce to 2D only for plotting

    # --- Remap cluster IDs to best matching class IDs for consistent coloring ---
    remapped_labels = _match_clusters_to_classes(cluster_labels, labels, n_clusters)

    # --- Plot ---
    plt.figure(figsize=(10, 7))
    cmap = plt.cm.get_cmap("tab10", n_clusters)             # Same colormap as PCA scatter

    for label_id in range(n_clusters):
        mask = remapped_labels == label_id                  # Select points assigned to this class
        class_name = config.SELECTED_CLASSES[label_id]
        plt.scatter(
            points_2d[mask, 0],                             # x coordinates (PC1)
            points_2d[mask, 1],                             # y coordinates (PC2)
            s=22,
            alpha=0.75,
            color=cmap(label_id),                           # Same color as in PCA scatter
            label=class_name,                               # Show class name instead of cluster number
        )

    ratio_pc1 = explained_ratio[0] * 100.0
    ratio_pc2 = explained_ratio[1] * 100.0
    plt.xlabel(f"PC1 ({ratio_pc1:.1f}% explained variance)")
    plt.ylabel(f"PC2 ({ratio_pc2:.1f}% explained variance)")
    plt.title("K-Means Clustering of CNN Features")
    plt.legend(loc="best", frameon=True)
    plt.tight_layout()

    output_path = os.path.join(config.OUT_DIR, "clustering_scatter.png")
    plt.savefig(output_path, dpi=config.SCATTER_PLOT_DPI)

    if config.DISPLAY_PCA_PLOT:
        plt.show()
    else:
        plt.close()

    print(f"Clustering scatter plot saved to: {output_path}")
    return output_path


def run_cluster_inspection(extraction_results: list[dict], config) -> str:

    if not extraction_results:
        raise ValueError("extraction_results is empty. Run feature extraction first.")

    features    = np.asarray([entry["features"] for entry in extraction_results], dtype=np.float32)
    true_labels = np.asarray([entry["label"]    for entry in extraction_results], dtype=np.int32)

    n_clusters     = config.NUM_CLASSES
    cluster_labels = _run_kmeans(features, n_clusters)
    remapped_labels = _match_clusters_to_classes(cluster_labels, true_labels, n_clusters)
    points_2d, explained_ratio = _pca_2d(features)

    # Boolean mask: True = correctly assigned, False = wrong
    correct_mask = remapped_labels == true_labels

    plt.figure(figsize=(10, 7))

    # Plot correct points (green)
    plt.scatter(
        points_2d[correct_mask, 0],
        points_2d[correct_mask, 1],
        s=22, alpha=0.75, color="green",
        label=f"Correct ({correct_mask.sum()})",
    )

    # Plot incorrect points (red)
    plt.scatter(
        points_2d[~correct_mask, 0],
        points_2d[~correct_mask, 1],
        s=22, alpha=0.75, color="red",
        label=f"Incorrect ({(~correct_mask).sum()})",
    )

    ratio_pc1 = explained_ratio[0] * 100.0
    ratio_pc2 = explained_ratio[1] * 100.0
    plt.xlabel(f"PC1 ({ratio_pc1:.1f}% explained variance)")
    plt.ylabel(f"PC2 ({ratio_pc2:.1f}% explained variance)")
    plt.title("K-Means Clustering: Correct vs. Incorrect ")
    plt.legend(loc="best", frameon=True)
    plt.tight_layout()

    output_path = os.path.join(config.OUT_DIR, "clustering_correct_incorrect.png")
    plt.savefig(output_path, dpi=config.SCATTER_PLOT_DPI)

    if config.DISPLAY_PCA_PLOT:
        plt.show()
    else:
        plt.close()

    print(f"Correct/incorrect scatter plot saved to: {output_path}")

    total_correct = int((confusion_matrix(true_labels, remapped_labels)).trace())
    total = len(true_labels)
    print(f"\nTotal correct: {total_correct}/{total}  ({100 * total_correct / total:.1f}%)")
    print(f"Total incorrect: {total - total_correct}/{total}  ({100 * (total - total_correct) / total:.1f}%)")

    return output_path