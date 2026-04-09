import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA

from plot_hover import enable_thumbnail_hover


def pca_2d(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
	"""Reduce features to 2D with PCA and return projected points plus explained variance ratios."""
	pca = PCA(n_components=2)
	points_2d = pca.fit_transform(features)
	explained_ratio = pca.explained_variance_ratio_
	return points_2d, explained_ratio


def run_pca_scatter(extraction_results: list[dict], config) -> str:
	"""
	Creates a 2D PCA scatter plot from extracted CNN features and returns the saved image path.
	"""
	if not extraction_results:
		raise ValueError("extraction_results is empty. Run feature extraction first.")

	features = np.asarray([entry["features"] for entry in extraction_results], dtype=np.float32)
	labels = np.asarray([entry["label"] for entry in extraction_results], dtype=np.int32)
	thumbnails = [entry.get("thumbnail") for entry in extraction_results]

	points_2d, explained_ratio = pca_2d(features)

	fig, ax = plt.subplots(figsize=(10, 7))
	cmap = plt.cm.get_cmap("tab10", max(len(config.SELECTED_CLASSES), int(np.max(labels)) + 1))

	unique_labels = np.unique(labels)
	legend_handles = []
	for label in unique_labels:
		class_name = config.SELECTED_CLASSES[int(label)] if int(label) < len(config.SELECTED_CLASSES) else str(int(label))
		legend_handles.append(
			Line2D(
				[0],
				[0],
				marker="o",
				color="w",
				label=class_name,
				markerfacecolor=cmap(int(label)),
				markersize=8,
			)
		)
		mask = labels == label
		ax.scatter(
			points_2d[mask, 0],
			points_2d[mask, 1],
			s=22,
			alpha=0.75,
			color=cmap(int(label)),
		)

	enable_thumbnail_hover(fig, ax, points_2d, thumbnails, config)

	ratio_pc1 = explained_ratio[0] * 100.0
	ratio_pc2 = explained_ratio[1] * 100.0
	ax.set_xlabel(f"PC1 ({ratio_pc1:.1f}% explained variance)")
	ax.set_ylabel(f"PC2 ({ratio_pc2:.1f}% explained variance)")
	ax.set_title("PCA Scatterplot of CNN Features")
	ax.legend(handles=legend_handles, loc="best", frameon=True)
	fig.tight_layout()

	output_path = os.path.join(config.OUT_DIR, "pca_scatter.png")
	fig.savefig(output_path, dpi=config.PLOT_DPI)

	if config.DISPLAY_PLOT:
		plt.show()
	else:
		plt.close(fig)

	return output_path
