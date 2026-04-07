import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


def _pca_2d(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
	"""Reduces features to 2D using sklearn PCA and returns points + explained variance ratio."""
	pca = PCA(n_components=2)
	points_2d = pca.fit_transform(features)             # Reduce to 2D
	explained_ratio = pca.explained_variance_ratio_     # Get variance ratio for PC1 and PC2        
	return points_2d, explained_ratio


def run_pca_scatter(extraction_results: list[dict], config, show_plot: bool = True) -> str:
	"""
	Creates a 2D PCA scatter plot from extracted CNN features and returns the saved image path.
	"""
	if not extraction_results:
		raise ValueError("extraction_results is empty. Run feature extraction first.")

	features = np.asarray([entry["features"] for entry in extraction_results], dtype=np.float32)    # get features(=output of forward hook) as numpy array
	labels = np.asarray([entry["label"] for entry in extraction_results], dtype=np.int32)           # get labels(=class indices) as numpy array 

	points_2d, explained_ratio = _pca_2d(features)

	plt.figure(figsize=(10, 7))
	cmap = plt.cm.get_cmap("tab10", max(len(config.SELECTED_CLASSES), int(np.max(labels)) + 1))     # Get a color map with enough distinct colors for all classes

	unique_labels = np.unique(labels) 
	for label in unique_labels:
		class_name = config.SELECTED_CLASSES[int(label)] if int(label) < len(config.SELECTED_CLASSES) else str(int(label))      # account for the possibility that there are more unique labels than selected classes
		mask = labels == label      # Create a boolean mask for the current class label i.e. select all points belonging to the current class from all classified points
		plt.scatter(
			points_2d[mask, 0],     # x coordinaates of the points belonging to the current class
			points_2d[mask, 1],     # y coordinaates of the points belonging to the current class
			s=22,
			alpha=0.75,
			color=cmap(int(label)),
			label=class_name,
		)

	ratio_pc1 = explained_ratio[0] * 100.0
	ratio_pc2 = explained_ratio[1] * 100.0
	plt.xlabel(f"PC1 ({ratio_pc1:.1f}% explained variance)")
	plt.ylabel(f"PC2 ({ratio_pc2:.1f}% explained variance)")
	plt.title("PCA Scatterplot of CNN Features")
	plt.legend(loc="best", frameon=True)
	plt.tight_layout()

	os.makedirs(config.OUT_DIR, exist_ok=True)
	output_path = os.path.join(config.OUT_DIR, "pca_scatter.png")
	plt.savefig(output_path, dpi=180)

	if show_plot:
		plt.show()
	else:
		plt.close()

	return output_path
