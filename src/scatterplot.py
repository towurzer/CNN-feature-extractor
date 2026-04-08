import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from sklearn.decomposition import PCA


def _pca_2d(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
	"""Reduces features to 2D using sklearn PCA and returns points + explained variance ratio."""
	pca = PCA(n_components=2)
	points_2d = pca.fit_transform(features)             # Reduce to 2D
	explained_ratio = pca.explained_variance_ratio_     # Get variance ratio for PC1 and PC2        
	return points_2d, explained_ratio


def run_pca_scatter(extraction_results: list[dict], config) -> str:
	"""
	Creates a 2D PCA scatter plot from extracted CNN features and returns the saved image path.
	"""
	if not extraction_results:
		raise ValueError("extraction_results is empty. Run feature extraction first.")

	features = np.asarray([entry["features"] for entry in extraction_results], dtype=np.float32)    # get features(=output of forward hook) as numpy array
	labels = np.asarray([entry["label"] for entry in extraction_results], dtype=np.int32)           # get labels(=class indices) as numpy array 
	thumbnails = [entry.get("thumbnail") for entry in extraction_results]

	points_2d, explained_ratio = _pca_2d(features)

	fig, ax = plt.subplots(figsize=(10, 7))
	cmap = plt.cm.get_cmap("tab10", max(len(config.SELECTED_CLASSES), int(np.max(labels)) + 1))     # Get a color map with enough distinct colors for all classes

	unique_labels = np.unique(labels)
	legend_handles = []
	for label in unique_labels:
		class_name = config.SELECTED_CLASSES[int(label)] if int(label) < len(config.SELECTED_CLASSES) else str(int(label))      # account for the possibility that there are more unique labels than selected classes
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
		mask = labels == label      # Create a boolean mask for the current class label i.e. select all points belonging to the current class from all classified points
		ax.scatter(
			points_2d[mask, 0],     # x coordinaates of the points belonging to the current class
			points_2d[mask, 1],     # y coordinaates of the points belonging to the current class
			s=22,
			alpha=0.75,
			color=cmap(int(label)),
		)

	hover_scatter = ax.scatter(
		points_2d[:, 0],
		points_2d[:, 1],
		s=28,
		alpha=0.0,
		picker=True,
	)

	if config.HOVER_PCA_PLOT and any(isinstance(img, np.ndarray) for img in thumbnails):
		imagebox = OffsetImage(np.zeros((config.THUMBNAIL_SIZE, config.THUMBNAIL_SIZE, 3), dtype=np.float32), zoom=1.25)
		annot = AnnotationBbox(
			imagebox,
			(0, 0),
			xybox=(45, 45),
			xycoords="data",
			boxcoords="offset points",
			frameon=True,
			pad=0.4,
			bboxprops={"edgecolor": "black", "linewidth": 0.8},
		)
		annot.set_visible(False)
		ax.add_artist(annot)

		def _on_move(event):
			if event.inaxes != ax:
				if annot.get_visible():
					annot.set_visible(False)
					fig.canvas.draw_idle()
				return

			contains, info = hover_scatter.contains(event)
			if not contains:
				if annot.get_visible():
					annot.set_visible(False)
					fig.canvas.draw_idle()
				return

			idx = int(info["ind"][0])
			image = thumbnails[idx]
			if not isinstance(image, np.ndarray):
				return

			annot.xy = (points_2d[idx, 0], points_2d[idx, 1])
			imagebox.set_data(image)
			annot.set_visible(True)
			fig.canvas.draw_idle()

		fig.canvas.mpl_connect("motion_notify_event", _on_move)

	ratio_pc1 = explained_ratio[0] * 100.0
	ratio_pc2 = explained_ratio[1] * 100.0
	ax.set_xlabel(f"PC1 ({ratio_pc1:.1f}% explained variance)")
	ax.set_ylabel(f"PC2 ({ratio_pc2:.1f}% explained variance)")
	ax.set_title("PCA Scatterplot of CNN Features")
	ax.legend(handles=legend_handles, loc="best", frameon=True)
	fig.tight_layout()

	os.makedirs(config.OUT_DIR, exist_ok=True)
	output_path = os.path.join(config.OUT_DIR, "pca_scatter.png")
	fig.savefig(output_path, dpi=config.SCATTER_PLOT_DPI)

	if config.DISPLAY_PCA_PLOT:
		plt.show()
	else:
		plt.close(fig)

	return output_path
