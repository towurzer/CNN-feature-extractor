import numpy as np
from matplotlib.offsetbox import AnnotationBbox, OffsetImage


def enable_thumbnail_hover(fig, ax, points_2d: np.ndarray, thumbnails: list, config) -> None:
	"""Attach thumbnail hover previews to a scatter plot."""
	if not getattr(config, "HOVER_PLOT", True):
		return

	valid_thumbnails = [img for img in thumbnails if isinstance(img, np.ndarray)]
	if not valid_thumbnails:
		return

	hover_scatter = ax.scatter(
		points_2d[:, 0],
		points_2d[:, 1],
		s=28,
		alpha=0.0,
		picker=True,
	)

	imagebox = OffsetImage(
		np.zeros((config.THUMBNAIL_SIZE, config.THUMBNAIL_SIZE, 3), dtype=np.float32),
		zoom=1.25,
	)
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