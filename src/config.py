from dataclasses import dataclass

@dataclass
class Config:
	# Dataset settings
	DATA_DIR:  str = "../dataset/"  # Path where the dataset will be downloaded/stored
	NUM_CLASSES: int = 5  # Number of classes to work on
	IMAGES_PER_CLASS: int = 200  # amount of images for each class
	DATA_URL: str = "https://data.caltech.edu/records/mzrjq-6wc02/files/caltech-101.zip"
	BATCH_SIZE: int = 32  # Number of images processed at once
	SELECTED_CLASSES = [
		"Faces_easy",
		"Motorbikes",
		"airplanes",
		"Leopards",
		"watch"
	]

	# Model settings
	MODEL_NAME: str = "efficientnet_b0"
	IMAGE_SIZE: int = 224  # Expected input resolution for EfficientNet-B0
	NC: int = 3  # Number of input channels (3 = RGB Color)
	ALWAYS_EXTRACT: bool = False  # If True, always run CNN, if false load extraction data from disc.

	# Paths
	OUT_DIR: str = "../output"  # Directory where results will be saved
	RESULTS_FILE = "extractedData.pkl"  # Filename for the extracted list of dictionaries containing features and labels
 
	# PCA scatter plot settings
	SCATTER_PLOT_DPI: int = 180  	# DPI for the saved scatter plot image
	DISPLAY_PCA_PLOT: bool = True  	# Whether to display the scatter plot after exectution
	HOVER_PCA_PLOT: bool = True	# Whether to show image thumbnails on hover in the PCA plot
	THUMBNAIL_SIZE: int = 64	# Size of the stored hover thumbnails