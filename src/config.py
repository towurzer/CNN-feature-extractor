from dataclasses import dataclass

@dataclass
class Config:
	# Dataset settings
	DATA_DIR:  str = "../dataset/"  # Path where the dataset will be downloaded/stored
	NUM_CLASSES: int = 5  # Number of classes to work on
	IMAGES_PER_CLASS: int = 150  # amount of images for each class
	BATCH_SIZE: int = 32  # Number of images processed at once

	# Model settings
	MODEL_NAME: str = "efficientnet_b0"
	IMAGE_SIZE: int = 224  # Expected input resolution for EfficientNet-B0
	NC: int = 3  # Number of input channels (3 = RGB Color)
	ALWAYS_EXTRACT: bool = False  # If True, always run CNN, if false load extraction data from disc.

	# Paths
	OUT_DIR: str = "../output"  # Directory where results will be saved
	RESULTS_FILE = "extracted_data.pkl"  # Filename for the extracted list of dictionaries containing features and labels