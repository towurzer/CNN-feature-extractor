import os
import pickle
import matplotlib.pyplot as plt
def create_dir(config):
	"""
	Creates the necessary project directories if they do not already exist.
	:arg config: Configuration object containing directory paths.
	"""
	os.makedirs(config.DATA_DIR, exist_ok=True)
	os.makedirs(config.OUT_DIR, exist_ok=True)


def saveExtractionResults(config, results):
	"""
	Serializes and saves the extraction results to a pickle file.
	:arg config: Configuration object for path construction.
	:arg results: List of dictionaries containing features, labels and predictions.
	"""
	save_path = os.path.join(config.OUT_DIR, config.RESULTS_FILE)

	# Save data in binary mode to preserve NumPy array structures
	with open(save_path, 'wb') as f:
		pickle.dump(results, f)

	print(f"Success! Saved {len(results)} items to {save_path}")


def loadExtractionResults(config) -> list[dict] | None:
	"""
	Loads previously saved feature extraction results from disk.
	:arg config: Configuration object for path construction.
	:return data: The loaded results if the file exists, otherwise None.
	"""
	path = os.path.join(config.OUT_DIR, config.RESULTS_FILE)

	# Check for file existence to prevent FileNotFoundError
	if os.path.exists(path):
		with open(path, 'rb') as f:
			data = pickle.load(f)
		print(f"Results successfully loaded from {path}")
		return data
	print("No previous results found, starting from scratch")
	return None


def gracefulExit():
	"""Wait for all pots to be closed before exiting"""
	while plt.get_fignums():
		plt.pause(0.1)

	exit(0)