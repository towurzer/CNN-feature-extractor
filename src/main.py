from config import Config
from efficientNet import EfficientNet
import utils
import dataset
import extractionhandler
import pca_scatterplot
import clustering

if __name__ == "__main__":
	print("Starting EfficientNet")
	config = Config()  # Load Configuration
	utils.create_dir(config)  # create directories to store dataset as well as the scatter plots and clustering
	print("Loaded config, created directories")

	extractionResults = None

	# Check for cached results to save computation time
	if not config.ALWAYS_EXTRACT:
		print("Try to load extraction results from cache")
		extractionResults = utils.loadExtractionResults(config)

	# If no cache is available or the configuration specifies that we should not use our cache, run the CNN
	if extractionResults is None:
		print("Download Dataset, Create Dataloader")
		dataloader = dataset.get_dataloader(config)  # download the dataset, crate and return the dataloader
		print("Dataloader Created, creating the model, loading pretrained weights, registering the hook")
		model = EfficientNet(train=False)  # download the model, apply weights and register the forward hook
		print("Model Loaded, starting Feature Extraction...")
		extractionResults = extractionhandler.extract_features(dataloader, model, config) # run the classification and feature extraction
		print(f"Extraction Complete! Total images processed: {len(extractionResults)}")
		print("Saving results to cache...")
		utils.saveExtractionResults(config, extractionResults) # Save results for fast future use

	print("Running PCA and creating scatter plot")
	plot_path = pca_scatterplot.run_pca_scatter(extractionResults, config)
	print(f"Saved PCA scatter plot to: {plot_path}")

	print("Running K-Means clustering")
	cluster_path = clustering.run_clustering_scatter(extractionResults, config)
	print(f"Saved clustering scatter plot to: {cluster_path}")

	print("Running inspection")
	inspection_path, total_correct, total = clustering.run_cluster_inspection(extractionResults, config)
	print(f"Saved cluster inspection scatter plot to: {inspection_path}")
	
	# Print results
	print("----------------------------------------------------")
	print("Inspection Results: ")
	print(f"Total correct: {total_correct}/{total}  ({100 * total_correct / total:.1f}%)")
	print(f"Total incorrect: {total - total_correct}/{total}  ({100 * (total - total_correct) / total:.1f}%)")
