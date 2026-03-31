# CNN-feature-extractor

The goal of this assignment is to implement a complete pipeline that:
+ loads an image dataset,
+ extracts feature vectors from the images using a pre-trained CNN,
+ visualizes the extracted features in 2D,
+ clusters the features,
+ and analyzes whether the learned feature space reflects the image classes.

## Project Structure
```text
src/
	/dataset				# Auto Generated Folder with the dataset
	config.py				# Model / Training Settings and Hyperparameters
	dataset.py				# Data Pipeline: Downloads and filters the CIFAR-10 Dataset
	main.py					# Manages the whole pipeline, loading, setup, data preparation, and model trainer initialization
	model.py				# Model Architecture
	model_trainer.py		# Manages the training loop, evaluation and logging
	visualizer.py           # visualizes the extracted features in 2D
	utils.py				# Utility functions to help seed the model trainer and more helper functions
model/						# trained models (.pth)
logs/						# logs
```

## Responsibilities
### Giuly:
* Step 1: Choose a Dataset  
* Step 3: Process the Images (image size 224x224)

### Tobi:
* Step 2: Choose a CNN Model 
* Step 4: Extract CNN - Features (implement forward hook)

### Sebi:
* Step 5: Reduce Dimensions & Plot

### Maky: 
* Step 6: Clustering

### Together:
* Finalize Report (2-3 Pages)
* Build presentation (max 10 Slides)