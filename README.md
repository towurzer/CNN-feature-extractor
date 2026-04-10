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
    config.py               # Settings
    dataset.py              # Data Pipeline: Downloads and filters the Dataset
    main.py                 # Manages the whole pipeline 
    extractionhandler.py    # Batch processing logic
    efficientNet.py         # The Model, edited to be a feature extractor
    scatterplot.py          # PCA and 2D visualization
    clustering.py           # K-means feature analysis
    utils.py                # utility functions
---
dataset/                    # Image source files
output/                     # Saved plots and extracted features
```

## Getting Started
### 1. Installation

Run
```bash
pip install -r requirements.txt
```
to install neccessary requirements.

### 2. Run the extraction pipeline

To reproduce the results, run the main extraction pipeline
```bash
python src/main.py
```
+ This will Load the current Configuration
+ automatically download the dataset
+ download the model including its pretrained weigths
+ start the extraction process
+ reduce the resulting feature vectors plot them
+ perform the K-means clustering and plot them
+ compare the feature vector and K-means plot and compare them for differences, shown in another plot
+ automatically open all the plots

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