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