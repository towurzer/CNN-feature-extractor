import torch
import torch.nn.functional as F


def extract_features(dataloader, model, thumbnail_size: int = 64) -> list[dict]:
    """
    Iterates through a dataloader to extract features and classifications using the pretrained CNN.
    :arg dataloader: DataLoader containing the image dataset.
    :arg model: The CNN model equipped with a forward hook.

    :return finalResults: A list of dictionaries, where each dictionary contains the classification index, ground truth label, feature vector, and a small thumbnail.
    """
    finalResults = []

    for i, (images, labels) in enumerate(dataloader):  # enumerate over batches
        # Forward pass through the model to trigger the hook and get predictions
        predictions, features = model.extract(images)

        # Convert the PyTorch tensor to a NumPy array
        batchFeaturesArray = features.cpu().numpy()

        # Build compact thumbnails for hover previews in the scatter plot.
        thumbnails = F.interpolate(
            images,
            size=(thumbnail_size, thumbnail_size),
            mode="bilinear",
            align_corners=False,
        ).clamp(0.0, 1.0)

        # Loop through the batch to create individual dictionaries
        for j in range(len(labels)):
            item = {
                'classification': int(predictions[j]),
                'label': int(labels[j]),
                'features': batchFeaturesArray[j],
                'thumbnail': thumbnails[j].permute(1, 2, 0).cpu().numpy()
            }
            finalResults.append(item)

        print(f"Batch {i} processed...")
    return finalResults


