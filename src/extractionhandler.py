import torch
import torch.nn.functional as F


def extract_features(dataloader, model, config) -> list[dict]:
    """
    Iterates through a dataloader to extract features and classifications using the pretrained CNN.
    :arg dataloader: DataLoader containing the image dataset.
    :arg model: The CNN model equipped with a forward hook.

    :arg config: Configuration object used for optional thumbnail generation.

    :return finalResults: A list of dictionaries, where each dictionary contains the classification index, ground truth label, feature vector, and optionally a small thumbnail.
    """
    finalResults = []

    for i, (images, labels) in enumerate(dataloader):  # enumerate over batches
        # Forward pass through the model to trigger the hook and get predictions
        predictions, features = model.extract(images)

        # Convert the PyTorch tensor to a NumPy array
        batchFeaturesArray = features.cpu().numpy()

        thumbnails = None
        if config.HOVER_PCA_PLOT:
            # Build compact thumbnails only when hover feature is enabled.
            thumbnails = F.interpolate(
                images,
                size=(config.THUMBNAIL_SIZE, config.THUMBNAIL_SIZE),
                mode="bilinear",
                align_corners=False,
            ).clamp(0.0, 1.0)

        # Loop through the batch to create individual dictionaries
        for j in range(len(labels)):
            item = {
                'classification': int(predictions[j]),
                'label': int(labels[j]),
                'features': batchFeaturesArray[j],
                'thumbnail': thumbnails[j].permute(1, 2, 0).cpu().numpy() if thumbnails is not None else None
            }
            finalResults.append(item)

        print(f"Batch {i} processed...")
    return finalResults


