import os
import shutil
import tarfile
import urllib
import zipfile

from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def download_and_extract_caltech101(config):
    """
    Automatically downloads and extracts the Caltech101 dataset
    from the official university repository if it doesn't already exist.
    """
    base_dir = os.path.join(config.DATA_DIR, "caltech101")
    target_dir = os.path.join(base_dir, "101_ObjectCategories")

    # If the folder already exists, we skip downloading
    if os.path.exists(target_dir):
        print("Dataset found, Downloading skipped, Creating Dataloader")
        return

    os.makedirs(base_dir, exist_ok=True)
    zip_path = os.path.join(base_dir, "caltech-101.zip")

    print("Downloading Caltech101 (this may take a few minutes)...")
    urllib.request.urlretrieve(config.DATA_URL, zip_path)

    print("Extracting downloaded zip file...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(base_dir)

    # The zip contains a tar.gz file with the actual images, we need to extract that too
    tar_path = os.path.join(base_dir, "caltech-101", "101_ObjectCategories.tar.gz")

    print("Extracting images from tarball...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(base_dir)

    print("Cleaning up temporary download files...")
    os.remove(zip_path)
    shutil.rmtree(os.path.join(base_dir, "caltech-101"))  # Remove the intermediate folder

    print("Download complete! Creating Dataloader")


def get_dataloader(config):
    # Preprocessing
    size = config.IMAGE_SIZE
    preprocess = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.ToTensor()
    ])

    download_and_extract_caltech101(config)

    # Load the Caltech101 dataset
    full_dataset = datasets.Caltech101(
        root=config.DATA_DIR,
        download=False,
        transform=preprocess,
        target_type="category"
    )

    # Create a mapping of the 5 selected classes to their original dataset indices
    category_name_to_original_index = {
        category_name: original_index
        for original_index, category_name in enumerate(full_dataset.categories)
    }

    # remap the original indices to new contiguous labels.
    valid_indices_map = {
        category_name_to_original_index[selected_category]: new_label
        for new_label, selected_category in enumerate(config.SELECTED_CLASSES)
    }

    # Iterate through the dataset, keeping only the selected classes and remapping their labels
    filtered_samples = []
    images_collected_per_class = {new_label: 0 for new_label in valid_indices_map.values()}

    for img, label in full_dataset:
        if label in valid_indices_map:
            new_label = valid_indices_map[label]

            # If we haven't reached the limit for this class yet, add it
            if images_collected_per_class[new_label] < config.IMAGES_PER_CLASS:
                filtered_samples.append((img, new_label))
                images_collected_per_class[new_label] += 1


    # Wrap our list of tuples into a PyTorch DataLoader
    loader = DataLoader(
        filtered_samples,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    print(f"Dataset ready: {len(filtered_samples)} images (approx.{len(filtered_samples) // config.NUM_CLASSES} per class).")
    return loader