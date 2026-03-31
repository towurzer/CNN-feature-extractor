from dataclasses import dataclass


@dataclass
class Config:
	# System
	# device: str = "cuda" if torch.cuda.is_available() else "cpu"
	# num_workers: int = 4 if torch.cuda.is_available() else 0

	# Hyperparameters
	image_size: int = 244
	nc: int = 3  # Number of channels (RGB vs Grayscale)

	# Data
	data_root: str = "./dataset"
	# CIFAR-10 labels: 0=airplane, 1=car, 2=bird, 3=cat, 4=deer, 5=dog, 6=frog, 7=horse, 8=ship, 9=truck
	target_class: tuple[int] = 0

	# Paths & Logging
	save_dir: str = "./logs"
	model_save_path: str = "./model"

