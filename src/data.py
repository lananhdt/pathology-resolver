from pathlib import Path
import random
import csv

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


CLASS_NAMES = [
    "ADI", "BACK", "DEB", "LYM", "MUC",
    "MUS", "NORM", "STR", "TUM"
]


def collect_samples(root, max_samples_per_class=None):
    root = Path(root)
    samples = []

    for label, class_name in enumerate(CLASS_NAMES):
        class_dir = root / class_name
        files = sorted(
            list(class_dir.glob("*.tif")) +
            list(class_dir.glob("*.tiff")) +
            list(class_dir.glob("*.png")) +
            list(class_dir.glob("*.jpg"))
        )

        if max_samples_per_class is not None:
            files = files[:int(max_samples_per_class)]

        samples.extend((str(path), label) for path in files)

    return samples


def split_samples(samples, val_ratio=0.15, seed=42):
    rng = random.Random(seed)
    by_class = {}

    for path, label in samples:
        by_class.setdefault(label, []).append((path, label))

    train_samples, val_samples = [], []

    for label, class_samples in by_class.items():
        rng.shuffle(class_samples)
        n_val = int(len(class_samples) * val_ratio)
        val_samples.extend(class_samples[:n_val])
        train_samples.extend(class_samples[n_val:])

    rng.shuffle(train_samples)
    rng.shuffle(val_samples)

    return train_samples, val_samples


def get_transforms(image_size=224, train=True):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if train:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(
                brightness=0.15,
                contrast=0.15,
                saturation=0.15,
                hue=0.03
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


class HistologyDataset(Dataset):
    def __init__(self, samples, image_size=224, train=True):
        self.samples = samples
        self.transform = get_transforms(image_size, train=train)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        image = Image.open(path).convert("RGB")
        image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long), path
        
import csv

def read_split_csv(path):
    rows = []

    with open(path, "r") as file:
        reader = csv.DictReader(file)

        for row in reader:
            rows.append(
                (
                    row["path"],
                    int(row["label"]),
                )
            )

    return rows
    
def get_stain_transform(
    severity="strong",
):
    if severity == "strong":
        return transforms.Compose([
            transforms.ColorJitter(
                brightness=0.35,
                contrast=0.35,
                saturation=0.35,
                hue=0.08,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225],
            ),
        ])
