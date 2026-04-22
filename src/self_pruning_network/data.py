from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import datasets, transforms


def _apply_subset(dataset: Dataset, subset_size: int | None) -> Dataset:
    if subset_size is None or subset_size >= len(dataset):
        return dataset
    return Subset(dataset, list(range(subset_size)))


class TransformedDataset(Dataset):
    def __init__(self, base_dataset: Dataset, transform) -> None:
        self.base_dataset = base_dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int):
        image, target = self.base_dataset[index]
        if isinstance(image, torch.Tensor):
            image = transforms.ToPILImage()(image)
        elif not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        return self.transform(image), target


def build_cifar10_loaders(
    data_dir: str | Path,
    batch_size: int,
    num_workers: int = 0,
    validation_ratio: float = 0.1,
    train_subset: int | None = None,
    test_subset: int | None = None,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )

    train_full = datasets.CIFAR10(root=str(data_dir), train=True, download=True, transform=None)
    test_dataset = datasets.CIFAR10(root=str(data_dir), train=False, download=True, transform=eval_transform)

    train_full = _apply_subset(train_full, train_subset)
    test_dataset = _apply_subset(test_dataset, test_subset)

    validation_size = max(1, int(len(train_full) * validation_ratio))
    train_size = len(train_full) - validation_size
    generator = torch.Generator().manual_seed(seed)
    train_dataset, validation_dataset = random_split(train_full, [train_size, validation_size], generator=generator)
    train_dataset = TransformedDataset(train_dataset, train_transform)
    validation_dataset = TransformedDataset(validation_dataset, eval_transform)

    common = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **common)
    validation_loader = DataLoader(validation_dataset, shuffle=False, **common)
    test_loader = DataLoader(test_dataset, shuffle=False, **common)
    return train_loader, validation_loader, test_loader
