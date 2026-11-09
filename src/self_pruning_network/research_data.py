"""CIFAR-10 loaders for the research protocol.

Differences from the legacy ``data.build_cifar10_loaders`` (which is unchanged):

* the 45k/5k train/validation partition is fixed by ``split_seed`` (default 0) and does
  **not** depend on the training seed;
* images are held as one uint8 tensor and augmented with tensor ops (fast, picklable);
* a fixed calibration subset of the training partition (no augmentation) is exposed
  for data-dependent importance criteria;
* optional per-split limits support CPU smoke tests.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
TRAIN_SIZE, VALIDATION_SIZE = 45_000, 5_000


def _load_batch(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with open(path, "rb") as handle:
        entry = pickle.load(handle, encoding="latin1")
    data = np.asarray(entry["data"], dtype=np.uint8).reshape(-1, 3, 32, 32)
    return data, np.asarray(entry.get("labels", entry.get("fine_labels")), dtype=np.int64)


def load_cifar10_arrays(data_dir: str | Path) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    root = Path(data_dir) / "cifar-10-batches-py"
    if not root.exists():
        from torchvision import datasets
        datasets.CIFAR10(root=str(data_dir), train=True, download=True)
        datasets.CIFAR10(root=str(data_dir), train=False, download=True)
    train = [_load_batch(root / f"data_batch_{i}") for i in range(1, 6)]
    x_train = torch.from_numpy(np.concatenate([b[0] for b in train]))
    y_train = torch.from_numpy(np.concatenate([b[1] for b in train]))
    x_test, y_test = (torch.from_numpy(a) for a in _load_batch(root / "test_batch"))
    return x_train, y_train, x_test, y_test


def fixed_split_indices(n: int = 50_000, split_seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    permutation = torch.randperm(n, generator=torch.Generator().manual_seed(int(split_seed)))
    validation_size = VALIDATION_SIZE if n == 50_000 else max(1, n // 10)
    return permutation[validation_size:].sort().values, permutation[:validation_size].sort().values


class TensorImageDataset(Dataset):
    def __init__(self, images: torch.Tensor, labels: torch.Tensor, augmentation: str = "none") -> None:
        if augmentation not in {"none", "standard", "mlp_legacy"}:
            raise ValueError(f"unknown augmentation {augmentation!r}")
        self.images, self.labels, self.augmentation = images, labels, augmentation
        self.mean = torch.tensor(CIFAR10_MEAN).view(3, 1, 1)
        self.std = torch.tensor(CIFAR10_STD).view(3, 1, 1)
        self._jitter = None
        if augmentation == "mlp_legacy":
            from torchvision import transforms
            self._jitter = transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15)

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int):
        image = self.images[index].float().div_(255.0)
        if self.augmentation != "none":
            padded = torch.nn.functional.pad(image, (4, 4, 4, 4))
            top, left = torch.randint(0, 9, (2,)).tolist()
            image = padded[:, top:top + 32, left:left + 32]
            if torch.rand(()) < 0.5:
                image = image.flip(-1)
            if self._jitter is not None:
                image = self._jitter(image)
        return (image - self.mean) / self.std, self.labels[index]


@dataclass
class Loaders:
    train: DataLoader
    validation: DataLoader
    test: DataLoader
    calibration: DataLoader
    split_seed: int
    sizes: dict[str, int]


def _worker_init(worker_id: int) -> None:
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)


def build_research_loaders(data_dir: str | Path = "data", batch_size: int = 128, num_workers: int = 0,
                           augmentation: str = "standard", split_seed: int = 0, loader_seed: int = 0,
                           calibration_size: int = 2048, eval_batch_size: int | None = None,
                           train_limit: int | None = None, validation_limit: int | None = None,
                           test_limit: int | None = None, arrays=None) -> Loaders:
    x_train, y_train, x_test, y_test = arrays if arrays is not None else load_cifar10_arrays(data_dir)
    train_idx, val_idx = fixed_split_indices(int(x_train.shape[0]), split_seed)
    calib_idx = train_idx[: min(calibration_size, train_idx.numel())]
    if train_limit is not None:
        train_idx = train_idx[:train_limit]
    if validation_limit is not None:
        val_idx = val_idx[:validation_limit]
    test_slice = slice(0, test_limit) if test_limit is not None else slice(None)
    pin = torch.cuda.is_available()
    eval_bs = eval_batch_size or max(batch_size, 256)
    common = {"num_workers": num_workers, "pin_memory": pin, "worker_init_fn": _worker_init,
              "persistent_workers": num_workers > 0}
    train = DataLoader(TensorImageDataset(x_train[train_idx], y_train[train_idx], augmentation),
                       batch_size=batch_size, shuffle=True, drop_last=False,
                       generator=torch.Generator().manual_seed(int(loader_seed)), **common)
    validation = DataLoader(TensorImageDataset(x_train[val_idx], y_train[val_idx]), batch_size=eval_bs,
                            shuffle=False, **common)
    test = DataLoader(TensorImageDataset(x_test[test_slice], y_test[test_slice]), batch_size=eval_bs,
                      shuffle=False, **common)
    calibration = DataLoader(TensorImageDataset(x_train[calib_idx], y_train[calib_idx]), batch_size=min(256, eval_bs),
                             shuffle=False, num_workers=0)
    sizes = {"train": len(train.dataset), "validation": len(validation.dataset), "test": len(test.dataset),
             "calibration": len(calibration.dataset)}
    return Loaders(train, validation, test, calibration, int(split_seed), sizes)


def reseed_loader(loader: DataLoader, seed: int) -> None:
    """Make the next epochs' shuffling order a function of ``seed`` only."""
    generator = getattr(loader, "generator", None)
    if generator is not None:
        generator.manual_seed(int(seed))
