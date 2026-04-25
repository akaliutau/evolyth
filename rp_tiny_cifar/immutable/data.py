from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


@dataclass
class DataBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    dataset_mode: str
    train_size: int
    val_size: int


class SyntheticCifarLike(Dataset):
    """Deterministic CIFAR-shaped fixture for offline smoke tests.

    Labels are generated from simple visual-ish rules over channel means and
    quadrants, so the model can learn above random even when no internet or CIFAR
    files are available. This is not a benchmark substitute; it is a stable demo
    fallback and dry-run dataset.
    """

    def __init__(self, size: int, seed: int, train: bool = True):
        rng = np.random.default_rng(seed + (0 if train else 10_000))
        images = rng.normal(0.0, 0.7, size=(size, 3, 32, 32)).astype("float32")

        # Add class-dependent color/position signals.
        labels = np.zeros((size,), dtype="int64")
        for i in range(size):
            c = i % 10
            labels[i] = c
            channel = c % 3
            row0 = 0 if c < 5 else 16
            col0 = (c % 5) * 6
            images[i, channel, row0 : row0 + 12, col0 : col0 + 12] += 1.4
            images[i, :, 8:24, 8:24] += (c - 4.5) / 15.0

        perm = rng.permutation(size)
        self.images = torch.from_numpy(images[perm]).clamp(-2.5, 2.5)
        self.labels = torch.from_numpy(labels[perm])

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, idx: int):
        return self.images[idx], self.labels[idx]


def _seeded_indices(n: int, take: int, seed: int) -> list[int]:
    take = min(max(int(take), 1), n)
    rng = np.random.default_rng(seed)
    return rng.permutation(n)[:take].tolist()


def _make_cifar10_loaders(
    *,
    data_dir: Path,
    batch_size: int,
    train_subset: int,
    val_subset: int,
    seed: int,
    num_workers: int,
    download: bool,
) -> DataBundle:
    from torchvision import datasets, transforms

    train_tf = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )
    val_tf = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )

    # CIFAR-10 has 50k train examples. We make a deterministic train/val split
    # from the official train set to keep the public test set untouched.
    train_full = datasets.CIFAR10(root=str(data_dir), train=True, transform=train_tf, download=download)
    val_full = datasets.CIFAR10(root=str(data_dir), train=True, transform=val_tf, download=False)

    n = len(train_full)
    all_idx = _seeded_indices(n, n, seed)
    val_idx = all_idx[: min(val_subset, n)]
    train_pool = all_idx[min(val_subset, n) :]
    train_idx = train_pool[: min(train_subset, len(train_pool))]

    train_ds = Subset(train_full, train_idx)
    val_ds = Subset(val_full, val_idx)

    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return DataBundle(train_loader, val_loader, "cifar10", len(train_ds), len(val_ds))


def _make_synthetic_loaders(
    *,
    batch_size: int,
    train_subset: int,
    val_subset: int,
    seed: int,
    num_workers: int,
    mode: str,
) -> DataBundle:
    train_ds = SyntheticCifarLike(size=train_subset, seed=seed, train=True)
    val_ds = SyntheticCifarLike(size=val_subset, seed=seed, train=False)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        generator=generator,
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return DataBundle(train_loader, val_loader, mode, len(train_ds), len(val_ds))


def get_dataloaders(
    *,
    dataset: str,
    data_dir: Path,
    batch_size: int,
    train_subset: int,
    val_subset: int,
    seed: int,
    num_workers: int = 0,
    download_cifar: bool = False,
    allow_synthetic_fallback: bool = True,
) -> DataBundle:
    if dataset == "synthetic":
        return _make_synthetic_loaders(
            batch_size=batch_size,
            train_subset=train_subset,
            val_subset=val_subset,
            seed=seed,
            num_workers=num_workers,
            mode="synthetic",
        )

    if dataset != "cifar10":
        raise ValueError(f"unknown dataset: {dataset!r}")

    try:
        return _make_cifar10_loaders(
            data_dir=data_dir,
            batch_size=batch_size,
            train_subset=train_subset,
            val_subset=val_subset,
            seed=seed,
            num_workers=num_workers,
            download=download_cifar,
        )
    except Exception as exc:
        if not allow_synthetic_fallback:
            raise
        print(f"[data] CIFAR-10 unavailable ({type(exc).__name__}: {exc}); using synthetic fallback.", flush=True)
        return _make_synthetic_loaders(
            batch_size=batch_size,
            train_subset=train_subset,
            val_subset=val_subset,
            seed=seed,
            num_workers=num_workers,
            mode="synthetic_fallback",
        )
