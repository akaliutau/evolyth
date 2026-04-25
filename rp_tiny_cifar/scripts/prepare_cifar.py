from __future__ import annotations

import argparse
from pathlib import Path

from torchvision import datasets


def main() -> None:
    parser = argparse.ArgumentParser(description="Download CIFAR-10 into the RP data directory.")
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    datasets.CIFAR10(root=str(data_dir), train=True, download=True)
    datasets.CIFAR10(root=str(data_dir), train=False, download=True)
    print(f"CIFAR-10 prepared at {data_dir.resolve()}")


if __name__ == "__main__":
    main()
