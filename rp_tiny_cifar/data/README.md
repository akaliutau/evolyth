# Data directory

This directory is intentionally committed without CIFAR-10 files.

Prepare CIFAR-10 locally with:

```bash
python scripts/prepare_cifar.py --data-dir ./data
```

Or let the immutable trainer download it during a run:

```bash
python train_eval.py --dataset cifar10 --download-cifar
```

For offline smoke tests, use:

```bash
python train_eval.py --dry-run --dataset synthetic
```

The synthetic dataset is only for contract checks and no-network demos. Use real CIFAR-10 for leaderboard claims.
