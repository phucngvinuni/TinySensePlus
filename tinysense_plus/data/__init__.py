from typing import Dict, Optional

import numpy as np
import torch
import yaml
from sklearn.model_selection import train_test_split

from .mmfi import make_dataloader, make_datasets


def build_mmfi_dataloaders(
    data_root: str,
    split_config_path: str,
    max_samples: Optional[int] = None,
) -> Dict[str, torch.utils.data.DataLoader]:
    """Build train / valid / test loaders.

    The training partition comes from the split config. The held-out partition is
    split 50/50 at frame level (``random_state=41``) into validation and test sets,
    which are materialised in memory.

    Args:
        max_samples: optional cap on the number of samples per partition, for quick
            smoke tests.
    """
    with open(split_config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    train_dataset, held_out = make_datasets(data_root, config)
    generator = torch.Generator().manual_seed(config["init_rand_seed"])

    val_idx, test_idx = train_test_split(
        np.arange(len(held_out)), test_size=0.5, random_state=41
    )
    if max_samples is not None:
        train_dataset = torch.utils.data.Subset(train_dataset, range(max_samples))
        val_idx, test_idx = val_idx[:max_samples], test_idx[:max_samples]
    val_data = [held_out[i] for i in val_idx]
    test_data = [held_out[i] for i in test_idx]

    return {
        "train": make_dataloader(
            train_dataset, True, generator, config["train_loader"]["batch_size"]
        ),
        "valid": make_dataloader(
            val_data, False, generator, config["val_loader"]["batch_size"]
        ),
        "test": make_dataloader(
            test_data, False, generator, config["test_loader"]["batch_size"]
        ),
    }
