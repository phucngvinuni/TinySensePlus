import os
import random

import numpy as np
import torch


def set_reproducibility(seed: int, deterministic: bool = False) -> None:
    """Seed all RNGs. ``deterministic=True`` trades speed for run-to-run determinism."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
