from __future__ import annotations

import os
import random
from typing import Any

import numpy as np


def set_random_seeds(seed: int) -> dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch_info: dict[str, Any]
    try:
        import torch
    except ImportError:
        torch_info = {"torch_installed": False}
    else:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch_info = {
            "torch_installed": True,
            "torch_cuda_available": bool(torch.cuda.is_available()),
        }
    return {"seed": int(seed), **torch_info}
