import torch
import random
import numpy as np

from pathlib import Path


def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def set_training_device(device=None):
    if device is None or device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"

    device = device.lower()

    if device == "cpu":
        return "cpu"

    if device == "cuda":
        if torch.cuda.is_available():
            return "cuda"
        raise RuntimeError("CUDA requested but no GPU is available.")

    raise ValueError(f"Unknown device '{device}'. Expected 'cpu', 'cuda', or 'auto'.")


def set_output_directory(root_directory: str, parameters: dict) -> Path:
    parts = []
    key_map = {
        "crank_nicolson": "cn",
    }

    for key in sorted(parameters):
        name = key_map.get(key, key)
        value = parameters[key]

        if isinstance(value, bool):
            value = str(value).lower()

        parts.append(f"{name}-{value}")

    output_directory = Path(root_directory) / "_".join(parts)
    output_directory.mkdir(parents=True, exist_ok=True)

    return output_directory


import numpy as np
import torch


class preserve_rng_state:
    """Snapshots torch/cuda/numpy RNG state on entry and restores it on exit,
    so any randomness consumed inside the block leaves no trace afterward."""

    def __enter__(self):
        self.torch_state = torch.get_rng_state()
        self.cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        self.numpy_state = np.random.get_state()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        torch.set_rng_state(self.torch_state)
        if self.cuda_state is not None:
            torch.cuda.set_rng_state_all(self.cuda_state)
        np.random.set_state(self.numpy_state)


