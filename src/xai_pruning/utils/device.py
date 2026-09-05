"""Device helpers shared by evaluation and benchmarking."""

import torch


def get_device(requested: str | torch.device | None = None) -> torch.device:
    """Resolve an explicit device or choose CUDA when it is available."""

    if requested is not None:
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def synchronize(device: str | torch.device) -> None:
    """Synchronize CUDA and remain a no-op for CPU devices."""

    resolved = torch.device(device)
    if resolved.type == "cuda":
        torch.cuda.synchronize(resolved)
