"""Device selection helpers for HPC and local execution."""

import torch


def get_device(requested: str = "auto") -> torch.device:
    """Return a torch device based on the request.

    Parameters
    ----------
    requested : str
        ``"auto"`` (default) - use CUDA if available, else CPU.
        ``"cpu"``  - force CPU.
        ``"cuda"`` - force CUDA (raises if unavailable).
        ``"cuda:N"`` - specific GPU index.
    """
    if requested == "auto":
        if torch.cuda.is_available():
            dev = torch.device("cuda")
        else:
            dev = torch.device("cpu")
    else:
        dev = torch.device(requested)
        if dev.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")

    return dev


def print_device_info(device: torch.device) -> None:
    """Print device diagnostics."""
    print(f"Device          : {device}")
    print(f"PyTorch version : {torch.__version__}")
    if device.type == "cuda":
        idx = device.index or 0
        print(f"CUDA device     : {torch.cuda.get_device_name(idx)}")
        mem = torch.cuda.get_device_properties(idx).total_memory
        print(f"GPU memory      : {mem / 1e9:.1f} GB")
    print()
