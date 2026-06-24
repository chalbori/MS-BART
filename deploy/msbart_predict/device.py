"""Device auto-detection for CUDA / Apple MPS / CPU."""
import os
import torch


def pick_device(prefer: str = "auto") -> torch.device:
    """Pick the best available torch device.

    prefer: "auto" (default), "cuda", "mps", or "cpu".
    Falls back gracefully if the preferred backend is unavailable.
    """
    prefer = (prefer or "auto").lower()

    def cuda_ok():
        return torch.cuda.is_available()

    def mps_ok():
        return getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()

    if prefer == "cuda":
        return torch.device("cuda:0") if cuda_ok() else torch.device("cpu")
    if prefer == "mps":
        return torch.device("mps") if mps_ok() else torch.device("cpu")
    if prefer == "cpu":
        return torch.device("cpu")

    # auto
    if cuda_ok():
        return torch.device("cuda:0")
    if mps_ok():
        # MIST/BART float ops are MPS-friendly; allow CPU fallback for unsupported ops.
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        return torch.device("mps")
    return torch.device("cpu")
