"""Resolve weight paths and make the vendored MIST source importable.

Resolution order for each artifact:
  1. explicit argument passed to MSBartPredictor
  2. environment variable (MSBART_WEIGHTS / MIST_CKPT / MIST_SRC)
  3. default location relative to the deploy bundle root
"""
import os
import sys
from pathlib import Path

# .../deploy/msbart_predict/config.py -> bundle root is .../deploy
BUNDLE_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_MSBART_WEIGHTS = BUNDLE_ROOT / "weights" / "msbart"
DEFAULT_MIST_CKPT = BUNDLE_ROOT / "weights" / "mist.ckpt"
# Dir that CONTAINS the `mist` package (so it can go on sys.path for `import mist`).
DEFAULT_MIST_SRC = BUNDLE_ROOT / "vendor"


def resolve_msbart_weights(explicit=None) -> Path:
    p = explicit or os.environ.get("MSBART_WEIGHTS") or DEFAULT_MSBART_WEIGHTS
    return Path(p)


def resolve_mist_ckpt(explicit=None) -> Path:
    p = explicit or os.environ.get("MIST_CKPT") or DEFAULT_MIST_CKPT
    return Path(p)


def ensure_mist_importable(explicit_src=None) -> None:
    """Add the vendored MIST source dir to sys.path if `mist` isn't already importable."""
    try:
        import mist  # noqa: F401
        return
    except ImportError:
        pass

    src = explicit_src or os.environ.get("MIST_SRC") or DEFAULT_MIST_SRC
    src = Path(src)
    if src.exists():
        sys.path.insert(0, str(src))
    try:
        import mist  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "Could not import `mist`. Either pip-install the MIST repo, or place its "
            f"source so that `{DEFAULT_MIST_SRC}/mist/` exists (or set MIST_SRC to the "
            f"dir containing the `mist` package). Original error: {e}"
        )
