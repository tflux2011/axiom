"""
AXIOM Utility Functions

Shared helpers for serialisation, logging, and device management.
No secrets or credentials are handled here.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator

import torch

logger = logging.getLogger("axiom")


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(level: int = logging.INFO) -> None:
    """Configure structured logging for AXIOM."""
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] %(name)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    root = logging.getLogger("axiom")
    root.setLevel(level)
    if not root.handlers:
        root.addHandler(handler)


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

@contextmanager
def timer(label: str) -> Generator[None, None, None]:
    """Context manager that logs elapsed time for a block."""
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    logger.info("%s completed in %.4f s", label, elapsed)


# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------

def get_device(requested: str = "cpu") -> torch.device:
    """Return a valid torch device, falling back gracefully."""
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if requested == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------

def save_tensor(tensor: torch.Tensor, path: Path) -> None:
    """Save a tensor to disk in a portable format."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tensor, path)
    logger.info("Saved tensor [%s] → %s", tensor.shape, path)


def load_tensor(path: Path, device: str = "cpu") -> torch.Tensor:
    """Load a tensor from disk."""
    if not path.exists():
        raise FileNotFoundError(f"Tensor file not found: {path}")
    tensor = torch.load(path, map_location=device, weights_only=False)
    logger.info("Loaded tensor [%s] ← %s", tensor.shape, path)
    return tensor


def content_hash(text: str) -> str:
    """Return a stable SHA-256 hex digest for deduplication."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# JSON helpers (for metadata / results)
# ---------------------------------------------------------------------------

def save_json(data: Any, path: Path) -> None:
    """Write JSON with consistent formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False, default=str)


def load_json(path: Path) -> Any:
    """Read JSON from disk."""
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)
