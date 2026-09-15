"""Read a NumPy ``.npz`` archive: every array by its key, no pickled objects."""

from __future__ import annotations

import os

import numpy as np

__all__ = ["read"]


def read(path: str | os.PathLike[str]) -> dict[str, np.ndarray]:
    """Every array in the archive, keyed as stored. Object arrays are refused by NumPy."""
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}
