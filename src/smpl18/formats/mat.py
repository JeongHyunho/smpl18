"""Read a MATLAB ``.mat`` file: variables by name, structs kept as structs.

``scipy.io.loadmat`` is called with ``squeeze_me=True`` and ``struct_as_record=False``, so a
MATLAB scalar arrives as a Python number, a struct as a ``mat_struct`` whose fields are
attributes, and a struct array as an object array of them. ``struct_to_dict`` turns that
into nested dicts and lists when a caller prefers plain containers. Nothing here knows what
any variable means.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import scipy.io
from scipy.io.matlab import mat_struct

__all__ = ["read", "struct_to_dict"]


def read(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Every variable in the file by name; the ``__header__``-style bookkeeping keys are dropped."""
    loaded = scipy.io.loadmat(os.fspath(path), squeeze_me=True, struct_as_record=False)
    return {name: value for name, value in loaded.items() if not name.startswith("__")}


def struct_to_dict(value: Any) -> Any:
    """``mat_struct`` to dict, recursively; object arrays (cells, struct arrays) to nested lists.

    Numeric arrays, strings and scalars are returned as they are.
    """
    if isinstance(value, mat_struct):
        return {name: struct_to_dict(getattr(value, name)) for name in value._fieldnames}
    if isinstance(value, np.ndarray) and value.dtype == object:
        if value.ndim == 0:
            return struct_to_dict(value.item())
        return [struct_to_dict(item) for item in value]
    if isinstance(value, list):
        return [struct_to_dict(item) for item in value]
    return value
