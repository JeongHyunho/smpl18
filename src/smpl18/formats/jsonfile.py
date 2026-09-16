"""Read a JSON file as UTF-8 and return the object it encodes."""

from __future__ import annotations

import json
import os
import pathlib
from typing import Any

__all__ = ["read"]


def read(path: str | os.PathLike[str]) -> Any:
    """The decoded object (dict, list, or scalar) exactly as the file states it."""
    return json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
