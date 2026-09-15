"""The exception every reader raises when a file is not what its own header or format says."""

from __future__ import annotations

__all__ = ["FormatError"]


class FormatError(RuntimeError):
    """The file does not match the layout its own format or header declares."""
