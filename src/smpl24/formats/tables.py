"""The table shared by the marker-trajectory readers (``trc`` and ``c3d``)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["TrcTable"]


@dataclass(frozen=True)
class TrcTable:
    """Labelled marker trajectories as the file stores them: its labels, its units, its frame.

    ``positions`` is ``(frames, markers, 3)`` float64 with NaN wherever ``valid`` is False. A
    sample is invalid when the file marks it so (an empty field in a ``.trc``, a negative
    residual in a ``.c3d``) or when it is not finite. Nothing else is inferred: a marker a
    writer parks at the origin is a number here, and blanking it is a policy for another layer.
    """

    labels: tuple[str, ...]
    data_rate_hz: float
    units: str
    positions: np.ndarray
    valid: np.ndarray
    frame_numbers: np.ndarray
    times_s: np.ndarray
    #: The camera rate when the file declares one apart from the data rate (``.trc`` does).
    camera_rate_hz: float | None = None
    #: Every other header field the file declares, uninterpreted, keyed as the file spells it.
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        frames, markers = self.positions.shape[0], len(self.labels)
        if self.positions.shape != (frames, markers, 3):
            raise ValueError(
                f"positions must be (frames, {markers}, 3), got {self.positions.shape}"
            )
        if self.valid.shape != (frames, markers):
            raise ValueError(f"valid must be ({frames}, {markers}), got {self.valid.shape}")
        if self.frame_numbers.shape != (frames,) or self.times_s.shape != (frames,):
            raise ValueError(
                f"frame_numbers and times_s must be ({frames},), got "
                f"{self.frame_numbers.shape} and {self.times_s.shape}"
            )

    @property
    def frame_count(self) -> int:
        return int(self.positions.shape[0])

    def index_of(self, label: str) -> int:
        """Column of ``label``; a KeyError names the label when the file has no such marker."""
        try:
            return self.labels.index(label)
        except ValueError as exc:
            raise KeyError(f"no marker named {label!r}") from exc

    def positions_of(self, label: str) -> np.ndarray:
        """The ``(frames, 3)`` trajectory of one marker, NaN where the sample is invalid."""
        return self.positions[:, self.index_of(label)]
