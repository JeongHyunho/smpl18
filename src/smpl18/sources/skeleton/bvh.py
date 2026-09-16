"""Forward kinematics of a Biovision Hierarchy skeleton.

A BVH joint is a frame placed by its parent: first the rest ``OFFSET``, in the parent's frame,
plus whatever position channels the joint carries; then its rotation channels, composed in the
order the joint lists them. ``Zrotation Xrotation Yrotation`` is ``Rz @ Rx @ Ry``, each turn
about the axis the previous ones left, and reversing that order is a different pose. A joint's
frame is also the segment it moves, so every joint is a body of the same name, and an ``End
Site`` is one more centre on its joint's segment.

The file never says what its length unit is, so the caller states it (``length_scale``, metres
per file unit) and nothing guesses. Channel values are degrees and file units; ``motion`` takes
radians and metres, and ``coordinates_from_frames`` converts a file's rows to those.
"""

from __future__ import annotations

import os
from collections.abc import Mapping

import numpy as np

from ...formats import bvh
from ...formats.bvh import BvhFile
from ..base import SegmentPlacement, SkeletonModel

__all__ = ["END_SITE_SUFFIX", "BvhSkeleton", "coordinates_from_frames", "read"]

#: ``joint_centres`` names an End Site after its joint with this suffix.
END_SITE_SUFFIX = ".end"

_ROTATION_AXES = {"Xrotation": 0, "Yrotation": 1, "Zrotation": 2}
_POSITION_AXES = {"Xposition": 0, "Yposition": 1, "Zposition": 2}


def _about(axis: int, angle: np.ndarray) -> np.ndarray:
    """``[T, 3, 3]`` rotations by ``angle`` (radians) about coordinate axis ``axis``."""
    cosine, sine = np.cos(angle), np.sin(angle)
    first, second = ((1, 2), (2, 0), (0, 1))[axis]
    out = np.zeros(angle.shape + (3, 3))
    out[..., axis, axis] = 1.0
    out[..., first, first] = cosine
    out[..., second, second] = cosine
    out[..., first, second] = -sine
    out[..., second, first] = sine
    return out


class BvhSkeleton(SkeletonModel):
    """A BVH hierarchy as a skeleton model: one body per joint, one coordinate per channel.

    Coordinates are named ``"<joint>.<channel>"`` in the file's column order; rotation channels
    are radians and position channels metres. Every channel must be given to ``motion``: a BVH
    declares no values for a missing one.
    """

    def __init__(self, bvh_file: BvhFile, *, length_scale: float):
        scale = float(length_scale)
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError(
                f"length_scale must be a positive number of metres per file unit, got "
                f"{length_scale!r}"
            )
        names = bvh_file.names
        repeated = sorted({name for name in names if names.count(name) > 1})
        if repeated:
            raise ValueError(f"joint names must be unique; repeated: {repeated}")
        end_sites = {
            f"{joint.name}{END_SITE_SUFFIX}"
            for joint in bvh_file.joints
            if joint.end_site is not None
        }
        clashes = sorted(end_sites & set(names))
        if clashes:
            raise ValueError(f"joint names collide with End Site centre names: {clashes}")
        self._file = bvh_file
        self._scale = scale
        columns = sorted(
            (joint.channel_start + offset, f"{joint.name}.{channel}")
            for joint in bvh_file.joints
            for offset, channel in enumerate(joint.channels)
        )
        self._coordinates = tuple(name for _, name in columns)
        self._rotational = tuple(
            name for name in self._coordinates if name.rsplit(".", 1)[1] in _ROTATION_AXES
        )

    @property
    def file(self) -> BvhFile:
        return self._file

    @property
    def length_scale(self) -> float:
        """Metres per file unit, as the caller stated it."""
        return self._scale

    # --- SkeletonModel ---------------------------------------------------------------------------

    @property
    def bodies(self) -> tuple[str, ...]:
        return self._file.names

    @property
    def joints(self) -> tuple[str, ...]:
        return self._file.names

    @property
    def parents(self) -> Mapping[str, str | None]:
        names = self._file.names
        return {
            joint.name: (None if joint.parent < 0 else names[joint.parent])
            for joint in self._file.joints
        }

    @property
    def joint_child_bodies(self) -> Mapping[str, str]:
        return {name: name for name in self._file.names}

    @property
    def coordinate_names(self) -> tuple[str, ...]:
        return self._coordinates

    @property
    def rotational_coordinates(self) -> tuple[str, ...]:
        return self._rotational

    def rest_transform(self, body: str) -> tuple[np.ndarray, np.ndarray]:
        """The joint's frame in its parent's with every channel at zero: its scaled offset."""
        for joint in self._file.joints:
            if joint.name == body:
                return np.eye(3), np.asarray(joint.offset, dtype=np.float64) * self._scale
        raise KeyError(f"no joint named {body!r}")

    def motion(self, coordinates: Mapping[str, np.ndarray]) -> SegmentPlacement:
        """World rotation and origin per joint, and every joint's origin plus one centre per
        End Site (``"<joint>.end"``), for ``T`` frames of channel values."""
        values, frames = self._values(coordinates)
        names = self._file.names
        rotations: dict[str, np.ndarray] = {}
        positions: dict[str, np.ndarray] = {}
        centres: dict[str, np.ndarray] = {}
        for joint in self._file.joints:
            rotation = np.broadcast_to(np.eye(3), (frames, 3, 3))
            translation = np.tile(np.asarray(joint.offset, dtype=np.float64) * self._scale,
                                  (frames, 1))
            for channel in joint.channels:
                value = values[f"{joint.name}.{channel}"]
                if channel in _ROTATION_AXES:
                    rotation = rotation @ _about(_ROTATION_AXES[channel], value)
                else:
                    translation[:, _POSITION_AXES[channel]] += value
            if joint.parent < 0:
                world_rotation = np.array(rotation)
                world_position = translation
            else:
                parent = names[joint.parent]
                parent_rotation = rotations[parent]
                world_rotation = parent_rotation @ rotation
                world_position = (
                    positions[parent] + (parent_rotation @ translation[:, :, None])[:, :, 0]
                )
            rotations[joint.name] = world_rotation
            positions[joint.name] = world_position
            centres[joint.name] = world_position
            if joint.end_site is not None:
                tip = np.asarray(joint.end_site, dtype=np.float64) * self._scale
                centres[f"{joint.name}{END_SITE_SUFFIX}"] = world_position + world_rotation @ tip
        return SegmentPlacement(rotations=rotations, positions=positions, joint_centres=centres)

    def _values(self, coordinates: Mapping[str, np.ndarray]) -> tuple[dict[str, np.ndarray], int]:
        known = set(self._coordinates)
        unknown = sorted(set(coordinates) - known)
        if unknown:
            raise ValueError(f"coordinates the skeleton does not declare: {unknown}")
        missing = [name for name in self._coordinates if name not in coordinates]
        if missing:
            raise ValueError(f"no values for channels {missing}")
        values: dict[str, np.ndarray] = {}
        for name, value in coordinates.items():
            array = np.asarray(value, dtype=np.float64)
            if array.ndim != 1:
                raise ValueError(f"coordinate {name!r} must be a (T,) array, got {array.shape}")
            values[name] = array
        lengths = sorted({array.shape[0] for array in values.values()})
        if len(lengths) > 1:
            raise ValueError(f"coordinates disagree on the number of frames: {lengths}")
        if not lengths:
            raise ValueError("the hierarchy has no channels, so nothing says how many frames to place")
        return values, lengths[0]


def coordinates_from_frames(skeleton: BvhSkeleton, bvh_file: BvhFile) -> dict[str, np.ndarray]:
    """The file's motion rows as ``skeleton`` coordinates: degrees to radians, file units to
    metres by the skeleton's ``length_scale``. The file must declare the skeleton's hierarchy
    and channels."""
    layout = [(joint.name, joint.channels) for joint in bvh_file.joints]
    expected = [(joint.name, joint.channels) for joint in skeleton.file.joints]
    if layout != expected:
        raise ValueError("the file's joints and channels differ from the skeleton's")
    values: dict[str, np.ndarray] = {}
    for joint in bvh_file.joints:
        for offset, channel in enumerate(joint.channels):
            column = np.array(bvh_file.frames[:, joint.channel_start + offset], dtype=np.float64)
            if channel in _ROTATION_AXES:
                column = np.deg2rad(column)
            else:
                column = column * skeleton.length_scale
            values[f"{joint.name}.{channel}"] = column
    return values


def read(path: str | os.PathLike[str], *, length_scale: float) -> BvhSkeleton:
    """The skeleton of one ``.bvh`` file."""
    return BvhSkeleton(bvh.read(path), length_scale=length_scale)
