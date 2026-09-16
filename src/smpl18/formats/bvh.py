"""Read a Biovision Hierarchy (``.bvh``) file: the joint tree, its channels, the motion block.

The hierarchy is parsed by a small recursive descent over whitespace tokens (``ROOT`` /
``JOINT`` / ``End Site`` blocks in braces), which copes with any indentation and with several
joints per line. The motion block is read line by line so that a row with the wrong number of
values is reported by row. No forward kinematics is evaluated here; the channel order each
joint declares is kept as written, and the meaning of ``Xrotation`` before ``Yrotation`` is a
skeleton model's concern.
"""

from __future__ import annotations

import os
import pathlib
from dataclasses import dataclass

import numpy as np

from .errors import FormatError

__all__ = ["BvhFile", "BvhJoint", "CHANNEL_NAMES", "parse", "read"]

CHANNEL_NAMES = frozenset(
    {"Xposition", "Yposition", "Zposition", "Xrotation", "Yrotation", "Zrotation"}
)


@dataclass(frozen=True)
class BvhJoint:
    """One joint in file order: its parent, rest offset, channels, and any End Site offset."""

    name: str
    parent: int  # -1 for a root
    offset: tuple[float, float, float]
    channels: tuple[str, ...]
    #: Index of this joint's first channel in ``BvhFile.frames``.
    channel_start: int
    end_site: tuple[float, float, float] | None


@dataclass(frozen=True)
class BvhFile:
    """The joint tree, the frame period, and every motion row as ``(frames, channels)`` float64."""

    joints: tuple[BvhJoint, ...]
    frame_time_s: float
    frames: np.ndarray

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(j.name for j in self.joints)

    @property
    def parents(self) -> tuple[int, ...]:
        return tuple(j.parent for j in self.joints)

    @property
    def channel_count(self) -> int:
        return sum(len(j.channels) for j in self.joints)

    @property
    def frame_count(self) -> int:
        return int(self.frames.shape[0])

    def channel_index(self, joint: str, channel: str) -> int:
        """Column of ``channel`` on ``joint`` in ``frames``; a KeyError names what is missing."""
        for entry in self.joints:
            if entry.name == joint:
                if channel not in entry.channels:
                    raise KeyError(f"joint {joint!r} declares no channel {channel!r}")
                return entry.channel_start + entry.channels.index(channel)
        raise KeyError(f"no joint named {joint!r}")

    def channels_of(self, joint: str) -> np.ndarray:
        """The ``(frames, len(channels))`` block of one joint, in its own channel order."""
        for entry in self.joints:
            if entry.name == joint:
                return self.frames[:, entry.channel_start:entry.channel_start + len(entry.channels)]
        raise KeyError(f"no joint named {joint!r}")


class _Tokens:
    def __init__(self, tokens: list[str]):
        self._tokens = tokens
        self.position = 0

    def peek(self) -> str | None:
        return self._tokens[self.position] if self.position < len(self._tokens) else None

    def take(self) -> str:
        token = self.peek()
        if token is None:
            raise FormatError("hierarchy ends before its braces close")
        self.position += 1
        return token

    def expect(self, literal: str) -> None:
        token = self.take()
        if token != literal:
            raise FormatError(f"expected {literal!r} in the hierarchy, found {token!r}")

    def floats(self, count: int, what: str) -> tuple[float, ...]:
        out = []
        for _ in range(count):
            token = self.take()
            try:
                out.append(float(token))
            except ValueError as exc:
                raise FormatError(f"{what}: {token!r} is not a number") from exc
        return tuple(out)


def _parse_joint(tokens: _Tokens, parent: int, joints: list[BvhJoint], channel_start: int) -> int:
    """Parse one ROOT/JOINT block into ``joints``; return the channel index after its subtree."""
    name = tokens.take()
    tokens.expect("{")
    offset: tuple[float, float, float] | None = None
    channels: tuple[str, ...] = ()
    end_site: tuple[float, float, float] | None = None
    index = len(joints)
    joints.append(BvhJoint(name, parent, (0.0, 0.0, 0.0), (), channel_start, None))
    next_start = channel_start
    seen_channels = False
    while True:
        token = tokens.take()
        if token == "}":
            break
        if token == "OFFSET":
            offset = tokens.floats(3, f"OFFSET of {name}")  # type: ignore[assignment]
        elif token == "CHANNELS":
            if seen_channels:
                raise FormatError(f"joint {name!r} declares CHANNELS twice")
            seen_channels = True
            count_text = tokens.take()
            try:
                count = int(count_text)
            except ValueError as exc:
                raise FormatError(f"CHANNELS of {name}: {count_text!r} is not a count") from exc
            channels = tuple(tokens.take() for _ in range(count))
            unknown = [c for c in channels if c not in CHANNEL_NAMES]
            if unknown:
                raise FormatError(f"joint {name!r} declares unknown channels {unknown}")
            next_start = channel_start + count
        elif token == "JOINT":
            next_start = _parse_joint(tokens, index, joints, next_start)
        elif token == "End":
            tokens.expect("Site")
            tokens.expect("{")
            tokens.expect("OFFSET")
            end_site = tokens.floats(3, f"End Site of {name}")  # type: ignore[assignment]
            tokens.expect("}")
        else:
            raise FormatError(f"unexpected token {token!r} inside joint {name!r}")
    if offset is None:
        raise FormatError(f"joint {name!r} declares no OFFSET")
    joints[index] = BvhJoint(name, parent, offset, channels, channel_start, end_site)
    return next_start


def _split_sections(text: str) -> tuple[list[str], list[str]]:
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if line.strip() == "MOTION":
            return lines[:index], lines[index + 1:]
    raise FormatError("no MOTION section")


def _header_value(lines: list[str], slot: int, key: str) -> str:
    if slot >= len(lines):
        raise FormatError(f"MOTION section ends before {key!r}")
    line = lines[slot].strip()
    if not line.lower().startswith(key.lower()):
        raise FormatError(f"expected {key!r} in the MOTION section, found {line!r}")
    return line[len(key):].strip()


def parse(text: str) -> BvhFile:
    """Parse BVH text."""
    hierarchy_lines, motion_lines = _split_sections(text)
    tokens = _Tokens(" ".join(hierarchy_lines).split())
    if tokens.take() != "HIERARCHY":
        raise FormatError("file does not start with HIERARCHY")
    joints: list[BvhJoint] = []
    channel_start = 0
    while tokens.peek() is not None:
        if tokens.take() != "ROOT":
            raise FormatError("expected ROOT at the top of the hierarchy")
        channel_start = _parse_joint(tokens, -1, joints, channel_start)
    if not joints:
        raise FormatError("hierarchy declares no joints")

    motion = [line for line in motion_lines if line.strip()]
    frames_text = _header_value(motion, 0, "Frames:")
    frame_time_text = _header_value(motion, 1, "Frame Time:")
    try:
        frame_count = int(frames_text)
        frame_time = float(frame_time_text)
    except ValueError as exc:
        raise FormatError(f"Frames {frames_text!r} / Frame Time {frame_time_text!r}") from exc

    rows = motion[2:]
    if len(rows) != frame_count:
        raise FormatError(f"Frames: declares {frame_count} rows, the file carries {len(rows)}")
    frames = np.empty((frame_count, channel_start), dtype=np.float64)
    for index, row in enumerate(rows):
        fields = row.split()
        if len(fields) != channel_start:
            raise FormatError(
                f"motion row {index} has {len(fields)} values against {channel_start} channels"
            )
        try:
            frames[index] = [float(f) for f in fields]
        except ValueError as exc:
            raise FormatError(f"motion row {index} is not numeric") from exc
    return BvhFile(joints=tuple(joints), frame_time_s=frame_time, frames=frames)


def read(path: str | os.PathLike[str]) -> BvhFile:
    """Parse one ``.bvh`` file."""
    return parse(pathlib.Path(path).read_text(encoding="utf-8", errors="replace"))
