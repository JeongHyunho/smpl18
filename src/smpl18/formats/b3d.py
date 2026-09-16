"""Address and decode ``.b3d`` containers without nimblephysics.

A ``.b3d`` is an 8-byte little-endian header length, a ``SubjectOnDiskHeader`` protobuf, then a
flat run of fixed-width records: for each trial, for each frame, one ``SubjectOnDiskSensorFrame``
followed by one ``SubjectOnDiskProcessingPassFrame`` per processing pass *of that trial*. There
is no index and no delimiter, so a frame is reached only by summing the trials before it -- and
a slip lands on a neighbouring record that decodes perfectly well into the wrong pose. The
layout is therefore checked against the file's own length before anything is read.

Pass counts vary per trial, so a pass of a given type does not sit at a fixed index; it is looked
up per trial from that trial's own pass headers, by name, and the caller says which one it wants.

Everything comes back as the file stores it: the header fields unmapped, the frames in the
model's own world frame (Y-up, as OpenSim models and this format lay it out; the embedded model's
``gravity`` is the file's own statement of it), no unit changes. Files are opened read-only and
seeked; a single subject can exceed 2 GB and is never read whole.
"""

from __future__ import annotations

import os
import pathlib
import struct
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

from ._vendor.nimblephysics import SubjectOnDisk_pb2 as _protobuf
from .errors import FormatError
from .osim import OsimModel
from .osim import parse as _parse_osim

__all__ = [
    "B3dFile",
    "B3dLayoutError",
    "B3dPass",
    "B3dPassFrames",
    "B3dSensorFrames",
    "B3dSubject",
    "B3dTrial",
    "read",
    "read_pass_frames",
    "read_sensor_frames",
]

_HEADER_LENGTH_BYTES = 8
#: The up axis of the world frame the format lays its data out in (OpenSim's convention).
UP_AXIS = "y"


class B3dLayoutError(FormatError):
    """The file does not match the layout its own header declares."""


def _enum_name(enum, value: int) -> str:
    """The enum's name for ``value``, or the number as text when the file uses one we lack."""
    try:
        return enum.Name(value)
    except ValueError:
        return str(value)


@dataclass(frozen=True)
class B3dSubject:
    """The subject fields of the header, exactly as stored: no mapping, no defaulting."""

    biological_sex: str
    height_m: float
    mass_kg: float
    age_years: int
    href: str
    notes: str
    subject_tags: tuple[str, ...]
    data_quality: str


@dataclass(frozen=True)
class B3dPass:
    """One processing pass declared file-wide: its type name and the model text it carries."""

    index: int
    name: str
    model_osim_text: str


@dataclass(frozen=True)
class B3dTrial:
    """One trial's header fields plus where its records start and how wide they are."""

    index: int
    name: str
    original_name: str
    split_index: int
    start_byte: int
    frame_size: int
    pass_names: tuple[str, ...]
    length: int
    timestep_s: float
    num_force_plates: int
    force_plate_corners: tuple[float, ...]
    trial_tags: tuple[str, ...]
    trial_type: str
    detected_trial_features: tuple[str, ...]
    marker_names_guessed: bool
    original_trial_start_frame: int
    original_trial_end_frame: int
    original_trial_start_time: float
    original_trial_end_time: float

    @property
    def pass_count(self) -> int:
        return len(self.pass_names)

    def pass_index(self, name: str) -> int:
        """Where this trial keeps its first pass named ``name``; not a per-file constant."""
        try:
            return self.pass_names.index(name)
        except ValueError as exc:
            raise B3dLayoutError(
                f"trial {self.index} has no {name} pass: {self.pass_names}"
            ) from exc

    @property
    def source_rate_hz(self) -> float:
        return 1.0 / self.timestep_s if self.timestep_s > 0 else float("nan")


@dataclass(frozen=True)
class B3dFile:
    """The header laid out: trials, passes, subject, and the sizes every seek depends on."""

    path: pathlib.Path
    header: object
    payload_offset: int
    trials: tuple[B3dTrial, ...]
    passes: tuple[B3dPass, ...]
    subject: B3dSubject
    num_dofs: int
    num_joints: int
    version: int
    ground_contact_bodies: tuple[str, ...]
    marker_names: tuple[str, ...]
    acc_names: tuple[str, ...]
    gyro_names: tuple[str, ...]
    emg_names: tuple[str, ...]
    emg_dim: int
    custom_value_names: tuple[str, ...]
    custom_value_lengths: tuple[int, ...]
    exo_dof_indices: tuple[int, ...]
    trailing_bytes: int
    #: The world frame's up axis as the format lays it out. Recorded, never applied.
    up_axis: str = UP_AXIS

    def model_text(self, pass_name: str) -> str:
        """The ``.osim`` text of the first file-wide pass named ``pass_name`` that carries one."""
        for entry in self.passes:
            if entry.name == pass_name and entry.model_osim_text:
                return entry.model_osim_text
        raise B3dLayoutError(
            f"{self.path.name}: no {pass_name} pass carries a model; passes are "
            f"{[p.name for p in self.passes]}"
        )

    def model(self, pass_name: str) -> OsimModel:
        """The model of that pass, parsed by ``formats.osim``."""
        return _parse_osim(self.model_text(pass_name))


@dataclass(frozen=True)
class B3dPassFrames:
    """Selected frames of one processing pass of one trial, field for field as the proto names them.

    Per-DOF fields are ``(frames, dofs)``; 3-vector fields ``(frames, n, 3)``; the wrench
    fields ``(frames, n, 6)``; fields whose width the proto leaves open stay ``(frames, k)``.
    A field the pass did not store has a zero trailing width.
    """

    trial_index: int
    pass_index: int
    pass_name: str
    frame_indices: np.ndarray
    timestamps_s: np.ndarray
    source_rate_hz: float
    pos: np.ndarray
    vel: np.ndarray
    acc: np.ndarray
    tau: np.ndarray
    ground_contact_wrench: np.ndarray
    ground_contact_center_of_pressure: np.ndarray
    ground_contact_torque: np.ndarray
    ground_contact_force: np.ndarray
    com_pos: np.ndarray
    com_vel: np.ndarray
    com_acc: np.ndarray
    root_frame_com_acc: np.ndarray
    root_frame_residual: np.ndarray
    root_frame_ground_contact_wrench: np.ndarray
    root_frame_ground_contact_center_of_pressure: np.ndarray
    root_frame_ground_contact_torques: np.ndarray
    root_frame_ground_contact_force: np.ndarray
    root_frame_joint_centers: np.ndarray
    world_frame_joint_centers: np.ndarray
    root_frame_spatial_velocity: np.ndarray
    root_frame_spatial_acceleration: np.ndarray
    root_frame_root_pos_history: np.ndarray
    root_frame_root_euler_history: np.ndarray


@dataclass(frozen=True)
class B3dSensorFrames:
    """Selected raw sensor frames of one trial, as the proto names them."""

    trial_index: int
    frame_indices: np.ndarray
    timestamps_s: np.ndarray
    custom_values: np.ndarray
    marker_obs: np.ndarray
    acc_obs: np.ndarray
    gyro_obs: np.ndarray
    emg_obs: np.ndarray
    exo_obs: np.ndarray
    raw_force_plate_cop: np.ndarray
    raw_force_plate_torque: np.ndarray
    raw_force_plate_force: np.ndarray


# Field widths the proto documents: 3-vectors per joint / body / plate, 6-vectors per body.
_PASS_VECTOR_FIELDS: dict[str, int] = {
    "ground_contact_wrench": 6,
    "ground_contact_center_of_pressure": 3,
    "ground_contact_torque": 3,
    "ground_contact_force": 3,
    "root_frame_ground_contact_wrench": 6,
    "root_frame_ground_contact_center_of_pressure": 3,
    "root_frame_ground_contact_torques": 3,
    "root_frame_ground_contact_force": 3,
    "root_frame_joint_centers": 3,
    "world_frame_joint_centers": 3,
}
_PASS_FLAT_FIELDS = (
    "pos", "vel", "acc", "tau", "com_pos", "com_vel", "com_acc", "root_frame_com_acc",
    "root_frame_residual", "root_frame_spatial_velocity", "root_frame_spatial_acceleration",
    "root_frame_root_pos_history", "root_frame_root_euler_history",
)
_SENSOR_VECTOR_FIELDS: dict[str, int] = {
    "marker_obs": 3,
    "acc_obs": 3,
    "gyro_obs": 3,
    "raw_force_plate_cop": 3,
    "raw_force_plate_torque": 3,
    "raw_force_plate_force": 3,
}
_SENSOR_FLAT_FIELDS = ("custom_values", "emg_obs", "exo_obs")


def _pass_name(value: int) -> str:
    return _enum_name(_protobuf.ProcessingPassType, value)


def _read_header(path: pathlib.Path):
    with path.open("rb") as handle:
        prefix = handle.read(_HEADER_LENGTH_BYTES)
        if len(prefix) < _HEADER_LENGTH_BYTES:
            raise B3dLayoutError(f"{path.name}: shorter than the header length prefix")
        (header_size,) = struct.unpack("<q", prefix)
        if header_size <= 0:
            raise B3dLayoutError(f"{path.name}: declares a header of {header_size} bytes")
        blob = handle.read(header_size)
        if len(blob) < header_size:
            raise B3dLayoutError(
                f"{path.name}: header declares {header_size} bytes, file holds {len(blob)}"
            )
    header = _protobuf.SubjectOnDiskHeader()
    header.ParseFromString(blob)
    return header, _HEADER_LENGTH_BYTES + header_size


def read(path: str | os.PathLike[str], *, strict_size: bool = True) -> B3dFile:
    """Read the header, lay out every trial, and check the arithmetic against the file size.

    ``strict_size=False`` tolerates bytes past the declared layout and reports them in
    ``trailing_bytes``; a file shorter than its layout is refused either way.
    """
    path = pathlib.Path(path)
    header, payload_offset = _read_header(path)

    offset = payload_offset
    trials: list[B3dTrial] = []
    for index, trial in enumerate(header.trial_header):
        pass_names = tuple(_pass_name(p.type) for p in trial.processing_pass_header)
        frame_size = (
            header.raw_sensor_frame_size
            + len(pass_names) * header.processing_pass_frame_size
        )
        trials.append(
            B3dTrial(
                index=index,
                name=trial.name,
                original_name=trial.original_name,
                split_index=trial.split_index,
                start_byte=offset,
                frame_size=frame_size,
                pass_names=pass_names,
                length=trial.trial_length,
                timestep_s=trial.trial_timestep,
                num_force_plates=trial.num_force_plates,
                force_plate_corners=tuple(trial.force_plate_corners),
                trial_tags=tuple(trial.trial_tag),
                trial_type=_enum_name(_protobuf.BasicTrialType, trial.trial_type),
                detected_trial_features=tuple(
                    _enum_name(_protobuf.DetectedTrialFeature, f)
                    for f in trial.detected_trial_feature
                ),
                marker_names_guessed=trial.marker_names_guessed,
                original_trial_start_frame=trial.original_trial_start_frame,
                original_trial_end_frame=trial.original_trial_end_frame,
                original_trial_start_time=trial.original_trial_start_time,
                original_trial_end_time=trial.original_trial_end_time,
            )
        )
        offset += trial.trial_length * frame_size

    size = path.stat().st_size
    if offset > size:
        raise B3dLayoutError(
            f"{path.name}: layout needs {offset} bytes, file holds {size}"
        )
    trailing = size - offset
    if strict_size and trailing:
        raise B3dLayoutError(
            f"{path.name}: {trailing} bytes past the end of the declared layout"
        )

    return B3dFile(
        path=path,
        header=header,
        payload_offset=payload_offset,
        trials=tuple(trials),
        passes=tuple(
            B3dPass(index=i, name=_pass_name(p.pass_type), model_osim_text=p.model_osim_text)
            for i, p in enumerate(header.passes)
        ),
        subject=B3dSubject(
            biological_sex=header.biological_sex,
            height_m=header.height_m,
            mass_kg=header.mass_kg,
            age_years=header.age_years,
            href=header.href,
            notes=header.notes,
            subject_tags=tuple(header.subject_tag),
            data_quality=_enum_name(_protobuf.DataQuality, header.data_quality),
        ),
        num_dofs=header.num_dofs,
        num_joints=header.num_joints,
        version=header.version,
        ground_contact_bodies=tuple(header.ground_contact_body),
        marker_names=tuple(header.marker_name),
        acc_names=tuple(header.acc_name),
        gyro_names=tuple(header.gyro_name),
        emg_names=tuple(header.emg_name),
        emg_dim=header.emg_dim,
        custom_value_names=tuple(header.custom_value_name),
        custom_value_lengths=tuple(header.custom_value_length),
        exo_dof_indices=tuple(header.exo_dof_index),
        trailing_bytes=trailing,
    )


def _as_vectors(values: Sequence[float], width: int, what: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.size % width:
        raise B3dLayoutError(f"{what}: {array.size} values do not form {width}-vectors")
    return array.reshape(-1, width) if array.size else array.reshape(0, width)


def _stack(records: list[np.ndarray]) -> np.ndarray:
    return np.stack(records) if records else np.empty((0,), dtype=np.float64)


def _selected_frames(b3d: B3dFile, trial_index: int, frame_indices: Iterable[int]) -> np.ndarray:
    if not 0 <= trial_index < len(b3d.trials):
        raise B3dLayoutError(
            f"trial {trial_index} is outside 0..{len(b3d.trials) - 1}"
        )
    trial = b3d.trials[trial_index]
    wanted = np.asarray(list(frame_indices), dtype=np.int64)
    if wanted.size and (wanted.min() < 0 or wanted.max() >= trial.length):
        raise B3dLayoutError(
            f"trial {trial_index} holds {trial.length} frames; asked for "
            f"{wanted.min()}..{wanted.max()}"
        )
    return wanted


def _read_records(
    b3d: B3dFile, trial: B3dTrial, wanted: np.ndarray, within_frame: int, record_size: int, message
):
    """Decode one fixed-width record per wanted frame at ``within_frame`` bytes into the frame."""
    records = []
    with b3d.path.open("rb") as handle:
        for frame in wanted:
            handle.seek(trial.start_byte + int(frame) * trial.frame_size + within_frame)
            blob = handle.read(record_size)
            if len(blob) < record_size:
                raise B3dLayoutError(
                    f"trial {trial.index} frame {frame}: record truncated"
                )
            record = message()
            record.ParseFromString(blob)
            records.append(record)
    return records


def read_pass_frames(
    b3d: B3dFile,
    trial_index: int,
    frame_indices: Iterable[int],
    *,
    pass_index: int,
) -> B3dPassFrames:
    """Decode selected frames of one processing pass of one trial.

    ``pass_index`` is that trial's own index of the pass (``B3dTrial.pass_index(name)`` finds
    it by name); which pass a conversion wants is the caller's decision, not this reader's.
    """
    wanted = _selected_frames(b3d, trial_index, frame_indices)
    trial = b3d.trials[trial_index]
    if not 0 <= pass_index < trial.pass_count:
        raise B3dLayoutError(
            f"trial {trial_index} has {trial.pass_count} passes; asked for {pass_index}"
        )
    record_size = b3d.header.processing_pass_frame_size
    records = _read_records(
        b3d, trial, wanted,
        b3d.header.raw_sensor_frame_size + pass_index * record_size,
        record_size, _protobuf.SubjectOnDiskProcessingPassFrame,
    )
    fields: dict[str, np.ndarray] = {}
    for name in _PASS_FLAT_FIELDS:
        fields[name] = _stack([np.asarray(getattr(r, name), dtype=np.float64) for r in records])
    for name, width in _PASS_VECTOR_FIELDS.items():
        fields[name] = _stack([_as_vectors(getattr(r, name), width, name) for r in records])
    return B3dPassFrames(
        trial_index=trial_index,
        pass_index=pass_index,
        pass_name=trial.pass_names[pass_index],
        frame_indices=wanted,
        timestamps_s=wanted.astype(np.float64) * trial.timestep_s,
        source_rate_hz=trial.source_rate_hz,
        **fields,
    )


def read_sensor_frames(
    b3d: B3dFile, trial_index: int, frame_indices: Iterable[int]
) -> B3dSensorFrames:
    """Decode the raw sensor record of selected frames of one trial."""
    wanted = _selected_frames(b3d, trial_index, frame_indices)
    trial = b3d.trials[trial_index]
    records = _read_records(
        b3d, trial, wanted, 0, b3d.header.raw_sensor_frame_size,
        _protobuf.SubjectOnDiskSensorFrame,
    )
    fields: dict[str, np.ndarray] = {}
    for name in _SENSOR_FLAT_FIELDS:
        fields[name] = _stack([np.asarray(getattr(r, name), dtype=np.float64) for r in records])
    for name, width in _SENSOR_VECTOR_FIELDS.items():
        fields[name] = _stack([_as_vectors(getattr(r, name), width, name) for r in records])
    return B3dSensorFrames(
        trial_index=trial_index,
        frame_indices=wanted,
        timestamps_s=wanted.astype(np.float64) * trial.timestep_s,
        **fields,
    )
