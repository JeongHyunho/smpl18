"""From source files to a corpus: one reader per source kind, one fit, one reduction, one writer.

Every kind but SMPL parameters becomes :class:`~smpl18.fit.Targets` in the output frame:

========================  ======================================================================
kind                      how
========================  ======================================================================
marker_trajectories       ``.trc`` / ``.c3d`` -> gaps bridged -> marker-set rules -> targets
joint_centres             ``.trc`` / ``.c3d`` / ``.npz`` -> correspondence -> targets
skeleton_motion           ``.osim`` + ``.mot``, or ``.bvh`` -> the skeleton's own forward
                          kinematics -> correspondence -> targets
smpl_parameters           ``.npz`` -> the 22 body joints, hands at identity (nothing to fit)
========================  ======================================================================

A subject's trials are then fitted together (:func:`fit_subject`): the betas on their pooled
frames, a pose per trial, and the four frozen-joint constants of the 18-joint reduction on the
pooled poses. :func:`write_subject_corpus` writes the result in the corpus format.

Nothing here knows a dataset. The tables (marker set, correspondence) and the settings file say
everything that differs between sources.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from smpl18.corpus import converter_record, write_subject, write_summary, write_trial
from smpl18.fit.correspondence import Correspondence
from smpl18.fit.pose import (
    PoseFit,
    PoseSettings,
    calibrate_orientations,
    hold_nearest,
    solve_pose,
)
from smpl18.fit.shape import ShapeFit, fit_shape
from smpl18.fit.targets import OrientationTargets, PositionTargets, Targets
from smpl18.formats import c3d as c3d_format
from smpl18.formats import npz as npz_format
from smpl18.formats import trc as trc_format
from smpl18.model.load import Model
from smpl18.reduce import apply as apply_reduction
from smpl18.reduce import fit_constants, per_joint_rms, sample_indices, to_18
from smpl18.skeleton.definition import (
    ABSORBERS,
    BODY_JOINTS,
    FROZEN_JOINT_NAMES,
    FROZEN_JOINTS,
    JOINT_NAMES,
    NUM_JOINTS,
)
from smpl18.skeleton.frames import FrameChange, frame_change, transform_for_gravity
from smpl18.skeleton.kinematics import fk_batch, rest_joints
from smpl18.skeleton.rotations import axis_angle_to_matrix, matrix_to_axis_angle
from smpl18.sources.base import Provenance
from smpl18.sources.markers import MarkerSet, fill_gaps, length_scale
from smpl18.sources.subject import SubjectInfo

__all__ = [
    "ConversionError",
    "SubjectFit",
    "TrialInput",
    "bvh_trial",
    "centre_trial",
    "check_subject_free",
    "validate_settings",
    "fit_subject",
    "load_settings",
    "marker_trial",
    "opensim_trial",
    "parameter_trial",
    "write_subject_corpus",
]


class ConversionError(ValueError):
    pass


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _files(*paths: Path) -> list[dict[str, str]]:
    return [{"path": Path(p).name, "sha256": _sha256(p)} for p in paths]


def load_settings(paths: Sequence[str | Path]) -> tuple[dict[str, Any], list[dict[str, str]]]:
    """Merge settings files in order (later sections update earlier ones) and hash them."""
    if not paths:
        raise ConversionError("no settings file given")
    merged: dict[str, Any] = {}
    record = []
    for path in paths:
        path = Path(path)
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(data, Mapping) or data.get("schema") != "smpl18_settings_v1":
            raise ConversionError(f"{path} is not an smpl18_settings_v1 file")
        for key, value in data.items():
            if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
                merged[key] = {**merged[key], **value}
            else:
                merged[key] = value
        record.append({"path": path.name, "sha256": _sha256(path)})
    return merged, record


def validate_settings(settings: Mapping[str, Any], kind: str) -> None:
    """Check, before anything is read or fitted, every settings value a conversion of ``kind``
    will need, so that a run cannot fail on a missing number after doing its work."""
    from smpl18.reduce import FitSettings

    _output_up(settings)
    FitSettings.from_settings(settings)
    _check_settings(settings)
    if kind != "smpl_parameters":
        from smpl18.fit.shape import ShapeSettings

        ShapeSettings.from_settings(settings)
        PoseSettings.from_settings(settings)
    if kind == "marker_trajectories":
        _setting(settings, "markers", "max_gap_frames")


def _output_up(settings: Mapping[str, Any]) -> str:
    try:
        return str(settings["output"]["up_axis"])
    except (KeyError, TypeError):
        raise ConversionError("settings is missing output.up_axis") from None


def _setting(settings: Mapping[str, Any], section: str, key: str):
    try:
        return settings[section][key]
    except (KeyError, TypeError):
        raise ConversionError(f"settings is missing {section}.{key}") from None


@dataclass
class TrialInput:
    """One trial, read and turned into what the fit needs."""

    id: str
    kind: str
    format: str
    fps: float
    #: Position and orientation targets in the output frame (every kind but SMPL parameters).
    targets: Targets | None = None
    #: SMPL parameters: ``(T, 24, 3, 3)`` local rotations and ``(T, 3)`` translation already in
    #: the output frame, the source's betas, and per-joint provenance.
    local: np.ndarray | None = None
    trans: np.ndarray | None = None
    betas: np.ndarray | None = None
    provenance: tuple[str, ...] | None = None
    #: SMPL parameters: frames whose pose and translation were finite.
    frame_valid: np.ndarray | None = None
    #: What reading the trial found doubtful; recorded with the plausibility checks.
    warnings: list[str] = field(default_factory=list)
    #: Manifest blocks.
    source: dict[str, Any] = field(default_factory=dict)
    tables: dict[str, Any] = field(default_factory=dict)
    repairs: dict[str, Any] = field(default_factory=dict)

    @property
    def frames(self) -> int:
        return self.targets.frames if self.targets is not None else int(self.local.shape[0])


def _frame(source_up: str | None, output_up: str, what: str) -> FrameChange:
    if source_up is None:
        raise ConversionError(f"{what}: the source's up axis must be given (--up-axis)")
    return frame_change(source_up, output_up)


# --- markers ----------------------------------------------------------------------------------------


def _read_markers(path: Path):
    suffix = path.suffix.lower()
    if suffix == ".trc":
        return trc_format.read(path), "trc"
    if suffix == ".c3d":
        return c3d_format.read(path), "c3d"
    raise ConversionError(f"{path.name}: marker files are .trc or .c3d")


def marker_trial(path: str | Path, *, markerset: MarkerSet, subject: SubjectInfo,
                 up_axis: str | None, settings: Mapping[str, Any],
                 occlusion_sentinel: str = "none") -> TrialInput:
    """Labelled markers -> marker-set rules -> targets.

    ``occlusion_sentinel`` ``zero`` treats an exact ``(0, 0, 0)`` sample as a lost marker, for
    writers that park lost markers at the origin.
    """
    path = Path(path)
    table, file_format = _read_markers(path)
    scale = length_scale(table.units)
    positions = table.positions * scale
    valid = table.valid.copy()
    if occlusion_sentinel == "zero":
        valid &= ~np.all(positions == 0.0, axis=2)
    elif occlusion_sentinel != "none":
        raise ConversionError(f"occlusion sentinel must be none or zero, got {occlusion_sentinel!r}")
    change = _frame(up_axis, _output_up(settings), path.name)
    positions = change.apply(np.where(valid[..., None], positions, np.nan))
    max_gap = int(_setting(settings, "markers", "max_gap_frames"))
    before = int((~valid).sum())
    positions, valid = fill_gaps(positions, valid, max_gap)
    bridged = before - int((~valid).sum())
    targets, result = markerset.targets(table.labels, positions, valid, subject.measurements,
                                        fps=table.data_rate_hz)
    return TrialInput(
        id=path.stem,
        kind="marker_trajectories",
        format=file_format,
        fps=float(table.data_rate_hz),
        targets=targets,
        source={
            "files": _files(path),
            "native_fps": float(table.data_rate_hz),
            "native_up_axis": up_axis,
            "native_units": table.units,
            "frame_change": change.record(),
        },
        tables={
            "markerset": {"id": markerset.id, "sha256": markerset.sha256,
                          "path": None if markerset.path is None else markerset.path.name},
            "measured_lengths_m": result.lengths,
            "subject_measurements_m": dict(subject.measurements),
        },
        repairs={"gap_fill": {"max_frames": max_gap, "samples_bridged": bridged},
                 "occlusion_sentinel": occlusion_sentinel},
        warnings=list(result.warnings),
    )


# --- joint centres -------------------------------------------------------------------------------------


def _read_centres(path: Path):
    suffix = path.suffix.lower()
    if suffix in (".trc", ".c3d"):
        table, file_format = _read_markers(path)
        scale = length_scale(table.units)
        return (table.labels, table.positions * scale, table.valid.copy(),
                float(table.data_rate_hz), None, file_format, table.units)
    if suffix == ".npz":
        data = npz_format.read(path)
        for key in ("names", "positions", "fps", "units"):
            if key not in data:
                raise ConversionError(
                    f"{path.name}: a joint-centre npz holds names, positions, fps and units"
                )
        names = tuple(str(n) for n in np.asarray(data["names"]).ravel())
        scale = length_scale(str(np.asarray(data["units"])))
        positions = np.asarray(data["positions"], dtype=np.float64) * scale
        rotations = None
        if "segment_names" in data:
            segments = [str(n) for n in np.asarray(data["segment_names"]).ravel()]
            stack = np.asarray(data["segment_rotations"], dtype=np.float64)
            rotations = {name: stack[:, i] for i, name in enumerate(segments)}
        return (names, positions, np.isfinite(positions).all(axis=2),
                float(np.asarray(data["fps"])), rotations, "npz", str(np.asarray(data["units"])))
    raise ConversionError(f"{path.name}: joint-centre files are .trc, .c3d or .npz")


def centre_trial(path: str | Path, *, correspondence: Correspondence, up_axis: str | None,
                 settings: Mapping[str, Any]) -> TrialInput:
    """Joint-centre trajectories (optionally with segment rotations) -> correspondence -> targets."""
    path = Path(path)
    names, positions, valid, fps, rotations, file_format, units = _read_centres(path)
    change = _frame(up_axis, _output_up(settings), path.name)
    positions = change.apply(np.where(valid[..., None], positions, np.nan))
    if rotations is not None:
        rotations = {name: change.rotation @ stack for name, stack in rotations.items()}
        rotations_valid = {name: np.isfinite(stack).all(axis=(1, 2))
                           for name, stack in rotations.items()}
    else:
        rotations_valid = None
    targets, use = correspondence.centre_targets(
        names, positions, valid, fps=fps, rotations=rotations, rotations_valid=rotations_valid
    )
    return TrialInput(
        id=path.stem, kind="joint_centres", format=file_format, fps=fps, targets=targets,
        source={"files": _files(path), "native_fps": fps, "native_up_axis": up_axis,
                "native_units": units, "frame_change": change.record()},
        tables={"correspondence": correspondence.record(), "use": use.record()},
    )


# --- skeletons --------------------------------------------------------------------------------------------


def opensim_trial(osim_path: str | Path, mot_path: str | Path, *, correspondence: Correspondence,
                  settings: Mapping[str, Any], up_axis: str | None = None,
                  angle_unit: str | None = None) -> TrialInput:
    """An OpenSim model and a coordinate file -> forward kinematics -> targets.

    The world frame comes from the gravity the model declares; ``up_axis`` is only needed for a
    model that declares none (or a zero vector). ``angle_unit`` is only needed when the file does
    not say.
    """
    from smpl18.formats import mot as mot_format
    from smpl18.sources.skeleton import opensim

    osim_path, mot_path = Path(osim_path), Path(mot_path)
    skeleton = opensim.read(osim_path)
    table = mot_format.read(mot_path)
    coordinates, times = opensim.coordinates_from_mot(skeleton, table, angle_unit=angle_unit)
    output = _output_up(settings)
    gravity = skeleton.gravity
    if gravity is not None and np.linalg.norm(gravity) > 0:
        change = transform_for_gravity(gravity, target_up=output)
        native_up = None
    else:
        change = _frame(up_axis, output, f"{osim_path.name} declares no gravity, so")
        native_up = up_axis
    if times.size < 2:
        raise ConversionError(f"{mot_path.name}: fewer than two frames")
    steps = np.diff(times)
    if not np.all(steps > 0):
        raise ConversionError(f"{mot_path.name}: the time column does not increase")
    fps = float(1.0 / np.median(steps))
    uneven = bool(np.abs(steps - np.median(steps)).max() > 1e-6 * np.median(steps) + 1e-9)
    placement = skeleton.motion(coordinates)
    targets, use = correspondence.skeleton_targets(skeleton, placement, frame=change, fps=fps)
    return TrialInput(
        id=mot_path.stem, kind="skeleton_motion", format="osim_mot", fps=fps, targets=targets,
        source={"files": _files(osim_path, mot_path), "native_fps": fps,
                "native_up_axis": native_up, "model_gravity": gravity,
                "uneven_time_steps": uneven, "frame_change": change.record()},
        tables={"correspondence": correspondence.record(), "use": use.record()},
    )


def bvh_trial(path: str | Path, *, correspondence: Correspondence, up_axis: str | None,
              length_unit: str, settings: Mapping[str, Any]) -> TrialInput:
    """A BVH clip -> forward kinematics -> targets. ``length_unit`` is the file's (m, cm, mm)."""
    from smpl18.formats import bvh as bvh_format
    from smpl18.sources.skeleton import bvh

    path = Path(path)
    clip = bvh_format.read(path)
    if not clip.frame_time_s > 0:
        raise ConversionError(f"{path.name}: Frame Time is {clip.frame_time_s}, not a period")
    skeleton = bvh.BvhSkeleton(clip, length_scale=length_scale(length_unit))
    coordinates = bvh.coordinates_from_frames(skeleton, clip)
    change = _frame(up_axis, _output_up(settings), path.name)
    fps = 1.0 / clip.frame_time_s
    placement = skeleton.motion(coordinates)
    targets, use = correspondence.skeleton_targets(skeleton, placement, frame=change, fps=fps)
    return TrialInput(
        id=path.stem, kind="skeleton_motion", format="bvh", fps=fps, targets=targets,
        source={"files": _files(path), "native_fps": fps, "native_up_axis": up_axis,
                "native_units": length_unit, "frame_change": change.record()},
        tables={"correspondence": correspondence.record(), "use": use.record()},
    )


# --- SMPL parameters ------------------------------------------------------------------------------------

#: Keys a parameter npz commonly stores its frame rate under, tried in order.
FPS_KEYS = ("mocap_framerate", "mocap_frame_rate", "fps", "frame_rate")
#: Model family by the number of joints a pose array carries. SMPL-H shares SMPL's body and
#: shape space for the body joints; SMPL-X does not, and is refused.
FAMILIES = {22: "smpl", 24: "smpl", 52: "smplh"}
SMPLX_JOINTS = 55


def parameter_trial(path: str | Path, *, model: Model, up_axis: str | None,
                    settings: Mapping[str, Any], fps: float | None = None,
                    poses_key: str = "poses", trans_key: str = "trans",
                    betas_key: str = "betas") -> TrialInput:
    """SMPL / SMPL-H parameters: the 22 body joints kept, the hands at identity.

    SMPL-X parameters are refused: its betas describe another template, so they cannot be
    carried onto SMPL, and its pelvis sits elsewhere. Convert an SMPL-X sequence's joint
    positions with :func:`centre_trial` instead. A frame with a non-finite pose or translation
    is marked invalid and holds its nearest valid frame.
    """
    path = Path(path)
    data = npz_format.read(path)
    for key in (poses_key, trans_key, betas_key):
        if key not in data:
            raise ConversionError(f"{path.name} has no {key!r} array")
    poses = np.asarray(data[poses_key], dtype=np.float64)
    if poses.ndim == 2:
        if poses.shape[1] % 3:
            raise ConversionError(f"{path.name}: {poses_key} is {poses.shape}, not [T, 3J]")
        poses = poses.reshape(poses.shape[0], -1, 3)
    if poses.ndim != 3:
        raise ConversionError(f"{path.name}: {poses_key} must be [T, J, 3] or [T, 3J]")
    joints = poses.shape[1]
    if joints == SMPLX_JOINTS:
        raise ConversionError(
            f"{path.name} holds SMPL-X parameters ({joints} joints). Their betas and rest pelvis "
            "belong to the SMPL-X template, not SMPL's; compute the sequence's joint positions "
            "with the SMPL-X model and convert them with `smpl18 convert centres`"
        )
    if joints not in FAMILIES:
        raise ConversionError(
            f"{path.name}: {joints} joints per frame is not SMPL (22 or 24) or SMPL-H (52)"
        )
    betas = np.asarray(data[betas_key], dtype=np.float64)
    betas = betas[0] if betas.ndim == 2 else betas.ravel()
    stored_betas = int(betas.size)
    width = model.num_betas
    betas = np.pad(betas[:width], (0, max(0, width - betas.size)))
    trans = np.asarray(data[trans_key], dtype=np.float64)
    if trans.shape != (poses.shape[0], 3):
        raise ConversionError(f"{path.name}: {trans_key} must be [T, 3]")
    if fps is None:
        for key in FPS_KEYS:
            if key in data:
                fps = float(np.asarray(data[key]).ravel()[0])
                break
        else:
            raise ConversionError(f"{path.name} stores no frame rate; pass --fps")
    if not fps > 0:
        raise ConversionError(f"{path.name}: frame rate {fps} is not positive")
    body = list(BODY_JOINTS)
    valid = np.isfinite(poses[:, body]).all(axis=(1, 2)) & np.isfinite(trans).all(axis=1)
    if not valid.any():
        raise ConversionError(f"{path.name}: no frame has a finite pose and translation")
    if not valid.all():
        poses = hold_nearest(np.where(valid[:, None, None], poses, 0.0), valid)
        trans = hold_nearest(np.where(valid[:, None], trans, 0.0), valid)
    frames = poses.shape[0]
    local = np.tile(np.eye(3), (frames, NUM_JOINTS, 1, 1))
    local[:, body] = axis_angle_to_matrix(poses[:, body])
    change = _frame(up_axis, _output_up(settings), path.name)
    pelvis = rest_joints(model, betas)[0]
    local, trans = change.apply_to_pose(local, trans, pelvis_rest=pelvis)
    provenance = tuple(
        Provenance.MEASURED.value if joint in BODY_JOINTS else Provenance.ABSENT.value
        for joint in range(NUM_JOINTS)
    )
    return TrialInput(
        id=path.stem, kind="smpl_parameters", format="npz", fps=fps, local=local, trans=trans,
        betas=betas, provenance=provenance, frame_valid=valid,
        source={"files": _files(path), "native_fps": fps, "native_up_axis": up_axis,
                "model_family": FAMILIES[joints], "joints_stored": joints,
                "betas_stored": stored_betas, "frame_change": change.record(),
                "frames_nonfinite": int((~valid).sum())},
        repairs={"nonfinite_frames_held": int((~valid).sum())},
    )


# --- fitting a subject ------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SubjectFit:
    betas: np.ndarray
    rest: np.ndarray
    shape: ShapeFit | None
    #: Per trial id: the 24-joint solve (None for SMPL parameters).
    poses: dict[str, PoseFit | None]
    #: Per trial id: ``(T, 24, 3, 3)`` local rotations and ``(T, 3)`` translation before the
    #: reduction, and the frames that are valid.
    local: dict[str, np.ndarray]
    trans: dict[str, np.ndarray]
    frame_valid: dict[str, np.ndarray]
    constants: Any
    per_joint_rms_m: dict[str, float]
    #: Orientation-target constants shared by the subject's trials, by target label.
    calibration: dict[str, np.ndarray]
    #: For SMPL parameters: the largest difference between the trials' betas.
    betas_spread: float | None


def _aligned(trials: Sequence[TrialInput]) -> Targets:
    """The trials' targets end to end, their columns united by label."""
    position_labels: list[str] = []
    position_joint: dict[str, int] = {}
    position_weight: dict[str, float] = {}
    orientation_labels: list[str] = []
    orientation_joint: dict[str, int] = {}
    orientation_weight: dict[str, float] = {}
    for trial in trials:
        columns = trial.targets.positions
        for label, joint, weight in zip(columns.labels, columns.joints, columns.weights):
            if label not in position_joint:
                position_labels.append(label)
                position_joint[label], position_weight[label] = joint, weight
        if trial.targets.orientations is not None:
            columns = trial.targets.orientations
            for label, joint, weight in zip(columns.labels, columns.joints, columns.weights):
                if label not in orientation_joint:
                    orientation_labels.append(label)
                    orientation_joint[label], orientation_weight[label] = joint, weight
    frames = sum(t.frames for t in trials)
    positions = np.full((frames, len(position_labels), 3), np.nan)
    valid = np.zeros((frames, len(position_labels)), bool)
    rotations = np.tile(np.eye(3), (frames, len(orientation_labels), 1, 1))
    rotations_valid = np.zeros((frames, len(orientation_labels)), bool)
    start = 0
    for trial in trials:
        stop = start + trial.frames
        columns = trial.targets.positions
        for column, label in enumerate(columns.labels):
            slot = position_labels.index(label)
            positions[start:stop, slot] = columns.positions[:, column]
            valid[start:stop, slot] = columns.valid[:, column]
        if trial.targets.orientations is not None:
            columns = trial.targets.orientations
            for column, label in enumerate(columns.labels):
                slot = orientation_labels.index(label)
                rotations[start:stop, slot] = columns.rotations[:, column]
                rotations_valid[start:stop, slot] = columns.valid[:, column]
        start = stop
    orientations = None
    if orientation_labels:
        orientations = OrientationTargets(
            [orientation_joint[label] for label in orientation_labels], rotations,
            rotations_valid, [orientation_weight[label] for label in orientation_labels],
            orientation_labels,
        )
    provenance = _merged_provenance([t.targets.provenance for t in trials])
    return Targets(
        PositionTargets([position_joint[label] for label in position_labels], positions, valid,
                        [position_weight[label] for label in position_labels], position_labels),
        orientations, trials[0].fps, provenance,
    )


def _merged_provenance(codes: Sequence[Sequence[str]]) -> tuple[str, ...]:
    """Per joint, the strongest code any trial gives (measured > derived > absent)."""
    rank = {Provenance.ABSENT.value: 0, Provenance.DERIVED.value: 1, Provenance.MEASURED.value: 2}
    return tuple(max((c[j] for c in codes), key=rank.__getitem__) for j in range(NUM_JOINTS))


def _shared_calibration(rest, pooled: Targets, first: Sequence[PoseFit]) -> dict[str, np.ndarray]:
    """One constant per orientation target for the whole subject, from its positions-only solves."""
    orientations = pooled.orientations
    solved = np.concatenate([fit.frame_valid for fit in first])
    usable = OrientationTargets(orientations.joints, orientations.rotations,
                                orientations.valid & solved[:, None], orientations.weights,
                                orientations.labels)
    constants = calibrate_orientations(
        rest, np.concatenate([fit.local for fit in first]),
        np.concatenate([fit.trans for fit in first]), usable,
    )
    return dict(zip(orientations.labels, constants))


def fit_subject(model: Model, trials: Sequence[TrialInput], settings: Mapping[str, Any],
                *, progress=None) -> SubjectFit:
    """Shape on the pooled frames, a pose per trial, then the reduction constants.

    The orientation constants are calibrated once on all of the subject's trials, so a segment
    seen only through its own frame keeps one neutral twist across them.
    """
    if not trials:
        raise ConversionError("no trials to fit")
    kinds = {t.kind for t in trials}
    if len(kinds) != 1:
        raise ConversionError(f"one subject's trials must share a source kind, got {sorted(kinds)}")
    say = progress or (lambda message: None)
    poses: dict[str, PoseFit | None] = {}
    local: dict[str, np.ndarray] = {}
    trans: dict[str, np.ndarray] = {}
    valid: dict[str, np.ndarray] = {}
    shape = None
    calibration: dict[str, np.ndarray] = {}
    spread = None
    if kinds == {"smpl_parameters"}:
        betas = trials[0].betas
        spread = float(max(np.abs(t.betas - betas).max() for t in trials))
        rest = rest_joints(model, betas)
        for trial in trials:
            poses[trial.id] = None
            local[trial.id], trans[trial.id] = trial.local, trial.trans
            valid[trial.id] = trial.frame_valid
    else:
        pooled = _aligned(trials)
        say(f"shape: fitting betas on {pooled.frames} frames")
        shape = fit_shape(model, pooled, settings)
        betas = shape.betas
        rest = rest_joints(model, betas)
        first = []
        for trial in trials:
            say(f"pose: {trial.id} ({trial.frames} frames), positions")
            first.append(solve_pose(rest, trial.targets, settings, positions_only=True))
        use_orientations = (pooled.orientations is not None
                            and PoseSettings.from_settings(settings).orientation_weight > 0)
        if use_orientations:
            say("pose: calibrating segment frames on all trials")
            calibration = _shared_calibration(rest, pooled, first)
        for trial, start in zip(trials, first):
            constants = None
            if use_orientations and trial.targets.orientations is not None:
                constants = np.stack([calibration[label]
                                      for label in trial.targets.orientations.labels])
                say(f"pose: {trial.id}, orientations")
            fit = solve_pose(rest, trial.targets, settings, calibration=constants, start=start)
            poses[trial.id] = fit
            local[trial.id], trans[trial.id], valid[trial.id] = fit.local, fit.trans, fit.frame_valid
    say("reduce: fitting the four frozen-joint constants")
    pooled_local = np.concatenate([local[t.id][valid[t.id]] for t in trials])
    if pooled_local.shape[0] == 0:
        raise ConversionError("no valid frame to fit the reduction on")
    constants = fit_constants(pooled_local, rest, settings=settings)
    sample = pooled_local[sample_indices(pooled_local.shape[0],
                                         int(settings["reduce"]["sample_frames"]))]
    return SubjectFit(
        betas=betas, rest=rest, shape=shape, poses=poses, local=local, trans=trans,
        frame_valid=valid, constants=constants,
        per_joint_rms_m=per_joint_rms(sample, rest, constants.constants),
        calibration=calibration, betas_spread=spread,
    )


# --- checking -------------------------------------------------------------------------------------------

CHECKS_SECTION = "checks"
_CHECK_NAMES = ("bone_length_range_m", "max_position_rms_m", "upright_min_cosine",
                "upright_min_fraction")


def _check_settings(settings: Mapping[str, Any]) -> dict[str, Any]:
    section = settings.get(CHECKS_SECTION) if isinstance(settings, Mapping) else None
    if not isinstance(section, Mapping):
        raise ConversionError(f"settings has no {CHECKS_SECTION!r} section")
    missing = [name for name in _CHECK_NAMES if name not in section]
    if missing:
        raise ConversionError(
            "settings is missing " + ", ".join(f"{CHECKS_SECTION}.{n}" for n in missing)
        )
    low, high = (float(v) for v in section["bone_length_range_m"])
    return {"bones": (low, high), "rms": float(section["max_position_rms_m"]),
            "cosine": float(section["upright_min_cosine"]),
            "fraction": float(section["upright_min_fraction"])}


def _subject_checks(fit: SubjectFit, limits) -> list[str]:
    """Warnings a wrong length unit gives away: bones no human has."""
    if fit.shape is None:
        return []
    low, high = limits["bones"]
    odd = {name: entry["measured_m"] for name, entry in fit.shape.bones.items()
           if not low <= entry["measured_m"] <= high}
    if not odd:
        return []
    listed = ", ".join(f"{name} {length:.3g} m" for name, length in sorted(odd.items()))
    return [f"bone lengths outside {low:g}-{high:g} m ({listed}): check the length unit"]


def _trial_checks(rest, stored, trans, valid, up_axis: str, stored_error, limits) -> list[str]:
    """Warnings a wrong up axis or a wrong table gives away."""
    warnings = []
    rms = stored_error.get("rms_m") if stored_error else None
    if rms is not None and rms > limits["rms"]:
        warnings.append(
            f"the stored pose is {rms * 1000:.0f} mm RMS from the source's joint centres: check "
            "the length unit, the table and the subject's measurements"
        )
    positions, _ = fk_batch(rest, stored[valid], trans[valid])
    trunk = positions[:, JOINT_NAMES.index("neck")] - positions[:, JOINT_NAMES.index("pelvis")]
    up = np.eye(3)["xyz".index(up_axis)]
    cosine = trunk @ up / np.maximum(np.linalg.norm(trunk, axis=1), 1e-12)
    upright = float(np.mean(cosine >= limits["cosine"])) if cosine.size else 1.0
    if upright < limits["fraction"]:
        warnings.append(
            f"the trunk points along the {up_axis} axis in only {upright:.0%} of the frames: check "
            "--up-axis (a trial spent lying or bending over is flagged too)"
        )
    return warnings


# --- writing ----------------------------------------------------------------------------------------------


def _target_error(rest, local, trans, targets: Targets, frames) -> dict[str, Any]:
    """RMS and maximum distance of the solved pose from its position targets."""
    positions, _ = fk_batch(rest, local[frames], trans[frames])
    columns = targets.positions
    distance = np.linalg.norm(positions[:, columns.joints] - columns.positions[frames], axis=2)
    ok = columns.valid[frames]
    values = distance[ok]
    per_target = {}
    for column, label in enumerate(columns.labels):
        column_values = distance[ok[:, column], column]
        per_target[label] = (float(np.sqrt(np.mean(column_values**2)))
                             if column_values.size else None)
    return {
        "rms_m": float(np.sqrt(np.mean(values**2))) if values.size else None,
        "max_m": float(values.max()) if values.size else None,
        "per_target_rms_m": per_target,
    }


_SUBJECT_FILES = ("*.npz", "*.manifest.json", "subject.json")


def check_subject_free(out: str | Path, subject_id: str, *, replace: bool) -> list[str]:
    """Refuse to write over a subject already in the corpus, unless ``replace``.

    A subject's trials share its betas and frozen-joint constants, so trials written by an
    earlier run would no longer match a new record. Returns the files ``replace`` would remove.
    """
    subject_dir = Path(out) / subject_id
    present = sorted(p for pattern in _SUBJECT_FILES for p in subject_dir.glob(pattern))
    if present and not replace:
        trials = sorted(p.stem for p in subject_dir.glob("*.npz"))
        raise ConversionError(
            f"subject {subject_id} is already in {out} (trials: {', '.join(trials) or 'none'}). "
            "Convert all of a subject's trials in one run; pass --replace to convert the subject "
            "anew, which removes its earlier trials"
        )
    return [p.name for p in present]


def write_subject_corpus(out: str | Path, *, subject: SubjectInfo, model: Model,
                         trials: Sequence[TrialInput], fit: SubjectFit,
                         settings: Mapping[str, Any], settings_files: list[dict[str, str]],
                         profile: dict[str, Any] | None = None,
                         replace: bool = False) -> dict[str, Any]:
    """Write one subject's record and trials, then rebuild the corpus summary.

    A subject already in the corpus is refused unless ``replace``, which first removes that
    subject's record and trials (and nothing else).
    """
    out = Path(out)
    subject_dir = out / subject.id
    stale = check_subject_free(out, subject.id, replace=replace)
    validate_settings(settings, trials[0].kind)
    output_up = _output_up(settings)
    limits = _check_settings(settings)
    constants = fit.constants
    settings_hash = hashlib.sha256(
        yaml.safe_dump(dict(settings), sort_keys=True).encode("utf-8")
    ).hexdigest()
    profile = profile or {"id": "adhoc"}
    frozen_provenance = {}
    for joint, name in zip(FROZEN_JOINTS, FROZEN_JOINT_NAMES):
        codes = sorted({(t.targets.provenance if t.targets is not None else t.provenance)[joint]
                        for t in trials})
        frozen_provenance[name] = codes[0] if len(codes) == 1 else codes
    subject_warnings = _subject_checks(fit, limits)

    trial_records = []
    for trial in trials:
        local, trans = fit.local[trial.id], fit.trans[trial.id]
        reduced = apply_reduction(local, constants.constants)
        poses18 = matrix_to_axis_angle(to_18(reduced))
        provenance = trial.targets.provenance if trial.targets is not None else trial.provenance
        valid = fit.frame_valid[trial.id]
        validation: dict[str, Any] = {}
        pose_fit = fit.poses[trial.id]
        if pose_fit is not None:
            frames = np.flatnonzero(valid)
            validation = {
                "solved_24_joint": _target_error(fit.rest, local, trans, trial.targets, frames),
                "stored_18_joint": _target_error(fit.rest, reduced, trans, trial.targets, frames),
                "orientation_rms_rad": (None if pose_fit.orientation_error is None
                                        else pose_fit.orientation_error.rms),
                "passes": list(pose_fit.passes),
            }
        warnings = list(trial.warnings) + _trial_checks(
            fit.rest, reduced, trans, valid, output_up, validation.get("stored_18_joint"), limits
        )
        tables = dict(trial.tables)
        if fit.calibration and trial.targets is not None and trial.targets.orientations is not None:
            tables["orientation_calibration"] = {
                "shared_by_subject_trials": True,
                "rotation_vectors": {
                    label: matrix_to_axis_angle(fit.calibration[label])
                    for label in trial.targets.orientations.labels
                },
            }
        manifest = {
            "subject_id": subject.id,
            "source": {"kind": trial.kind, "format": trial.format, **trial.source},
            "profile": profile,
            "tables": tables,
            "repairs": trial.repairs,
            "discontinuities": [],
            "settings": {key: settings[key] for key in
                         ("output", "shape", "pose", "markers", "reduce", CHECKS_SECTION)
                         if key in settings},
            "settings_files": settings_files,
            "settings_sha256": settings_hash,
            "converter": converter_record(),
            "frames": {"total": int(valid.size), "valid": int(valid.sum())},
            "validation": validation,
            "checks": warnings,
        }
        trial_records.append((trial, poses18, trans, provenance, valid, manifest))
        subject_warnings.extend(f"{trial.id}: {w}" for w in warnings)

    record = {
        "subject_id": subject.id,
        "gender": subject.gender,
        "gender_source": "subject_file" if subject.gender_from == "file" else "user_argument",
        "model_file": None if model.path is None else model.path.name,
        "model_sha256": model.sha256,
        "model_is_stand_in": bool(model.stand_in),
        "betas": fit.betas,
        "subject_file": subject.record(),
        "fit": None if fit.shape is None else {
            "betas_fitted": fit.shape.fitted_betas,
            "bone_lengths": fit.shape.bones,
            "bone_rms_m": fit.shape.bone_rms_m,
            "position_rms_m": fit.shape.position_rms_m,
            "frames_used": fit.shape.frames_used,
            "refinements": fit.shape.refinements,
            "prior_weight": settings["shape"]["prior_weight"],
        },
        "betas_spread_across_trials": fit.betas_spread,
        "reduced_model": {
            "frozen_joints": list(FROZEN_JOINT_NAMES),
            "constants": constants.rotation_vectors,
            "absorbed_by": {JOINT_NAMES[j]: JOINT_NAMES[ABSORBERS[j]] for j in FROZEN_JOINTS},
            "frozen_joint_provenance": frozen_provenance,
            "fit": {
                "residual_rms_m": constants.fitted.rms_m,
                "residual_max_m": constants.fitted.max_m,
                "starting_guess": {"rms_m": constants.initial.rms_m,
                                   "max_m": constants.initial.max_m,
                                   "guess": "mean rotation of each frozen joint"},
                "per_joint_rms_m": fit.per_joint_rms_m,
                "frames_used": constants.frames,
                "solver": settings["reduce"]["optimiser"],
                "converged": constants.success,
            },
        },
        "trials": [t.id for t in trials],
        "checks": subject_warnings,
    }
    # Everything is computed; only now does the earlier conversion of this subject go.
    for name in stale:
        (subject_dir / name).unlink()
    write_subject(subject_dir, record)
    for trial, poses18, trans, provenance, valid, manifest in trial_records:
        write_trial(
            subject_dir, trial.id, poses=poses18, trans=trans, fps=trial.fps, up_axis=output_up,
            joint_provenance=to_18(np.array(provenance)[None])[0], frame_valid=valid,
            manifest=manifest,
        )
    return write_summary(out)
