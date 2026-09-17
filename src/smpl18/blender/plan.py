"""The scene plan: everything Blender needs about one trial, in one npz.

The plan exists so that none of the arithmetic happens inside Blender. It carries, per frame, the
rest-to-posed transform of each joint (:func:`smpl18.mesh.lbs_transforms`) and, once, the rest
surface, its triangles and its skinning weights. Blender puts those transforms on the armature's
bones and lets its own skinning run; because a bone's rest matrix cancels in Blender's deformation
(``pose.matrix @ bone.matrix_local^-1``), the result does not depend on how the bones are drawn or
on any convention this package would otherwise have to guess at.

The plan also carries a few frames of vertices computed here, with their tolerance, so the script
running inside Blender can check its own output against this package's arithmetic and say so
(:mod:`smpl18.blender.scene`). A convention error therefore fails loudly in the render rather than
quietly in the picture.

Bulky arrays are float32, which is what Blender keeps anyway; the transforms stay float64.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from smpl18 import __version__
from smpl18.mesh import lbs_transforms, pose_features, pose_offsets, posed_vertices
from smpl18.skeleton.definition import CHILDREN, JOINT_NAMES, NUM_JOINTS, PARENTS
from smpl18.skeleton.kinematics import fk_batch, rest_joints, shaped_vertices

if TYPE_CHECKING:
    from smpl18.corpus import CorpusTrial
    from smpl18.model.load import Model

__all__ = [
    "PLAN_SCHEMA",
    "ScenePlan",
    "bone_tails",
    "build_plan",
    "plan_for_trial",
    "read_plan",
]

PLAN_SCHEMA = "smpl18_blender_plan_v1"
#: Arrays the reader insists on; the correctives and the samples are optional.
_REQUIRED = ("rest_joints", "bone_tails", "parents", "transforms", "vertices", "faces", "weights")


class PlanError(ValueError):
    """The plan file is not one, or is missing what a scene needs."""


def bone_tails(rest: np.ndarray, *, up_axis: str, leaf_reach: float) -> np.ndarray:
    """``(24, 3)`` tail position per bone: purely how the armature is drawn.

    A bone runs from its joint to its child; a joint with several children follows the one along
    the body's axis (the pelvis up the spine, ``spine3`` up the neck) so the rig reads like a
    skeleton, and a leaf runs on past itself along the bone that arrives at it. None of this
    affects the motion, which the transforms carry.
    """
    rest = np.asarray(rest, dtype=np.float64)
    up = "xyz".index(up_axis)
    tails = np.zeros_like(rest)
    for joint in range(NUM_JOINTS):
        children = CHILDREN[joint]
        if len(children) == 1:
            tails[joint] = rest[children[0]]
        elif children:
            along = [abs(rest[child][up] - rest[joint][up]) for child in children]
            tails[joint] = rest[children[int(np.argmax(along))]]
        else:
            parent = PARENTS[joint]
            reach = rest[joint] - rest[parent] if parent >= 0 else np.eye(3)[up]
            tails[joint] = rest[joint] + leaf_reach * reach
    return tails


@dataclass(frozen=True)
class ScenePlan:
    """One trial as Blender needs it. :meth:`write` stores it; :func:`read_plan` reads it back."""

    #: ``(24, 3)`` rest joint centres at the subject's betas, and the tails the bones are drawn to.
    rest: np.ndarray
    tails: np.ndarray
    #: ``(T, 24, 4, 4)`` rest-to-posed transform per joint and frame.
    transforms: np.ndarray
    #: The rest surface: ``(V, 3)`` vertices, ``(F, 3)`` triangles, ``(V, 24)`` weights.
    vertices: np.ndarray
    faces: np.ndarray
    weights: np.ndarray
    fps: float
    #: ``(V, 3, 207)`` pose blend shape directions and ``(T, 207)`` their per-frame amounts, or
    #: ``None`` when the scene is built without correctives.
    posedirs: np.ndarray | None = None
    pose_amounts: np.ndarray | None = None
    #: Frames whose vertices are stored for the scene to check itself against, and the metres it
    #: may differ by.
    sample_frames: np.ndarray | None = None
    sample_vertices: np.ndarray | None = None
    tolerance: float = 0.0
    #: Where this came from and what it is; written into the npz as JSON.
    about: dict[str, Any] = field(default_factory=dict)

    @property
    def frames(self) -> int:
        return int(self.transforms.shape[0])

    @property
    def num_vertices(self) -> int:
        return int(self.vertices.shape[0])

    @property
    def has_correctives(self) -> bool:
        return self.posedirs is not None and self.pose_amounts is not None

    def arrays(self) -> dict[str, np.ndarray]:
        """The npz contents, bulky parts in float32."""
        arrays: dict[str, np.ndarray] = {
            "schema": np.array(PLAN_SCHEMA),
            "about": np.array(json.dumps(self.about, ensure_ascii=False, sort_keys=True)),
            "joint_names": np.array(JOINT_NAMES),
            "parents": np.array(PARENTS, dtype=np.int64),
            "rest_joints": np.asarray(self.rest, dtype=np.float64),
            "bone_tails": np.asarray(self.tails, dtype=np.float64),
            "transforms": np.asarray(self.transforms, dtype=np.float64),
            "vertices": np.asarray(self.vertices, dtype=np.float32),
            "faces": np.asarray(self.faces, dtype=np.int64),
            "weights": np.asarray(self.weights, dtype=np.float32),
            "fps": np.array(float(self.fps)),
            "tolerance": np.array(float(self.tolerance)),
        }
        if self.has_correctives:
            arrays["posedirs"] = np.asarray(self.posedirs, dtype=np.float32)
            arrays["pose_amounts"] = np.asarray(self.pose_amounts, dtype=np.float64)
        if self.sample_frames is not None and self.sample_vertices is not None:
            arrays["sample_frames"] = np.asarray(self.sample_frames, dtype=np.int64)
            arrays["sample_vertices"] = np.asarray(self.sample_vertices, dtype=np.float32)
        return arrays

    def write(self, path: str | Path) -> Path:
        """Write the plan to ``path`` (an npz), creating its directory."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **self.arrays())
        return path


def read_plan(path: str | Path) -> ScenePlan:
    """Read a plan back, checking its schema and the arrays a scene cannot do without."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        schema = str(data["schema"]) if "schema" in data else "(none)"
        if schema != PLAN_SCHEMA:
            raise PlanError(f"{path}: schema is {schema!r}, not {PLAN_SCHEMA!r}")
        missing = [key for key in _REQUIRED if key not in data]
        if missing:
            raise PlanError(f"{path} lacks {missing}")
        optional = {key: data[key] for key in
                    ("posedirs", "pose_amounts", "sample_frames", "sample_vertices")
                    if key in data}
        return ScenePlan(
            rest=data["rest_joints"],
            tails=data["bone_tails"],
            transforms=data["transforms"],
            vertices=data["vertices"],
            faces=data["faces"],
            weights=data["weights"],
            fps=float(data["fps"]),
            tolerance=float(data["tolerance"]) if "tolerance" in data else 0.0,
            about=json.loads(str(data["about"])) if "about" in data else {},
            **optional,
        )


def _sampled(frames: int, wanted: int) -> np.ndarray:
    """Up to ``wanted`` frame numbers spread over ``frames``, always including the first and last."""
    if wanted < 1:
        return np.zeros(0, dtype=np.int64)
    if wanted >= frames:
        return np.arange(frames, dtype=np.int64)
    return np.unique(np.linspace(0, frames - 1, wanted).round().astype(np.int64))


def build_plan(model: Model, *, betas, local: np.ndarray, trans: np.ndarray, fps: float,
               up_axis: str, correctives: bool, sample_frames: int, tolerance: float,
               leaf_reach: float, about: dict[str, Any] | None = None) -> ScenePlan:
    """Build a plan from a model and a trial's 24-joint pose.

    ``local`` is ``(T, 24, 3, 3)`` local rotations and ``trans`` is ``(T, 3)``, as
    :meth:`smpl18.corpus.CorpusTrial.local_rotations_24` and ``trial.trans`` give them.
    ``correctives`` carries SMPL's pose blend shapes into the scene as shape keys; without them the
    surface is the shaped template skinned rigidly, and the plan records in ``about`` how far that
    is from the model's own answer, so a reader can judge whether it matters for the picture.
    """
    local = np.asarray(local, dtype=np.float64)
    trans = np.asarray(trans, dtype=np.float64)
    rest = rest_joints(model, betas)
    positions, world_rotation = fk_batch(rest, local, trans)
    transforms = lbs_transforms(rest, positions, world_rotation)
    surface = shaped_vertices(model, betas)

    samples = _sampled(local.shape[0], sample_frames)
    sample_vertices = None
    offsets_mm = 0.0
    if samples.size:
        posed = posed_vertices(model, betas, local[samples], trans[samples],
                              with_pose_offsets=correctives)
        sample_vertices = posed.vertices
        offsets_mm = float(np.abs(pose_offsets(model, local[samples])).max() * 1000.0)

    record = {
        "schema": PLAN_SCHEMA,
        "written_by": f"smpl18 {__version__}",
        "frames": int(local.shape[0]),
        "fps": float(fps),
        "up_axis": up_axis,
        "vertices": int(surface.shape[0]),
        "correctives": "shape_keys" if correctives else "off",
        "pose_blend_shapes_mm": round(offsets_mm, 3),
        "checked_frames": [int(frame) for frame in samples],
        "check_tolerance_m": float(tolerance),
        **(about or {}),
    }
    return ScenePlan(
        rest=rest,
        tails=bone_tails(rest, up_axis=up_axis, leaf_reach=leaf_reach),
        transforms=transforms,
        vertices=surface,
        faces=model.faces,
        weights=model.weights,
        fps=float(fps),
        posedirs=model.posedirs if correctives else None,
        pose_amounts=pose_features(local) if correctives else None,
        sample_frames=samples,
        sample_vertices=sample_vertices,
        tolerance=float(tolerance),
        about=record,
    )


def plan_for_trial(trial: CorpusTrial, model: Model, *, correctives: bool, sample_frames: int,
                   tolerance: float, leaf_reach: float,
                   frames: slice | None = None) -> ScenePlan:
    """Build a plan for one corpus trial, rebuilding its 24-joint pose first.

    ``frames`` takes a slice of the trial, which is how a long capture is kept to a scene one can
    open. The subject's betas and frozen constants come from its record, so the body in Blender is
    the body the conversion fitted.
    """
    local = trial.local_rotations_24()
    trans = trial.trans
    chosen = frames or slice(None)
    local, trans = local[chosen], trans[chosen]
    if not local.size:
        raise PlanError(f"{trial.subject.id}/{trial.id}: that frame range selects no frame")
    about = {
        "subject": trial.subject.id,
        "trial": trial.id,
        "gender": trial.subject.gender,
        "stand_in_model": bool(model.stand_in),
        "corpus_frames": int(trial.frames),
        "frames_taken": f"{chosen.start or 0}:{chosen.stop if chosen.stop is not None else trial.frames}"
                        f"{'' if chosen.step in (None, 1) else ':' + str(chosen.step)}",
        "model": str(model.path) if model.path else "in memory",
        "model_sha256": model.sha256,
        "joint_provenance": list(trial.joint_provenance),
    }
    return build_plan(model, betas=trial.subject.betas, local=local, trans=trans, fps=trial.fps,
                      up_axis=trial.up_axis, correctives=correctives,
                      sample_frames=sample_frames, tolerance=tolerance, leaf_reach=leaf_reach,
                      about=about)
