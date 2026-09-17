"""The SMPL surface: pose blend shapes and linear blend skinning, in numpy.

The skeleton alone (:mod:`smpl18.skeleton.kinematics`) is all a conversion needs, but a renderer
needs the skin. This module adds the two steps SMPL puts between a pose and its vertices:

1. **Pose blend shapes.** ``posedirs`` (207 directions) displace the shaped template by a linear
   function of the 23 non-root local rotations, ``concat_j (R_j - I)``, which is what keeps a bent
   elbow from creasing. A model extracted without ``--with-mesh`` has none and cannot be skinned.
2. **Linear blend skinning.** Each joint carries a rigid transform from the rest pose to the
   posed one, ``A_j = [G_j | p_j - G_j j_j]``, and a vertex moves by the weighted average of the
   transforms of the joints it is bound to, ``v' = (sum_j w_vj A_j) v``.

:func:`lbs_transforms` is the piece a renderer wants on its own: those are the matrices an
armature's bones must end up carrying, whatever a given renderer calls them
(:mod:`smpl18.blender.plan` builds a Blender scene out of them).

Shapes follow the rest of the package: ``T`` frames, ``V`` vertices, 24 joints, and a single
frame is a ``T`` of one. Memory is the caller's to watch -- a full SMPL body is 6890 vertices,
so a thousand frames of posed vertices is about 165 MB; :func:`posed_vertices` takes a frame
slice for that reason.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from smpl18.skeleton.definition import NUM_JOINTS
from smpl18.skeleton.kinematics import fk_batch, rest_joints, shaped_vertices

if TYPE_CHECKING:
    from smpl18.model.load import Model

__all__ = [
    "NUM_POSE_FEATURES",
    "MeshError",
    "PosedSurface",
    "lbs_transforms",
    "pose_features",
    "pose_offsets",
    "posed_vertices",
    "skin",
]

#: ``(24 - 1) * 9``: SMPL's pose blend shape directions, nine per non-root joint.
NUM_POSE_FEATURES = (NUM_JOINTS - 1) * 9


class MeshError(ValueError):
    """The model has no mesh, or an array's shape does not fit the model."""


def _with_mesh(model: Model) -> None:
    if not model.has_mesh:
        missing = [key for key in ("weights", "posedirs", "faces") if getattr(model, key) is None]
        raise MeshError(
            f"this model has no {', '.join(missing)}, so it has no surface to skin; extract it "
            "again with `smpl18 extract-model --with-mesh`"
        )


def pose_features(local: np.ndarray) -> np.ndarray:
    """``(T, 207)``: SMPL's pose feature ``concat_j (R_j - I)`` over joints 1..23.

    ``local`` is ``(T, 24, 3, 3)`` local rotation matrices, as
    :meth:`smpl18.corpus.CorpusTrial.local_rotations_24` gives them.
    """
    local = np.asarray(local, dtype=np.float64)
    if local.ndim != 4 or local.shape[1:] != (NUM_JOINTS, 3, 3):
        raise MeshError(f"expected (T, 24, 3, 3) local rotations, got {local.shape}")
    return (local[:, 1:] - np.eye(3)).reshape(local.shape[0], NUM_POSE_FEATURES)


def pose_offsets(model: Model, local: np.ndarray) -> np.ndarray:
    """``(T, V, 3)`` pose blend shape displacement of the template, from ``posedirs``."""
    _with_mesh(model)
    return np.einsum("vij,tj->tvi", model.posedirs, pose_features(local))


def lbs_transforms(rest: np.ndarray, positions: np.ndarray,
                   world_rotation: np.ndarray) -> np.ndarray:
    """``(T, 24, 4, 4)`` rest-to-posed transform per joint, ``A_j = [G_j | p_j - G_j j_j]``.

    ``rest`` is ``(24, 3)`` rest joint centres, ``positions`` and ``world_rotation`` are what
    :func:`smpl18.skeleton.kinematics.fk_batch` returns for the same pose. A renderer that puts
    these on its bones and skins with the model's ``weights`` reproduces :func:`skin` exactly,
    without having to share any of this package's conventions.
    """
    rest = np.asarray(rest, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.float64)
    world_rotation = np.asarray(world_rotation, dtype=np.float64)
    if rest.shape != (NUM_JOINTS, 3):
        raise MeshError(f"expected (24, 3) rest joints, got {rest.shape}")
    frames = positions.shape[0]
    if positions.shape != (frames, NUM_JOINTS, 3):
        raise MeshError(f"expected (T, 24, 3) joint positions, got {positions.shape}")
    if world_rotation.shape != (frames, NUM_JOINTS, 3, 3):
        raise MeshError(f"expected (T, 24, 3, 3) world rotations, got {world_rotation.shape}")

    transforms = np.zeros((frames, NUM_JOINTS, 4, 4), dtype=np.float64)
    transforms[:, :, :3, :3] = world_rotation
    transforms[:, :, :3, 3] = positions - np.einsum("tjik,jk->tji", world_rotation, rest)
    transforms[:, :, 3, 3] = 1.0
    return transforms


def skin(vertices: np.ndarray, weights: np.ndarray, transforms: np.ndarray) -> np.ndarray:
    """``(T, V, 3)``: each vertex through the weighted average of its joints' transforms.

    ``vertices`` is the ``(V, 3)`` rest surface (shaped, with the frame's pose offsets already
    added when there are any), ``weights`` is ``(V, 24)`` and ``transforms`` is ``(T, 24, 4, 4)``
    from :func:`lbs_transforms`. ``vertices`` may also be ``(T, V, 3)``, one surface per frame,
    which is what pose blend shapes need.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    transforms = np.asarray(transforms, dtype=np.float64)
    frames = transforms.shape[0]
    if vertices.ndim == 2:
        vertices = np.broadcast_to(vertices, (frames, *vertices.shape))
    if vertices.ndim != 3 or vertices.shape[0] != frames or vertices.shape[2] != 3:
        raise MeshError(f"expected (V, 3) or ({frames}, V, 3) vertices, got {vertices.shape}")
    if weights.shape != (vertices.shape[1], NUM_JOINTS):
        raise MeshError(f"expected ({vertices.shape[1]}, 24) weights, got {weights.shape}")

    blended = np.einsum("vj,tjab->tvab", weights, transforms)
    return np.einsum("tvij,tvj->tvi", blended[:, :, :3, :3], vertices) + blended[:, :, :3, 3]


@dataclass(frozen=True)
class PosedSurface:
    """The posed surface of one trial, with what it was built from."""

    #: ``(T, V, 3)`` world vertices.
    vertices: np.ndarray
    #: The model's ``(F, 3)`` triangle list.
    faces: np.ndarray
    #: ``(T, 24, 3)`` world joint centres.
    joints: np.ndarray
    #: ``(T, 24, 4, 4)`` rest-to-posed transforms, from :func:`lbs_transforms`.
    transforms: np.ndarray
    #: ``(24, 3)`` rest joint centres at these betas.
    rest: np.ndarray


def posed_vertices(model: Model, betas, local: np.ndarray, trans: np.ndarray, *,
                   with_pose_offsets: bool = True) -> PosedSurface:
    """Skin ``model`` at ``betas`` through the pose of every frame.

    ``local`` is ``(T, 24, 3, 3)`` local rotations and ``trans`` is ``(T, 3)``. With
    ``with_pose_offsets`` false the pose blend shapes are left out, which is what a renderer that
    cannot carry 207 shape keys has to do; the difference is a few millimetres around bent joints
    and is what :func:`smpl18.blender.plan.build_plan` reports.
    """
    _with_mesh(model)
    local = np.asarray(local, dtype=np.float64)
    rest = rest_joints(model, betas)
    positions, world_rotation = fk_batch(rest, local, trans)
    transforms = lbs_transforms(rest, positions, world_rotation)
    surface = shaped_vertices(model, betas)
    vertices = surface + pose_offsets(model, local) if with_pose_offsets else surface
    return PosedSurface(
        vertices=skin(vertices, model.weights, transforms),
        faces=model.faces,
        joints=positions,
        transforms=transforms,
        rest=rest,
    )
