"""A stand-in body with the SMPL-24 tree, for trying the pipeline without the licensed models.

The SMPL models cannot be shipped, but every step of a conversion -- shape fit, pose solve,
reduction, corpus -- only needs the rest skeleton ``J(beta) = J_regressor (v_template +
shapedirs beta)``. This module builds a model whose "vertices" are the 24 joints themselves
(``J_regressor`` is the identity) and whose ten shape directions lengthen named groups of bones,
so the examples and tests can run end to end on any machine.

It is not a human shape model and says so: every file :func:`write_models` writes carries a
``stand_in`` entry, :func:`smpl18.model.load` reads it into ``Model.stand_in``, and a corpus made
with it records that. Convert real data with the real models.

The proportions are ordinary adult ones (stature about 1.7 m), in SMPL's rest frame: Y up, the
body facing +Z, the subject's left towards +X, arms out to the sides. The pelvis does not sit at
the origin, as in the real models, so a caller that confuses the pelvis with the model origin
is caught.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from smpl18.skeleton.definition import JOINT_NAMES, NUM_JOINTS, PARENTS

from .load import STAND_IN_KEY, Model
from .select import GENDERS, model_filename

__all__ = ["STAND_IN_KEY", "demo_model", "demo_rest_joints", "write_models"]

#: Offset of each joint from its parent, metres; the pelvis entry is its position.
_OFFSETS: dict[str, tuple[float, float, float]] = {
    "pelvis": (0.0, -0.10, 0.02),
    "left_hip": (0.08, -0.09, -0.01), "right_hip": (-0.08, -0.09, -0.01),
    "spine1": (0.0, 0.11, -0.02),
    "left_knee": (0.03, -0.38, 0.0), "right_knee": (-0.03, -0.38, 0.0),
    "spine2": (0.0, 0.13, 0.01),
    "left_ankle": (-0.01, -0.40, -0.04), "right_ankle": (0.01, -0.40, -0.04),
    "spine3": (0.0, 0.06, 0.02),
    "left_foot": (0.02, -0.05, 0.12), "right_foot": (-0.02, -0.05, 0.12),
    "neck": (0.0, 0.21, -0.04),
    "left_collar": (0.08, 0.12, -0.03), "right_collar": (-0.08, 0.12, -0.03),
    "head": (0.0, 0.08, 0.05),
    "left_shoulder": (0.10, 0.02, -0.01), "right_shoulder": (-0.10, 0.02, -0.01),
    "left_elbow": (0.26, -0.01, -0.02), "right_elbow": (-0.26, -0.01, -0.02),
    "left_wrist": (0.25, 0.01, 0.0), "right_wrist": (-0.25, 0.01, 0.0),
    "left_hand": (0.08, -0.01, -0.01), "right_hand": (-0.08, -0.01, -0.01),
}

_LEGS = ("left_knee", "right_knee", "left_ankle", "right_ankle")
_THIGHS = ("left_knee", "right_knee")
_SHANKS = ("left_ankle", "right_ankle")
_ARMS = ("left_elbow", "right_elbow", "left_wrist", "right_wrist")
_UPPER_ARMS = ("left_elbow", "right_elbow")
_FOREARMS = ("left_wrist", "right_wrist")
_TRUNK = ("spine1", "spine2", "spine3", "neck")
_HIPS = ("left_hip", "right_hip")
_SHOULDERS = ("left_collar", "right_collar", "left_shoulder", "right_shoulder")
_FEET = ("left_foot", "right_foot")

#: Shape directions: per unit beta, the named bones (by their distal joint) grow by the factor;
#: ``lateral`` restricts the growth to the X component, which is what widens a girdle.
_DIRECTIONS: tuple[tuple[tuple[str, ...], float, bool], ...] = (
    (tuple(name for name in JOINT_NAMES if name != "pelvis"), 0.04, False),   # overall size
    (_LEGS, 0.05, False),
    (_ARMS, 0.05, False),
    (_TRUNK, 0.05, False),
    (_HIPS, 0.10, True),
    (_SHOULDERS, 0.08, True),
    (_FEET, 0.06, False),
    (("head",), 0.08, False),
    (_THIGHS, 0.04, False),     # thigh against shank, with the next pair
    (_UPPER_ARMS, 0.04, False),
)
_COUNTER: dict[int, tuple[tuple[str, ...], float]] = {
    8: (_SHANKS, -0.04),
    9: (_FOREARMS, -0.04),
}


def demo_rest_joints() -> np.ndarray:
    """``(24, 3)`` rest joint centres of the stand-in at zero shape."""
    rest = np.zeros((NUM_JOINTS, 3))
    for joint, name in enumerate(JOINT_NAMES):
        offset = np.array(_OFFSETS[name])
        rest[joint] = offset if PARENTS[joint] < 0 else rest[PARENTS[joint]] + offset
    return rest


def _shape_directions() -> np.ndarray:
    """``(24, 3, 10)``: how each joint moves per unit of each beta."""
    growth = np.zeros((NUM_JOINTS, 3, len(_DIRECTIONS)))   # per bone, by distal joint
    for column, (bones, factor, lateral) in enumerate(_DIRECTIONS):
        for name in bones:
            offset = np.array(_OFFSETS[name])
            if lateral:
                offset = offset * np.array([1.0, 0.0, 0.0])
            growth[JOINT_NAMES.index(name), :, column] += factor * offset
        if column in _COUNTER:
            bones, factor = _COUNTER[column]
            for name in bones:
                growth[JOINT_NAMES.index(name), :, column] += factor * np.array(_OFFSETS[name])
    directions = np.zeros_like(growth)
    for joint in range(1, NUM_JOINTS):
        directions[joint] = directions[PARENTS[joint]] + growth[joint]
    return directions


def demo_model(gender: str = "neutral") -> Model:
    """The stand-in as a :class:`Model` in memory; ``gender`` is only a label here."""
    rest = demo_rest_joints()
    return Model(
        v_template=rest,
        shapedirs=_shape_directions(),
        J_regressor=np.eye(NUM_JOINTS),
        kintree_parents=np.array(PARENTS, dtype=np.int64),
        gender=gender,
        stand_in=True,
    )


def write_models(directory: str | Path) -> list[Path]:
    """Write the stand-in under the three model-set file names in ``directory``.

    The files are marked as stand-ins; see the module docstring before pointing a real
    conversion at them.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    written = []
    for gender in GENDERS:
        model = demo_model(gender)
        path = directory / model_filename(gender)
        np.savez(
            path,
            v_template=model.v_template,
            shapedirs=model.shapedirs,
            J_regressor=model.J_regressor,
            kintree_parents=model.kintree_parents,
            **{STAND_IN_KEY: np.array("stand-in body for trying the pipeline; not an SMPL model")},
        )
        written.append(path)
    return written
