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

from smpl18.skeleton.definition import CHILDREN, JOINT_NAMES, NUM_JOINTS, PARENTS

from .load import STAND_IN_KEY, Model
from .select import GENDERS, model_filename

__all__ = ["STAND_IN_KEY", "demo_boxes", "demo_model", "demo_rest_joints", "write_models"]

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


# --- a surface to look at -------------------------------------------------------------------------

#: Half-width of a box, as a fraction of the bone it wraps, and the metres it is held between.
_BOX_WIDTH = (0.22, 0.02, 0.06)
#: How far a leaf's box runs past the joint, as a fraction of the bone that arrives at it.
_LEAF_REACH = 0.7
#: Corner signs of a box in (along, side, front) order, the first four at the near end.
_CORNERS: tuple[tuple[int, int, int], ...] = (
    (0, -1, -1), (0, 1, -1), (0, 1, 1), (0, -1, 1),
    (1, -1, -1), (1, 1, -1), (1, 1, 1), (1, -1, 1),
)
#: Triangles of one box, into the eight corners above: near and far caps, then the four sides.
_BOX_FACES: tuple[tuple[int, int, int], ...] = (
    (0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7),
    (0, 1, 5), (0, 5, 4), (1, 2, 6), (1, 6, 5),
    (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7),
)


def _frame_about(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two unit vectors perpendicular to ``axis`` and to each other."""
    other = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    side = np.cross(axis, other)
    side /= np.linalg.norm(side)
    return side, np.cross(axis, side)


def _bones() -> list[tuple[int, dict[int, float]]]:
    """One box per bone, as ``(joint, far end)``: joint to child, and past each leaf.

    A bone's box belongs to the joint whose rotation carries it: the segment from a joint to its
    child turns with that joint, and a leaf's own rotation turns the box beyond it. The far end is
    given as weights over joints, so a box is a linear function of the rest skeleton and the shape
    directions carry onto its corners exactly. The near end is always the joint itself.
    """
    bones = []
    for joint in range(NUM_JOINTS):
        for child in CHILDREN[joint]:
            bones.append((joint, {child: 1.0}))
        if not CHILDREN[joint] and PARENTS[joint] >= 0:
            reach = {joint: 1.0 + _LEAF_REACH, PARENTS[joint]: -_LEAF_REACH}
            bones.append((joint, reach))
    return bones


def _end(rest: np.ndarray, weights: dict[int, float]) -> np.ndarray:
    """Where a box's far end sits, from its weights over the rest joints."""
    return sum(weight * rest[joint] for joint, weight in weights.items())


def demo_boxes(rest: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """A blocky surface for the stand-in: ``(vertices, faces, weights, J_regressor)``.

    One box per bone, each bound rigidly to the joint that turns it, so the body is a mannequin
    of straight blocks rather than a human shape. The regressor reads every joint back exactly:
    a box's near cap is centred on its joint, so the mean of its four corners is that joint.
    """
    rest = np.asarray(rest, dtype=np.float64)
    bones = _bones()
    vertices = np.zeros((8 * len(bones), 3))
    weights = np.zeros((8 * len(bones), NUM_JOINTS))
    faces = np.zeros((len(_BOX_FACES) * len(bones), 3), dtype=np.int64)
    near_corners: dict[int, list[int]] = {}
    for box, (joint, far_end) in enumerate(bones):
        near = rest[joint]
        along = _end(rest, far_end) - near
        length = float(np.linalg.norm(along))
        axis = along / length
        side, front = _frame_about(axis)
        fraction, low, high = _BOX_WIDTH
        width = min(max(fraction * length, low), high)
        for corner, (reach, out, ahead) in enumerate(_CORNERS):
            row = 8 * box + corner
            vertices[row] = near + reach * along + width * (out * side + ahead * front)
            weights[row, joint] = 1.0
            if reach == 0:
                near_corners.setdefault(joint, []).append(row)
        faces[len(_BOX_FACES) * box:len(_BOX_FACES) * (box + 1)] = 8 * box + np.array(_BOX_FACES)

    regressor = np.zeros((NUM_JOINTS, vertices.shape[0]))
    for joint, rows in near_corners.items():
        regressor[joint, rows] = 1.0 / len(rows)
    return vertices, faces, weights, regressor


def _mesh_shape_directions(joint_directions: np.ndarray) -> np.ndarray:
    """Carry the joints' shape directions onto the box corners, ``(V, 3, 10)``.

    Each corner follows the end of the bone it sits on, so the regressed rest joints are exactly
    the joints' own directions and the boxes skew a little instead of being rebuilt. The box widths
    do not follow the shape, which a body of blocks need not do.
    """
    bones = _bones()
    directions = np.zeros((8 * len(bones), 3, joint_directions.shape[2]))
    for box, (joint, far_end) in enumerate(bones):
        far = sum(weight * joint_directions[end] for end, weight in far_end.items())
        for corner, (reach, _out, _ahead) in enumerate(_CORNERS):
            directions[8 * box + corner] = far if reach else joint_directions[joint]
    return directions


def demo_model(gender: str = "neutral", *, with_mesh: bool = False) -> Model:
    """The stand-in as a :class:`Model` in memory; ``gender`` is only a label here.

    With ``with_mesh`` the model also carries the blocky surface of :func:`demo_boxes`, which is
    what a renderer needs; its ``posedirs`` are zero, since a body of rigid blocks has no pose
    blend shapes to correct.
    """
    rest = demo_rest_joints()
    joint_directions = _shape_directions()
    mesh: dict[str, np.ndarray] = {}
    if with_mesh:
        vertices, faces, weights, regressor = demo_boxes(rest)
        mesh = {
            "v_template": vertices,
            "shapedirs": _mesh_shape_directions(joint_directions),
            "J_regressor": regressor,
            "weights": weights,
            "posedirs": np.zeros((vertices.shape[0], 3, (NUM_JOINTS - 1) * 9)),
            "faces": faces,
        }
    return Model(
        v_template=mesh.get("v_template", rest),
        shapedirs=mesh.get("shapedirs", joint_directions),
        J_regressor=mesh.get("J_regressor", np.eye(NUM_JOINTS)),
        kintree_parents=np.array(PARENTS, dtype=np.int64),
        weights=mesh.get("weights"),
        posedirs=mesh.get("posedirs"),
        faces=mesh.get("faces"),
        gender=gender,
        stand_in=True,
    )


def write_models(directory: str | Path, *, with_mesh: bool = False) -> list[Path]:
    """Write the stand-in under the three model-set file names in ``directory``.

    The files are marked as stand-ins; see the module docstring before pointing a real
    conversion at them. ``with_mesh`` adds the blocky surface, which a render needs.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    written = []
    for gender in GENDERS:
        model = demo_model(gender, with_mesh=with_mesh)
        path = directory / model_filename(gender)
        arrays = {
            "v_template": model.v_template,
            "shapedirs": model.shapedirs,
            "J_regressor": model.J_regressor,
            "kintree_parents": model.kintree_parents,
        }
        if with_mesh:
            arrays.update(weights=model.weights, posedirs=model.posedirs, faces=model.faces)
        np.savez(
            path,
            **arrays,
            **{STAND_IN_KEY: np.array("stand-in body for trying the pipeline; not an SMPL model")},
        )
        written.append(path)
    return written
