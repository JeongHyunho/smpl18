"""Synthetic motion, and what a capture of it would record, for the examples and the tests.

No conversion uses this module. It exists so that every converter can be run, and checked,
without motion data: a walking-like SMPL-24 motion with a known answer, the labelled markers the
shipped ``conventional_full_body`` marker set expects, joint-centre trajectories, an OpenSim model
with a coordinate file, and a BVH clip.

The marker placement is built backwards from the marker set's own rules -- the hip regression,
the chords through the wands, the elbow hinge -- so the rules give the joint centres back
exactly. Real markers ride on skin and do not; the examples say so where it matters.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from smpl18.skeleton.definition import JOINT_NAMES, NUM_JOINTS, PARENTS
from smpl18.skeleton.kinematics import fk_batch
from smpl18.skeleton.rotations import axis_angle_to_matrix, matrix_to_axis_angle

__all__ = [
    "BVH_NAMES",
    "CENTRE_LABELS",
    "MEASUREMENTS",
    "conventional_markers",
    "joint_centres",
    "opensim_coordinates",
    "opensim_motion",
    "rotations_to_axis_angle",
    "walking_motion",
    "write_bvh",
    "write_mot",
    "write_opensim_model",
]

#: Subject measurements for the conventional marker set, in metres; ordinary adult values.
MEASUREMENTS: dict[str, float] = {
    "marker_radius": 0.007,
    "leg_length_left": 0.86, "leg_length_right": 0.86,
    "knee_width_left": 0.10, "knee_width_right": 0.10,
    "ankle_width_left": 0.07, "ankle_width_right": 0.07,
    "elbow_width_left": 0.065, "elbow_width_right": 0.065,
    "shoulder_offset_left": 0.035, "shoulder_offset_right": 0.035,
}

#: Joint-centre labels of the kind motion-analysis software exports, per SMPL joint.
CENTRE_LABELS: dict[str, str] = {
    "left_hip": "LHJC", "right_hip": "RHJC", "left_knee": "LKJC", "right_knee": "RKJC",
    "left_ankle": "LAJC", "right_ankle": "RAJC", "left_foot": "LTOE", "right_foot": "RTOE",
    "spine3": "THORAX", "neck": "NECK", "head": "HEAD",
    "left_shoulder": "LSJC", "right_shoulder": "RSJC", "left_elbow": "LEJC",
    "right_elbow": "REJC", "left_wrist": "LWJC", "right_wrist": "RWJC",
}

_X, _Y, _Z = np.eye(3)


def _about(axis: np.ndarray, angle: np.ndarray) -> np.ndarray:
    return axis_angle_to_matrix(np.asarray(angle)[..., None] * axis)


def walking_motion(frames: int, fps: float, *, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """``(T, 24, 3, 3)`` local rotations and ``(T, 3)`` translation of a walking-like motion.

    One stride per second along a slowly turning heading, with arm swing and elbows that never
    straighten (a straight elbow leaves its hinge axis undefined). ``seed`` shifts the phases so
    that two trials of one subject differ.
    """
    rng = np.random.default_rng(seed)
    time = np.arange(frames) / fps
    phase = 2.0 * np.pi * time + rng.uniform(0, 2.0 * np.pi)
    local = np.tile(np.eye(3), (frames, NUM_JOINTS, 1, 1))
    index = {name: i for i, name in enumerate(JOINT_NAMES)}

    heading = 0.25 * time + rng.uniform(-np.pi, np.pi)
    local[:, 0] = (_about(_Y, heading) @ _about(_X, 0.04 * np.sin(2 * phase))
                   @ _about(_Z, 0.05 * np.sin(phase)))
    for side, shift in (("left", 0.0), ("right", np.pi)):
        sign = 1.0 if side == "left" else -1.0
        step = phase + shift
        local[:, index[f"{side}_hip"]] = (_about(_X, -0.4 * np.sin(step))
                                          @ _about(_Z, sign * 0.05 * np.sin(step))
                                          @ _about(_Y, sign * 0.08 * np.sin(step + 0.4)))
        local[:, index[f"{side}_knee"]] = (_about(_X, 0.2 + 0.35 * (1 + np.sin(step - 1.2)))
                                           @ _about(_Y, 0.05 * np.sin(step)))
        local[:, index[f"{side}_ankle"]] = (_about(_X, 0.15 * np.sin(step + 0.3))
                                            @ _about(_Z, 0.05 * np.cos(step)))
        local[:, index[f"{side}_foot"]] = _about(_X, 0.1 * np.sin(step + 1.0))
        local[:, index[f"{side}_collar"]] = _about(_Z, sign * 0.05 * np.sin(step))
        local[:, index[f"{side}_shoulder"]] = (_about(_X, 0.35 * np.sin(step))
                                               @ _about(_Z, -sign * 1.15)
                                               @ _about(_X, 0.3))
        local[:, index[f"{side}_elbow"]] = _about(_Y, -sign * (0.7 + 0.3 * np.sin(step + 0.5)))
        # The wrist flexes and deviates; it does not twist about the hand's own axis.
        local[:, index[f"{side}_wrist"]] = _about(_Z, 0.2 * np.sin(step)) @ _about(_Y, 0.1)
    for name, amplitude in (("spine1", 0.04), ("spine2", 0.04), ("spine3", 0.05)):
        local[:, index[name]] = (_about(_Y, amplitude * np.sin(phase))
                                 @ _about(_X, 0.03 + 0.02 * np.sin(2 * phase)))
    local[:, index["neck"]] = _about(_X, 0.1 + 0.05 * np.sin(0.5 * phase))
    local[:, index["head"]] = _about(_Y, 0.3 * np.sin(0.3 * phase)) @ _about(_X, -0.05)

    forward = np.stack([np.sin(heading), np.zeros(frames), np.cos(heading)], axis=1)
    trans = np.cumsum(forward * 1.2 / fps, axis=0)
    trans[:, 1] = 0.02 * np.sin(2 * phase)
    return local, trans


def joint_centres(rest, local, trans, joints: Mapping[str, str] = CENTRE_LABELS):
    """``(labels, (T, K, 3))``: the forward-kinematics centres of ``joints``, under their labels."""
    positions, _ = fk_batch(rest, local, trans)
    labels = tuple(joints.values())
    columns = [JOINT_NAMES.index(name) for name in joints]
    return labels, positions[:, columns]


def _unit(vectors: np.ndarray) -> np.ndarray:
    return vectors / np.linalg.norm(vectors, axis=-1, keepdims=True)


def _across(vectors: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """The part of ``vectors`` perpendicular to the unit ``axis``."""
    return vectors - np.sum(vectors * axis, axis=-1, keepdims=True) * axis


def _chord_markers(proximal, centre, lateral_hint, distance, wand_offset):
    """The lateral marker and the wand for which the chord rule returns ``centre`` exactly."""
    outward = _unit(_across(lateral_hint, _unit(proximal - centre)))
    lateral = centre + distance * outward
    along = _unit(proximal - lateral)
    towards_centre = _unit(_across(centre - lateral, along))
    wand = lateral + 0.5 * (proximal - lateral) - wand_offset * towards_centre
    return lateral, wand


def conventional_markers(rest, local, trans, measurements: Mapping[str, float] = MEASUREMENTS):
    """``(labels, (T, M, 3))`` markers of the ``conventional_full_body`` set, in metres.

    Left and right leg lengths must agree: the hip regression then places both hips from one
    pelvis frame, which the SMPL hips allow only when they are mirror images.
    """
    if not math.isclose(measurements["leg_length_left"], measurements["leg_length_right"]):
        raise ValueError("synthetic markers need equal left and right leg lengths")
    p, g = fk_batch(rest, local, trans)
    j = {name: i for i, name in enumerate(JOINT_NAMES)}
    radius = measurements["marker_radius"]
    out: dict[str, np.ndarray] = {}

    # Pelvis: the conventional frame has x forward, y left, z up; SMPL's has x left, y up, z
    # forward. The regression puts each hip at (x, +/-y, z) in that frame.
    to_conventional = np.column_stack([_Z, _X, _Y])
    frame = g[:, j["pelvis"]] @ to_conventional
    leg = measurements["leg_length_left"]
    c = 0.115 * leg - 0.0153
    a = 0.1288 * leg - 0.04856
    theta, beta = 0.5, 0.314
    x = c * math.cos(theta) * math.sin(beta) - (a + radius) * math.cos(beta)
    z = -c * math.cos(theta) * math.cos(beta) - (a + radius) * math.sin(beta)
    hips = 0.5 * (p[:, j["left_hip"]] + p[:, j["right_hip"]])
    half = 0.5 * np.linalg.norm(p[:, j["left_hip"]] - p[:, j["right_hip"]], axis=1)
    spine = half + c * math.sin(theta)
    origin = hips - frame @ np.array([x, 0.0, z])
    lateral_axis, forward_axis = frame[:, :, 1], frame[:, :, 0]
    out["LASI"] = origin + spine[:, None] * lateral_axis
    out["RASI"] = origin - spine[:, None] * lateral_axis
    out["LPSI"] = origin - 0.16 * forward_axis + 0.05 * lateral_axis
    out["RPSI"] = origin - 0.16 * forward_axis - 0.05 * lateral_axis

    for side, letter, sign in (("left", "L", 1.0), ("right", "R", -1.0)):
        hip, knee, ankle = p[:, j[f"{side}_hip"]], p[:, j[f"{side}_knee"]], p[:, j[f"{side}_ankle"]]
        out[f"{letter}KNE"], out[f"{letter}THI"] = _chord_markers(
            hip, knee, sign * g[:, j[f"{side}_hip"], :, 0],
            0.5 * measurements[f"knee_width_{side}"] + radius, 0.06,
        )
        out[f"{letter}ANK"], out[f"{letter}TIB"] = _chord_markers(
            knee, ankle, sign * g[:, j[f"{side}_knee"], :, 0],
            0.5 * measurements[f"ankle_width_{side}"] + radius, 0.06,
        )
        forward = g[:, j[f"{side}_ankle"], :, 2]
        toe = p[:, j[f"{side}_foot"]]
        heel = toe - 0.18 * forward
        up = _unit(_across(ankle - 0.5 * (toe + heel), forward))
        out[f"{letter}TOE"] = toe + radius * up
        out[f"{letter}HEE"] = heel + radius * up

    spine3, neck, head = p[:, j["spine3"]], p[:, j["neck"]], p[:, j["head"]]
    ahead = g[:, j["spine3"], :, 2]
    out["STRN"], out["T10"] = spine3 + 0.10 * ahead, spine3 - 0.10 * ahead
    out["CLAV"], out["C7"] = neck + 0.07 * ahead, neck - 0.07 * ahead
    trunk_up = _unit(neck - spine3)
    head_forward, head_left = g[:, j["head"], :, 2], g[:, j["head"], :, 0]
    for label, f, s in (("LFHD", 1, 1), ("RFHD", 1, -1), ("LBHD", -1, 1), ("RBHD", -1, -1)):
        out[label] = head + 0.09 * f * head_forward + 0.07 * s * head_left

    for side, letter, sign in (("left", "L", 1.0), ("right", "R", -1.0)):
        shoulder = p[:, j[f"{side}_shoulder"]]
        elbow, wrist = p[:, j[f"{side}_elbow"]], p[:, j[f"{side}_wrist"]]
        out[f"{letter}SHO"] = shoulder + (
            measurements[f"shoulder_offset_{side}"] + radius
        ) * trunk_up
        axis = sign * _unit(np.cross(shoulder - elbow, wrist - elbow))
        out[f"{letter}ELB"] = elbow + (0.5 * measurements[f"elbow_width_{side}"] + radius) * axis
        out[f"{letter}UPA"] = 0.5 * (shoulder + elbow) + 0.04 * axis
        across_wrist = g[:, j[f"{side}_elbow"], :, 2]
        out[f"{letter}WRA"] = wrist + 0.03 * across_wrist
        out[f"{letter}WRB"] = wrist - 0.03 * across_wrist
        out[f"{letter}FIN"] = wrist + 0.08 * sign * g[:, j[f"{side}_wrist"], :, 0]

    labels = tuple(out)
    return labels, np.stack([out[label] for label in labels], axis=1)


# --- an OpenSim model and its coordinates ---------------------------------------------------------

#: Bodies of a Rajagopal-style model, each with its joint, its parent body and the SMPL joint
#: whose rest position places the joint centre. Three rotations per joint, so the model can
#: follow the synthetic motion closely; real models lock or couple some of them.
_OPENSIM_JOINTS: tuple[tuple[str, str, str, str], ...] = (
    ("ground_pelvis", "ground", "pelvis", "pelvis"),
    ("hip_l", "pelvis", "femur_l", "left_hip"),
    ("walker_knee_l", "femur_l", "tibia_l", "left_knee"),
    ("ankle_l", "tibia_l", "talus_l", "left_ankle"),
    ("subtalar_l", "talus_l", "calcn_l", "left_ankle"),
    ("mtp_l", "calcn_l", "toes_l", "left_foot"),
    ("hip_r", "pelvis", "femur_r", "right_hip"),
    ("walker_knee_r", "femur_r", "tibia_r", "right_knee"),
    ("ankle_r", "tibia_r", "talus_r", "right_ankle"),
    ("subtalar_r", "talus_r", "calcn_r", "right_ankle"),
    ("mtp_r", "calcn_r", "toes_r", "right_foot"),
    ("back", "pelvis", "torso", "spine1"),
    ("acromial_l", "torso", "humerus_l", "left_shoulder"),
    ("elbow_l", "humerus_l", "ulna_l", "left_elbow"),
    ("radioulnar_l", "ulna_l", "radius_l", "left_elbow"),
    ("radius_hand_l", "radius_l", "hand_l", "left_wrist"),
    ("acromial_r", "torso", "humerus_r", "right_shoulder"),
    ("elbow_r", "humerus_r", "ulna_r", "right_elbow"),
    ("radioulnar_r", "ulna_r", "radius_r", "right_elbow"),
    ("radius_hand_r", "radius_r", "hand_r", "right_wrist"),
)
#: Joints that only carry the next body; a weld keeps the chain.
_OPENSIM_WELDS = {"subtalar_l", "subtalar_r", "radioulnar_l", "radioulnar_r"}

#: OpenSim's axes (x forward, y up, z right) in SMPL's (x left, y up, z forward).
_SMPL_TO_OPENSIM = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])


def _vector(values) -> str:
    return " ".join(f"{float(v):.9g}" for v in values)


def _custom_joint(name, parent, child, offset, coordinates, translations):
    axes = []
    for index, (coordinate, axis) in enumerate(zip(coordinates, ("1 0 0", "0 1 0", "0 0 1"))):
        axes.append(
            f'<TransformAxis name="rotation{index + 1}"><coordinates>{coordinate}</coordinates>'
            f"<axis>{axis}</axis><LinearFunction><coefficients>1 0</coefficients>"
            "</LinearFunction></TransformAxis>"
        )
    for index, axis in enumerate(("1 0 0", "0 1 0", "0 0 1")):
        coordinate = translations[index] if translations else ""
        function = ("<LinearFunction><coefficients>1 0</coefficients></LinearFunction>"
                    if translations else "<Constant><value>0</value></Constant>")
        axes.append(
            f'<TransformAxis name="translation{index + 1}"><coordinates>{coordinate}'
            f"</coordinates><axis>{axis}</axis>{function}</TransformAxis>"
        )
    declared = "".join(
        f'<Coordinate name="{c}"><default_value>0</default_value></Coordinate>'
        for c in (*coordinates, *(translations or ()))
    )
    return (
        f'<CustomJoint name="{name}">'
        f"<socket_parent_frame>{parent}_offset</socket_parent_frame>"
        f"<socket_child_frame>{child}_offset</socket_child_frame>"
        f"<coordinates>{declared}</coordinates>"
        "<frames>"
        f'<PhysicalOffsetFrame name="{parent}_offset">'
        f"<socket_parent>/{'ground' if parent == 'ground' else 'bodyset/' + parent}"
        f"</socket_parent><translation>{_vector(offset)}</translation>"
        "<orientation>0 0 0</orientation></PhysicalOffsetFrame>"
        f'<PhysicalOffsetFrame name="{child}_offset">'
        f"<socket_parent>/bodyset/{child}</socket_parent>"
        "<translation>0 0 0</translation><orientation>0 0 0</orientation>"
        "</PhysicalOffsetFrame>"
        "</frames>"
        f"<SpatialTransform>{''.join(axes)}</SpatialTransform>"
        "</CustomJoint>"
    )


def _weld_joint(name, parent, child):
    return (
        f'<WeldJoint name="{name}">'
        f"<socket_parent_frame>{parent}_offset</socket_parent_frame>"
        f"<socket_child_frame>{child}_offset</socket_child_frame>"
        "<frames>"
        f'<PhysicalOffsetFrame name="{parent}_offset"><socket_parent>/bodyset/{parent}'
        "</socket_parent><translation>0 0 0</translation><orientation>0 0 0</orientation>"
        "</PhysicalOffsetFrame>"
        f'<PhysicalOffsetFrame name="{child}_offset"><socket_parent>/bodyset/{child}'
        "</socket_parent><translation>0 0 0</translation><orientation>0 0 0</orientation>"
        "</PhysicalOffsetFrame>"
        "</frames></WeldJoint>"
    )


def opensim_coordinates(joint: str) -> tuple[str, ...]:
    """The three rotation coordinates of a synthetic model joint (none for a weld)."""
    if joint in _OPENSIM_WELDS:
        return ()
    return tuple(f"{joint}_r{axis}" for axis in "xyz")


def write_opensim_model(path: str | Path, rest: np.ndarray) -> Path:
    """An OpenSim 4 model with Rajagopal-style names whose joint centres sit at ``rest``.

    Bodies are placed so that at zero coordinates each joint centre is the SMPL rest position of
    the joint named in the table above, expressed in OpenSim's axes.
    """
    rest = np.asarray(rest, dtype=np.float64) @ _SMPL_TO_OPENSIM.T
    centre = {smpl: rest[JOINT_NAMES.index(smpl)] for *_, smpl in _OPENSIM_JOINTS}
    body_origin: dict[str, np.ndarray] = {"ground": np.zeros(3)}
    bodies, joints = [], []
    for name, parent, child, smpl in _OPENSIM_JOINTS:
        offset = centre[smpl] - body_origin[parent]
        body_origin[child] = centre[smpl]
        bodies.append(
            f'<Body name="{child}"><mass>1</mass><mass_center>0 0 0</mass_center>'
            "<inertia>0.01 0.01 0.01 0 0 0</inertia></Body>"
        )
        if name in _OPENSIM_WELDS:
            joints.append(_weld_joint(name, parent, child))
        else:
            translations = ("pelvis_tx", "pelvis_ty", "pelvis_tz") if name == "ground_pelvis" else None
            joints.append(_custom_joint(name, parent, child, offset,
                                        opensim_coordinates(name), translations))
    text = (
        '<?xml version="1.0" encoding="UTF-8" ?>\n'
        '<OpenSimDocument Version="40000"><Model name="synthetic_rajagopal_style">'
        "<gravity>0 -9.80665 0</gravity>"
        '<Ground name="ground"/>'
        f"<BodySet><objects>{''.join(bodies)}</objects></BodySet>"
        f"<JointSet><objects>{''.join(joints)}</objects></JointSet>"
        "</Model></OpenSimDocument>\n"
    )
    path = Path(path)
    path.write_text(text, encoding="utf-8")
    return path


def _xyz_angles(rotations: np.ndarray) -> np.ndarray:
    """Body-fixed x-y-z angles ``(T, 3)`` of ``(T, 3, 3)`` rotations: ``R = Rx Ry Rz``."""
    from scipy.spatial.transform import Rotation

    return Rotation.from_matrix(rotations).as_euler("XYZ")


def opensim_motion(frames: int, fps: float, *, seed: int = 0) -> dict[str, np.ndarray]:
    """Coordinate values (radians, metres) that walk the synthetic OpenSim model.

    The walking motion is taken joint by joint into OpenSim's axes; the trunk, one body in the
    model, takes the three spine joints' combined turn.
    """
    local, trans = walking_motion(frames, fps, seed=seed)
    change = _SMPL_TO_OPENSIM
    index = {name: i for i, name in enumerate(JOINT_NAMES)}
    values: dict[str, np.ndarray] = {}
    turns = {
        "ground_pelvis": local[:, 0],
        "back": local[:, index["spine1"]] @ local[:, index["spine2"]] @ local[:, index["spine3"]],
    }
    for name, _parent, _child, smpl in _OPENSIM_JOINTS:
        if name in _OPENSIM_WELDS or name in turns:
            continue
        turns[name] = local[:, index[smpl]]
    for name, rotation in turns.items():
        angles = _xyz_angles(change @ rotation @ change.T)
        for column, coordinate in enumerate(opensim_coordinates(name)):
            values[coordinate] = angles[:, column]
    pelvis = trans @ change.T
    for column, coordinate in enumerate(("pelvis_tx", "pelvis_ty", "pelvis_tz")):
        values[coordinate] = pelvis[:, column]
    return values



def write_mot(path: str | Path, values: Mapping[str, np.ndarray], fps: float, *,
              in_degrees: bool) -> Path:
    """An OpenSim coordinate file. Rotations are given in radians and written in degrees when
    ``in_degrees``; translation coordinates (names ending ``_tx``/``_ty``/``_tz``) never are."""
    names = list(values)
    frames = len(next(iter(values.values())))
    columns = []
    for name in names:
        column = np.asarray(values[name], dtype=np.float64)
        if in_degrees and not name.endswith(("_tx", "_ty", "_tz")):
            column = np.degrees(column)
        columns.append(column)
    table = np.column_stack([np.arange(frames) / fps, *columns])
    lines = [
        Path(path).name,
        "version=1",
        f"nRows={frames}",
        f"nColumns={len(names) + 1}",
        f"inDegrees={'yes' if in_degrees else 'no'}",
        "endheader",
        "\t".join(["time", *names]),
    ]
    lines.extend("\t".join(f"{value:.9g}" for value in row) for row in table)
    path = Path(path)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# --- a BVH clip ------------------------------------------------------------------------------------

#: Joint names of a common humanoid BVH rig, per SMPL joint, in hierarchy order.
BVH_NAMES: dict[str, str] = {
    "pelvis": "Hips", "left_hip": "LeftUpLeg", "left_knee": "LeftLeg", "left_ankle": "LeftFoot",
    "left_foot": "LeftToeBase", "right_hip": "RightUpLeg", "right_knee": "RightLeg",
    "right_ankle": "RightFoot", "right_foot": "RightToeBase", "spine1": "Spine",
    "spine2": "Spine1", "spine3": "Spine2", "neck": "Neck", "head": "Head",
    "left_collar": "LeftShoulder", "left_shoulder": "LeftArm", "left_elbow": "LeftForeArm",
    "left_wrist": "LeftHand", "right_collar": "RightShoulder", "right_shoulder": "RightArm",
    "right_elbow": "RightForeArm", "right_wrist": "RightHand",
}


def _bvh_order() -> list[int]:
    order: list[int] = []

    def visit(joint: int) -> None:
        order.append(joint)
        for child in range(NUM_JOINTS):
            if PARENTS[child] == joint and JOINT_NAMES[child] in BVH_NAMES:
                visit(child)

    visit(0)
    return order


def write_bvh(path: str | Path, rest: np.ndarray, local: np.ndarray, trans: np.ndarray,
              fps: float, *, units_per_metre: float) -> Path:
    """A BVH clip of an SMPL-24 motion: the skeleton is the rest pose, one ZXY rotation triple
    per joint, root positions in the file's units (``units_per_metre``, e.g. 100 for cm)."""
    rest = np.asarray(rest, dtype=np.float64)
    order = _bvh_order()
    depth = {0: 0}
    lines = ["HIERARCHY"]
    stack: list[int] = []

    def close_until(level: int) -> None:
        while len(stack) > level:
            joint = stack.pop()
            indent = "  " * depth[joint]
            if not any(PARENTS[c] == joint and JOINT_NAMES[c] in BVH_NAMES for c in range(NUM_JOINTS)):
                lines.append(f"{indent}  End Site")
                lines.append(f"{indent}  {{")
                lines.append(f"{indent}    OFFSET 0 0 0")
                lines.append(f"{indent}  }}")
            lines.append(f"{indent}}}")

    for joint in order:
        parent = PARENTS[joint]
        if joint:
            depth[joint] = depth[parent] + 1
            close_until(depth[joint])
        indent = "  " * depth[joint]
        name = BVH_NAMES[JOINT_NAMES[joint]]
        if joint == 0:
            # The root's offset is zero and its position channels carry the absolute position.
            lines.append(f"ROOT {name}")
            offset = np.zeros(3)
            channels = "CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation"
        else:
            lines.append(f"{indent}JOINT {name}")
            offset = (rest[joint] - rest[parent]) * units_per_metre
            channels = "CHANNELS 3 Zrotation Xrotation Yrotation"
        lines.append(f"{indent}{{")
        lines.append(f"{indent}  OFFSET {_vector(offset)}")
        lines.append(f"{indent}  {channels}")
        stack.append(joint)
    close_until(0)

    from scipy.spatial.transform import Rotation

    frames = local.shape[0]
    rows = []
    root = (np.asarray(trans) + rest[0]) * units_per_metre
    for joint in order:
        angles = np.degrees(Rotation.from_matrix(local[:, joint]).as_euler("ZXY"))
        rows.append(angles)
    body = np.concatenate([root] + rows, axis=1)
    lines.append("MOTION")
    lines.append(f"Frames: {frames}")
    lines.append(f"Frame Time: {1.0 / fps:.8f}")
    lines.extend(" ".join(f"{value:.6f}" for value in row) for row in body)
    path = Path(path)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def rotations_to_axis_angle(local: np.ndarray) -> np.ndarray:
    """``(T, 24, 3)`` axis-angle of ``(T, 24, 3, 3)`` rotations, for writing SMPL parameters."""
    return matrix_to_axis_angle(local)
