"""The SMPL-24 skeleton as constants: joint names, parents, the body subset, the segments.

This is the table in ``docs/primer.md`` section 2.1, pinned by a test. Everything is a tuple
or a read-only array so no caller can edit the tree in place.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

import numpy as np

__all__ = [
    "ABSORBERS",
    "AFFECTED_BY_FREEZE",
    "BODY_JOINTS",
    "BODY_JOINT_NAMES",
    "CHILDREN",
    "DROPPED_JOINTS",
    "FROZEN_JOINTS",
    "FROZEN_JOINT_NAMES",
    "HAND_JOINTS",
    "HINGE_AXES",
    "JOINT18_NAMES",
    "JOINT_NAMES",
    "KEEP18",
    "NUM_JOINTS",
    "PARENTS",
    "PARENTS_ARRAY",
    "ROOT",
    "SEGMENTS",
    "SEGMENT_NAMES",
    "children_of",
    "index_of",
]

NUM_JOINTS = 24
ROOT = 0

JOINT_NAMES: tuple[str, ...] = (
    "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee", "spine2",
    "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot", "neck",
    "left_collar", "right_collar", "head", "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hand", "right_hand",
)

#: Parent index per joint; the root's parent is -1. Every parent precedes its child, so a
#: single pass in index order is a valid tree traversal.
PARENTS: tuple[int, ...] = (
    -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21,
)

#: The two hand joints below the wrists; body-only work leaves their rotation at identity.
HAND_JOINTS: tuple[int, ...] = (22, 23)

#: The 22 body joints: all but the hands.
BODY_JOINTS: tuple[int, ...] = tuple(j for j in range(NUM_JOINTS) if j not in HAND_JOINTS)
BODY_JOINT_NAMES: tuple[str, ...] = tuple(JOINT_NAMES[j] for j in BODY_JOINTS)

#: The four joints a reduced pose does not store per frame (primer section 5): the two lumbar
#: joints and the two clavicles, which carry almost no motion in most capture and which several
#: sources do not drive at all. Each becomes one constant per subject, fitted by ``smpl18.reduce``.
FROZEN_JOINTS: tuple[int, ...] = (3, 6, 13, 14)
FROZEN_JOINT_NAMES: tuple[str, ...] = tuple(JOINT_NAMES[j] for j in FROZEN_JOINTS)

#: The knees and elbows are hinges in the body but ball joints in SMPL. In SMPL's rest frame the
#: knees flex about the local x axis and the elbows (arms out to the sides) about the local y
#: axis; a fit uses this to keep off-axis rotation, which positions cannot see, small.
HINGE_AXES: Mapping[int, int] = MappingProxyType({4: 0, 5: 0, 18: 1, 19: 1})

#: The kept joint just below each frozen one, which takes over its rotation. Freezing a joint
#: would turn the whole chain below it; the absorber undoes exactly that, so no distal world
#: orientation changes. ``spine3`` absorbs both lumbar joints because both sit above it.
ABSORBERS: Mapping[int, int] = MappingProxyType({3: 9, 6: 9, 13: 16, 14: 17})

#: Dropped outright rather than frozen: the hands are not stored at all.
DROPPED_JOINTS: tuple[int, ...] = HAND_JOINTS

#: The 18 joints a reduced pose stores, in ascending index order.
KEEP18: tuple[int, ...] = tuple(
    j for j in range(NUM_JOINTS) if j not in FROZEN_JOINTS and j not in DROPPED_JOINTS
)
JOINT18_NAMES: tuple[str, ...] = tuple(JOINT_NAMES[j] for j in KEEP18)


def _below_a_frozen_joint(joint: int) -> bool:
    node = PARENTS[joint]
    while node >= 0:
        if node in FROZEN_JOINTS:
            return True
        node = PARENTS[node]
    return False


#: The kept joints a constant can move. Absorbing keeps every world orientation exact whatever
#: the constants are, but a frozen joint sits between two segments, so a constant that differs
#: from the joint's real rotation displaces everything below it; these joints pay that cost and
#: are what the constants are fitted against.
AFFECTED_BY_FREEZE: tuple[int, ...] = tuple(j for j in KEEP18 if _below_a_frozen_joint(j))

#: Major segments as (name, proximal joint, distal joint). The trunk spans pelvis to spine3
#: as one segment; the hands are not segments.
SEGMENTS: tuple[tuple[str, int, int], ...] = (
    ("thigh_l", 1, 4), ("thigh_r", 2, 5),
    ("shank_l", 4, 7), ("shank_r", 5, 8),
    ("foot_l", 7, 10), ("foot_r", 8, 11),
    ("trunk", 0, 9), ("neck", 9, 12), ("head", 12, 15),
    ("upperarm_l", 16, 18), ("upperarm_r", 17, 19),
    ("forearm_l", 18, 20), ("forearm_r", 19, 21),
)
SEGMENT_NAMES: tuple[str, ...] = tuple(s[0] for s in SEGMENTS)

#: Children per joint, in index order.
CHILDREN: tuple[tuple[int, ...], ...] = tuple(
    tuple(child for child, parent in enumerate(PARENTS) if parent == joint)
    for joint in range(NUM_JOINTS)
)


def _read_only(values, dtype) -> np.ndarray:
    array = np.array(values, dtype=dtype)
    array.flags.writeable = False
    return array


#: ``PARENTS`` as an int64 array for indexing; read-only.
PARENTS_ARRAY: np.ndarray = _read_only(PARENTS, np.int64)


def children_of(joint: int) -> tuple[int, ...]:
    """Joints whose parent is ``joint``."""
    return CHILDREN[joint]


def index_of(name: str) -> int:
    """Joint index for a name; ``ValueError`` for a name the skeleton does not have."""
    try:
        return JOINT_NAMES.index(name)
    except ValueError:
        raise ValueError(f"no SMPL-24 joint named {name!r}") from None


assert len(JOINT_NAMES) == NUM_JOINTS and len(PARENTS) == NUM_JOINTS
assert PARENTS[ROOT] == -1 and all(0 <= PARENTS[j] < j for j in range(1, NUM_JOINTS))
assert len(set(JOINT_NAMES)) == NUM_JOINTS
assert len(KEEP18) == 18 and list(KEEP18) == sorted(KEEP18)
assert not set(KEEP18) & (set(FROZEN_JOINTS) | set(DROPPED_JOINTS))
assert set(ABSORBERS) == set(FROZEN_JOINTS) and set(ABSORBERS.values()) <= set(KEEP18)
