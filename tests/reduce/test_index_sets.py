"""The reduction's index sets: 18 kept, no overlap, ascending, and each absorber below its joint.

These four tuples decide what a corpus stores. A quiet edit to one of them would change every
bundle written afterwards while every other test still passed, so they are pinned here against
the tree itself and against docs/primer.md section 5.
"""

from __future__ import annotations

import pytest

from smpl18.skeleton import definition as d


def test_the_frozen_joints_are_the_two_lumbar_joints_and_the_two_collars():
    assert d.FROZEN_JOINTS == (3, 6, 13, 14)
    assert d.FROZEN_JOINT_NAMES == ("spine1", "spine2", "left_collar", "right_collar")


def test_the_dropped_joints_are_the_hands():
    assert d.DROPPED_JOINTS == d.HAND_JOINTS == (22, 23)


def test_eighteen_joints_are_kept_ascending_and_without_overlap():
    assert len(d.KEEP18) == 18
    assert list(d.KEEP18) == sorted(d.KEEP18)
    assert len(set(d.KEEP18)) == 18
    assert not set(d.KEEP18) & set(d.FROZEN_JOINTS)
    assert not set(d.KEEP18) & set(d.DROPPED_JOINTS)
    assert set(d.KEEP18) | set(d.FROZEN_JOINTS) | set(d.DROPPED_JOINTS) == set(range(24))
    assert d.KEEP18 == (0, 1, 2, 4, 5, 7, 8, 9, 10, 11, 12, 15, 16, 17, 18, 19, 20, 21)


def test_the_names_follow_the_indices():
    assert d.JOINT18_NAMES == tuple(d.JOINT_NAMES[j] for j in d.KEEP18)
    assert d.JOINT18_NAMES[7] == "spine3" and "left_collar" not in d.JOINT18_NAMES
    assert "left_hand" not in d.JOINT18_NAMES and "right_hand" not in d.JOINT18_NAMES


def test_each_absorber_is_a_kept_joint_directly_below_its_frozen_joint():
    assert set(d.ABSORBERS) == set(d.FROZEN_JOINTS)
    for frozen, absorber in d.ABSORBERS.items():
        assert absorber in d.KEEP18
        node = d.PARENTS[absorber]
        while node != frozen and node >= 0:
            node = d.PARENTS[node]
        assert node == frozen, f"{d.JOINT_NAMES[absorber]} is not below {d.JOINT_NAMES[frozen]}"
    assert d.ABSORBERS[3] == d.ABSORBERS[6] == 9          # spine3 absorbs both lumbar joints
    assert (d.ABSORBERS[13], d.ABSORBERS[14]) == (16, 17)


def test_the_affected_joints_are_the_kept_joints_below_a_frozen_one():
    assert d.AFFECTED_BY_FREEZE == (9, 12, 15, 16, 17, 18, 19, 20, 21)
    assert set(d.AFFECTED_BY_FREEZE) <= set(d.KEEP18)
    for joint in d.KEEP18:
        ancestors = []
        node = d.PARENTS[joint]
        while node >= 0:
            ancestors.append(node)
            node = d.PARENTS[node]
        below = bool(set(ancestors) & set(d.FROZEN_JOINTS))
        assert below == (joint in d.AFFECTED_BY_FREEZE)


def test_the_sets_are_immutable():
    for values in (d.FROZEN_JOINTS, d.DROPPED_JOINTS, d.KEEP18, d.AFFECTED_BY_FREEZE):
        assert isinstance(values, tuple)
    with pytest.raises(TypeError):
        d.ABSORBERS[3] = 12
