"""The joint table is pinned to docs/primer.md section 2.1 and is a well-formed tree."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from smpl18.skeleton import definition as d

PRIMER = Path(__file__).resolve().parents[2] / "docs" / "primer.md"


def _primer_joint_table() -> list[tuple[int, str, int]]:
    """``(index, name, parent)`` rows of the table under '### 2.1 Joint table'."""
    text = PRIMER.read_text(encoding="utf-8")
    section = text.split("### 2.1 Joint table", 1)[1].split("### 2.2", 1)[0]
    rows = []
    for line in section.splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) < 3 or not re.fullmatch(r"\d+", cells[0]):
            continue
        parent = -1 if cells[2] in ("—", "-", "") else int(cells[2])
        rows.append((int(cells[0]), cells[1], parent))
    return rows


def test_the_table_matches_the_primer():
    rows = _primer_joint_table()
    assert len(rows) == 24
    assert [index for index, _, _ in rows] == list(range(24))
    assert tuple(name for _, name, _ in rows) == d.JOINT_NAMES
    assert tuple(parent for _, _, parent in rows) == d.PARENTS


def test_the_tree_is_topologically_ordered_from_one_root():
    assert d.NUM_JOINTS == 24 and d.ROOT == 0 and d.PARENTS[0] == -1
    assert all(0 <= d.PARENTS[j] < j for j in range(1, 24))
    assert len(set(d.JOINT_NAMES)) == 24


def test_children_invert_parents():
    for joint in range(24):
        assert d.children_of(joint) == tuple(c for c in range(24) if d.PARENTS[c] == joint)
    assert d.children_of(9) == (12, 13, 14)
    assert d.children_of(0) == (1, 2, 3)
    assert d.children_of(22) == () and d.children_of(23) == ()


def test_body_subset_leaves_out_the_two_hands():
    assert d.HAND_JOINTS == (22, 23)
    assert len(d.BODY_JOINTS) == 22 and 22 not in d.BODY_JOINTS and 23 not in d.BODY_JOINTS
    assert d.BODY_JOINT_NAMES == tuple(d.JOINT_NAMES[j] for j in d.BODY_JOINTS)
    assert "left_hand" not in d.BODY_JOINT_NAMES


def test_segments_run_from_an_ancestor_to_a_descendant():
    assert len(d.SEGMENTS) == 13 and len(set(d.SEGMENT_NAMES)) == 13
    for name, proximal, distal in d.SEGMENTS:
        node = distal
        while node != proximal:
            node = d.PARENTS[node]
            assert node >= 0, f"{name}: {proximal} is not above {distal}"
    assert ("trunk", 0, 9) in d.SEGMENTS
    assert all(22 not in seg[1:] and 23 not in seg[1:] for seg in d.SEGMENTS)


def test_names_resolve_to_indices():
    assert d.index_of("pelvis") == 0 and d.index_of("right_hand") == 23
    with pytest.raises(ValueError, match="no SMPL-24 joint"):
        d.index_of("torso")


def test_the_parents_array_is_read_only_and_matches_the_tuple():
    assert d.PARENTS_ARRAY.dtype == np.int64
    assert tuple(d.PARENTS_ARRAY) == d.PARENTS
    with pytest.raises(ValueError):
        d.PARENTS_ARRAY[0] = 5
