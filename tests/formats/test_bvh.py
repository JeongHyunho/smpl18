"""Reading a Biovision Hierarchy file: the tree, the channels in their own order, the rows."""

from __future__ import annotations

import numpy as np
import pytest

from smpl24.formats import FormatError, bvh

# A three-level tree with one End Site on each leaf, the root carrying six channels in the
# usual position-then-rotation order and the joints three rotations in a different order each.
SMALL_BVH = """HIERARCHY
ROOT Hips
{
    OFFSET 0.0 0.0 0.0
    CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation
    JOINT Spine
    {
        OFFSET 0.0 10.5 0.0
        CHANNELS 3 Zrotation Xrotation Yrotation
        End Site
        {
            OFFSET 0.0 20.0 0.0
        }
    }
    JOINT LeftUpLeg
    {
        OFFSET 9.0 -1.0 0.0
        CHANNELS 3 Xrotation Yrotation Zrotation
        JOINT LeftLeg
        {
            OFFSET 0.0 -40.0 0.0
            CHANNELS 3 Zrotation Yrotation Xrotation
            End Site
            {
                OFFSET 0.0 -38.0 0.0
            }
        }
    }
}
MOTION
Frames: 2
Frame Time: 0.0333333
1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
15 14 13 12 11 10 9 8 7 6 5 4 3 2 1
"""


@pytest.fixture(scope="module")
def clip():
    return bvh.parse(SMALL_BVH)


def test_joints_come_in_file_order_with_their_parents(clip):
    assert clip.names == ("Hips", "Spine", "LeftUpLeg", "LeftLeg")
    assert clip.parents == (-1, 0, 0, 2)


def test_offsets_and_end_sites_are_read(clip):
    by_name = {j.name: j for j in clip.joints}
    assert by_name["Spine"].offset == (0.0, 10.5, 0.0)
    assert by_name["LeftUpLeg"].offset == (9.0, -1.0, 0.0)
    assert by_name["Spine"].end_site == (0.0, 20.0, 0.0)
    assert by_name["LeftLeg"].end_site == (0.0, -38.0, 0.0)
    assert by_name["Hips"].end_site is None
    assert by_name["LeftUpLeg"].end_site is None


def test_channels_keep_the_order_each_joint_declares(clip):
    by_name = {j.name: j for j in clip.joints}
    assert by_name["Hips"].channels == (
        "Xposition", "Yposition", "Zposition", "Zrotation", "Xrotation", "Yrotation",
    )
    assert by_name["Spine"].channels == ("Zrotation", "Xrotation", "Yrotation")
    assert by_name["LeftLeg"].channels == ("Zrotation", "Yrotation", "Xrotation")
    assert [j.channel_start for j in clip.joints] == [0, 6, 9, 12]
    assert clip.channel_count == 15


def test_frames_are_rows_of_every_channel_in_hierarchy_order(clip):
    assert clip.frame_count == 2
    assert clip.frame_time_s == pytest.approx(0.0333333)
    assert clip.frames.shape == (2, 15)
    assert clip.frames.dtype == np.float64
    np.testing.assert_array_equal(clip.frames[0], np.arange(1, 16))
    np.testing.assert_array_equal(clip.frames[1], np.arange(15, 0, -1))


def test_a_channel_is_addressed_by_joint_and_name(clip):
    assert clip.channel_index("Hips", "Zrotation") == 3
    assert clip.channel_index("LeftLeg", "Xrotation") == 14
    np.testing.assert_array_equal(clip.channels_of("Spine")[0], [7, 8, 9])
    with pytest.raises(KeyError, match="no joint named"):
        clip.channel_index("RightLeg", "Xrotation")
    with pytest.raises(KeyError, match="declares no channel"):
        clip.channel_index("Spine", "Xposition")


def test_a_row_with_the_wrong_width_is_refused_by_row():
    broken = SMALL_BVH.replace("15 14 13 12 11 10 9 8 7 6 5 4 3 2 1", "15 14 13")
    with pytest.raises(FormatError, match="motion row 1 has 3 values against 15"):
        bvh.parse(broken)


def test_a_frame_count_that_disagrees_with_the_rows_is_refused():
    broken = SMALL_BVH.replace("Frames: 2", "Frames: 3")
    with pytest.raises(FormatError, match="declares 3 rows, the file carries 2"):
        bvh.parse(broken)


def test_an_unknown_channel_name_is_refused():
    broken = SMALL_BVH.replace("CHANNELS 3 Zrotation Xrotation Yrotation",
                               "CHANNELS 3 Zrotation Xrotation Wrotation")
    with pytest.raises(FormatError, match="Wrotation"):
        bvh.parse(broken)


def test_a_missing_motion_section_or_offset_is_refused():
    with pytest.raises(FormatError, match="MOTION"):
        bvh.parse(SMALL_BVH[: SMALL_BVH.index("MOTION")])
    no_offset = SMALL_BVH.replace("        OFFSET 0.0 10.5 0.0\n", "")
    with pytest.raises(FormatError, match="Spine.*OFFSET"):
        bvh.parse(no_offset)


def test_read_parses_a_file_the_same_as_the_text(tmp_path):
    path = tmp_path / "clip.bvh"
    path.write_text(SMALL_BVH, encoding="utf-8")
    from_file = bvh.read(path)
    from_text = bvh.parse(SMALL_BVH)
    assert from_file.joints == from_text.joints
    np.testing.assert_array_equal(from_file.frames, from_text.frames)
